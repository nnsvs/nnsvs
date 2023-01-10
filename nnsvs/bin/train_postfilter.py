from functools import partial
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from nnsvs.multistream import select_streams
from nnsvs.train_util import (
    collate_fn_default,
    collate_fn_random_segments,
    compute_distortions,
    eval_model,
    log_params_from_omegaconf_dict,
    save_checkpoint,
    save_configs,
    setup_gan,
)
from nnsvs.util import PyTorchStandardScaler, load_vocoder, make_non_pad_mask
from omegaconf import DictConfig
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F


def train_step(
    model_config,
    optim_config,
    netG,
    optG,
    netD,
    optD,
    grad_scaler,
    train,
    in_feats,
    out_feats,
    lengths,
    out_scaler,
    mse_weight=1.0,
    adv_weight=1.0,
    adv_streams=None,
    fm_weight=0.0,
    mask_nth_mgc_for_adv_loss=0,
    gan_type="lsgan",
    vuv_mask=False,
):
    netG.train() if train else netG.eval()
    netD.train() if train else netD.eval()

    log_metrics = {}

    if vuv_mask:
        # NOTE: Assuming 3rd stream is the V/UV
        vuv_idx = np.sum(model_config.stream_sizes[:2])
        is_v = torch.logical_and(
            out_feats[:, :, vuv_idx : vuv_idx + 1] > 0,
            in_feats[:, :, vuv_idx : vuv_idx + 1] > 0,
        )
        vuv = is_v
    else:
        vuv = 1.0

    # Run forward
    with autocast(enabled=grad_scaler is not None):
        pred_out_feats = netG(in_feats, lengths)

    real_netD_in_feats = select_streams(
        out_feats, model_config.stream_sizes, adv_streams
    )
    fake_netD_in_feats = select_streams(
        pred_out_feats,
        model_config.stream_sizes,
        adv_streams,
    )

    # Ref: http://sython.org/papers/ASJ/saito2017asja.pdf
    # 0-th mgc with adversarial trainging affects speech quality
    # NOTE: assuming that the first stream contains mgc
    if mask_nth_mgc_for_adv_loss > 0:
        real_netD_in_feats = real_netD_in_feats[:, :, mask_nth_mgc_for_adv_loss:]
        fake_netD_in_feats = fake_netD_in_feats[:, :, mask_nth_mgc_for_adv_loss:]

    # Real
    with autocast(enabled=grad_scaler is not None):
        D_real = netD(real_netD_in_feats * vuv, in_feats, lengths)
        # NOTE: must be list of list to support multi-scale discriminators
        assert isinstance(D_real, list) and isinstance(D_real[-1], list)
        # Fake
        D_fake_det = netD(fake_netD_in_feats.detach() * vuv, in_feats, lengths)

    # Mask (B, T, 1)
    mask = make_non_pad_mask(lengths).unsqueeze(-1).to(in_feats.device)

    # Update discriminator
    eps = 1e-14
    loss_real = 0
    loss_fake = 0

    with autocast(enabled=grad_scaler is not None):
        for idx, (D_real_, D_fake_det_) in enumerate(zip(D_real, D_fake_det)):
            if gan_type == "lsgan":
                loss_real_ = (D_real_[-1] - 1) ** 2
                loss_fake_ = D_fake_det_[-1] ** 2
            elif gan_type == "vanilla-gan":
                loss_real_ = -torch.log(D_real_[-1] + eps)
                loss_fake_ = -torch.log(1 - D_fake_det_[-1] + eps)
            elif gan_type == "hinge":
                loss_real_ = F.relu(1 - D_real_[-1])
                loss_fake_ = F.relu(1 + D_fake_det_[-1])
            else:
                raise ValueError(f"Unknown gan type: {gan_type}")

            # mask for D
            if (
                hasattr(netD, "downsample_scale")
                and mask.shape[1] // netD.downsample_scale == D_real_[-1].shape[1]
            ):
                D_mask = mask[:, :: netD.downsample_scale, :]
            else:
                if D_real_[-1].shape[1] == out_feats.shape[1]:
                    D_mask = mask
                else:
                    D_mask = None

            if D_mask is not None:
                loss_real_ = loss_real_.masked_select(D_mask).mean()
                loss_fake_ = loss_fake_.masked_select(D_mask).mean()
            else:
                loss_real_ = loss_real_.mean()
                loss_fake_ = loss_fake_.mean()

            log_metrics[f"Loss_Real_Scale{idx}"] = loss_real_.item()
            log_metrics[f"Loss_Fake_Scale{idx}"] = loss_fake_.item()

            loss_real += loss_real_
            loss_fake += loss_fake_

        loss_d = loss_real + loss_fake

    if train:
        optD.zero_grad()
        if grad_scaler is not None:
            grad_scaler.scale(loss_d).backward()
            grad_scaler.unscale_(optD)
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                netD.parameters(), optim_config.netD.clip_norm
            )
            log_metrics["GradNorm_D"] = grad_norm_d
            grad_scaler.step(optD)
        else:
            loss_d.backward()
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                netD.parameters(), optim_config.netD.clip_norm
            )
            log_metrics["GradNorm_D"] = grad_norm_d
            optD.step()

    # MSE loss
    with autocast(enabled=grad_scaler is not None):
        loss_feats = nn.MSELoss(reduction="none")(
            pred_out_feats.masked_select(mask), out_feats.masked_select(mask)
        ).mean()

    # adversarial loss
    with autocast(enabled=grad_scaler is not None):
        D_fake = netD(fake_netD_in_feats * vuv, in_feats, lengths)

        loss_adv = 0
        for idx, D_fake_ in enumerate(D_fake):
            if gan_type == "lsgan":
                loss_adv_ = (1 - D_fake_[-1]) ** 2
            elif gan_type == "vanilla-gan":
                loss_adv_ = -torch.log(D_fake_[-1] + eps)
            elif gan_type == "hinge":
                loss_adv_ = -D_fake_[-1]
            else:
                raise ValueError(f"Unknown gan type: {gan_type}")

            if (
                hasattr(netD, "downsample_scale")
                and mask.shape[1] // netD.downsample_scale == D_fake_[-1].shape[1]
            ):
                D_mask = mask[:, :: netD.downsample_scale, :]
            else:
                if D_real_[-1].shape[1] == out_feats.shape[1]:
                    D_mask = mask
                else:
                    D_mask = None

            if D_mask is not None:
                loss_adv_ = loss_adv_.masked_select(D_mask).mean()
            else:
                loss_adv_ = loss_adv_.mean()

            log_metrics[f"Loss_Adv_Scale{idx}"] = loss_adv_.item()

            loss_adv += loss_adv_

        # Feature matching loss
        loss_fm = torch.tensor(0.0).to(in_feats.device)
        if fm_weight > 0:
            for D_fake_, D_real_ in zip(D_fake, D_real):
                for fake_fmap, real_fmap in zip(D_fake_[:-1], D_real_[:-1]):
                    loss_fm += F.l1_loss(fake_fmap, real_fmap.detach())

        loss = mse_weight * loss_feats + adv_weight * loss_adv + fm_weight * loss_fm

    if train:
        optG.zero_grad()
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optG)
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                netG.parameters(), optim_config.netG.clip_norm
            )
            log_metrics["GradNorm_G"] = grad_norm_g
            grad_scaler.step(optG)
        else:
            loss.backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                netG.parameters(), optim_config.netG.clip_norm
            )
            log_metrics["GradNorm_G"] = grad_norm_g
            optG.step()

    # NOTE: this shouldn't be called multiple times in a training step
    if train and grad_scaler is not None:
        grad_scaler.update()

    # Metrics
    distortions = compute_distortions(
        pred_out_feats, out_feats, lengths, out_scaler, model_config
    )
    log_metrics.update(distortions)
    log_metrics.update(
        {
            "Loss": loss.item(),
            "Loss_Feats": loss_feats.item(),
            "Loss_Adv_Total": loss_adv.item(),
            "Loss_Feature_Matching": loss_fm.item(),
            "Loss_Real_Total": loss_real.item(),
            "Loss_Fake_Total": loss_fake.item(),
            "Loss_D": loss_d.item(),
        }
    )

    return loss, log_metrics


def train_loop(
    config,
    logger,
    device,
    netG,
    optG,
    schedulerG,
    netD,
    optD,
    schedulerD,
    grad_scaler,
    data_loaders,
    samplers,
    writer,
    out_scaler,
    use_mlflow,
    vocoder,
    vocoder_in_scaler,
):
    out_dir = Path(to_absolute_path(config.train.out_dir))
    best_dev_loss = torch.finfo(torch.float32).max
    last_dev_loss = torch.finfo(torch.float32).max

    adv_streams = config.train.adv_streams
    if len(adv_streams) != len(config.model.stream_sizes):
        raise ValueError("adv_streams must be specified for all streams")

    if dist.is_initialized() and dist.get_rank() != 0:

        def tqdm(x, **kwargs):
            return x

    else:
        from tqdm import tqdm

    train_iter = 1
    for epoch in tqdm(range(1, config.train.nepochs + 1)):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            if dist.is_initialized() and train and samplers[phase] is not None:
                samplers[phase].set_epoch(epoch)
            running_loss = 0
            running_metrics = {}
            evaluated = False
            for in_feats, out_feats, lengths in tqdm(
                data_loaders[phase], desc=f"{phase} iter", leave=False
            ):
                # NOTE: This is needed for pytorch's PackedSequence
                lengths, indices = torch.sort(lengths, dim=0, descending=True)
                in_feats, out_feats = (
                    in_feats[indices].to(device),
                    out_feats[indices].to(device),
                )
                if (not train) and (not evaluated):
                    eval_model(
                        phase,
                        epoch,
                        netG,
                        in_feats,
                        out_feats,
                        lengths,
                        config.model,
                        out_scaler,
                        writer,
                        sr=config.data.sample_rate,
                        use_world_codec=config.data.use_world_codec,
                        vocoder=vocoder,
                        vocoder_in_scaler=vocoder_in_scaler,
                        max_num_eval_utts=config.train.max_num_eval_utts,
                    )
                    evaluated = True

                loss, log_metrics = train_step(
                    model_config=config.model,
                    optim_config=config.train.optim,
                    netG=netG,
                    optG=optG,
                    netD=netD,
                    optD=optD,
                    grad_scaler=grad_scaler,
                    train=train,
                    in_feats=in_feats,
                    out_feats=out_feats,
                    lengths=lengths,
                    out_scaler=out_scaler,
                    mse_weight=config.train.mse_weight,
                    adv_weight=config.train.adv_weight,
                    adv_streams=adv_streams,
                    fm_weight=config.train.fm_weight,
                    mask_nth_mgc_for_adv_loss=config.train.mask_nth_mgc_for_adv_loss,
                    gan_type=config.train.gan_type,
                    vuv_mask=config.train.vuv_mask,
                )

                if train:
                    if writer is not None:
                        for key, val in log_metrics.items():
                            writer.add_scalar(f"{key}_Step/{phase}", val, train_iter)
                    train_iter += 1

                running_loss += loss.item()
                for k, v in log_metrics.items():
                    try:
                        running_metrics[k] += float(v)
                    except KeyError:
                        running_metrics[k] = float(v)

            ave_loss = running_loss / len(data_loaders[phase])
            logger.info("[%s] [Epoch %s]: loss %s", phase, epoch, ave_loss)
            if writer is not None:
                writer.add_scalar(f"Loss_Epoch/{phase}", ave_loss, epoch)
            if use_mlflow:
                mlflow.log_metric(f"{phase}_loss", ave_loss, step=epoch)

            for k, v in running_metrics.items():
                ave_v = v / len(data_loaders[phase])
                if writer is not None:
                    writer.add_scalar(f"{k}_Epoch/{phase}", ave_v, epoch)
                if use_mlflow:
                    mlflow.log_metric(f"{phase}_{k}", ave_v, step=epoch)

            if not train:
                last_dev_loss = ave_loss
            if not train and ave_loss < best_dev_loss:
                best_dev_loss = ave_loss
                for model, opt, scheduler, postfix in [
                    (netG, optG, schedulerG, ""),
                    (netD, optD, schedulerD, "_D"),
                ]:
                    save_checkpoint(
                        logger,
                        out_dir,
                        model,
                        opt,
                        scheduler,
                        epoch,
                        is_best=True,
                        postfix=postfix,
                    )

        schedulerG.step()
        schedulerD.step()

        if epoch % config.train.checkpoint_epoch_interval == 0:
            for model, opt, scheduler, postfix in [
                (netG, optG, schedulerG, ""),
                (netD, optD, schedulerD, "_D"),
            ]:
                save_checkpoint(
                    logger,
                    out_dir,
                    model,
                    opt,
                    scheduler,
                    epoch,
                    is_best=False,
                    postfix=postfix,
                )

    for model, opt, scheduler, postfix in [
        (netG, optG, schedulerG, ""),
        (netD, optD, schedulerD, "_D"),
    ]:
        save_checkpoint(
            logger,
            out_dir,
            model,
            opt,
            scheduler,
            config.train.nepochs,
            postfix=postfix,
        )
    logger.info("The best loss was %s", best_dev_loss)
    if use_mlflow:
        mlflow.log_metric("best_dev_loss", best_dev_loss, step=epoch)
        mlflow.log_artifacts(out_dir)

    return last_dev_loss


@hydra.main(config_path="conf/train_postfilter", config_name="config")
def my_app(config: DictConfig) -> None:
    # NOTE: set discriminator's in_dim automatically
    if config.model.netD.in_dim is None:
        stream_sizes = np.asarray(config.model.stream_sizes)
        D_in_dim = int((stream_sizes * np.asarray(config.train.adv_streams)).sum())
        if config.train.mask_nth_mgc_for_adv_loss > 0:
            D_in_dim -= config.train.mask_nth_mgc_for_adv_loss
        config.model.netD.in_dim = D_in_dim
    if "stream_sizes" in config.model.netG:
        config.model.netG.stream_sizes = config.model.stream_sizes

    if "max_time_frames" in config.data and config.data.max_time_frames > 0:
        collate_fn = partial(
            collate_fn_random_segments, max_time_frames=config.data.max_time_frames
        )
    else:
        collate_fn = collate_fn_default

    if config.train.use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        (netG, optG, schedulerG),
        (netD, optD, schedulerD),
        grad_scaler,
        data_loaders,
        samplers,
        writer,
        logger,
        _,
        out_scaler,
    ) = setup_gan(config, device, collate_fn)

    path = config.train.pretrained_vocoder_checkpoint
    if path is not None and len(path) > 0:
        logger.info(f"Loading pretrained vocoder checkpoint from {path}")
        vocoder, vocoder_in_scaler = load_vocoder(path, device)
    else:
        vocoder, vocoder_in_scaler = None, None

    out_scaler = PyTorchStandardScaler(
        torch.from_numpy(out_scaler.mean_), torch.from_numpy(out_scaler.scale_)
    ).to(device)
    use_mlflow = config.mlflow.enabled

    if use_mlflow:
        with mlflow.start_run() as run:
            # NOTE: modify out_dir when running with mlflow
            config.train.out_dir = f"{config.train.out_dir}/{run.info.run_id}"
            save_configs(config)
            log_params_from_omegaconf_dict(config)
            last_dev_loss = train_loop(
                config,
                logger,
                device,
                netG,
                optG,
                schedulerG,
                netD,
                optD,
                schedulerD,
                grad_scaler,
                data_loaders,
                samplers,
                writer,
                out_scaler,
                use_mlflow,
                vocoder,
                vocoder_in_scaler,
            )
    else:
        save_configs(config)
        last_dev_loss = train_loop(
            config,
            logger,
            device,
            netG,
            optG,
            schedulerG,
            netD,
            optD,
            schedulerD,
            grad_scaler,
            data_loaders,
            samplers,
            writer,
            out_scaler,
            use_mlflow,
            vocoder,
            vocoder_in_scaler,
        )

    return last_dev_loss


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
