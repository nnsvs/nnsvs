from functools import partial
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from hydra.utils import to_absolute_path
from nnsvs.multistream import select_streams
from nnsvs.train_util import (
    collate_fn_default,
    collate_fn_random_segments,
    compute_distortions,
    eval_spss_model,
    log_params_from_omegaconf_dict,
    save_checkpoint,
    save_configs,
    setup_cyclegan,
)
from nnsvs.util import PyTorchStandardScaler, make_non_pad_mask
from omegaconf import DictConfig
from torch.cuda.amp import autocast
from torch.nn import functional as F
from tqdm import tqdm


def train_step(
    model_config,
    optim_config,
    netG_A2B,
    netG_B2A,
    optG,
    netD_A,
    netD_B,
    optD,
    grad_scaler,
    train,
    in_feats,
    out_feats,
    lengths,
    out_scaler,
    adv_weight=1.0,
    adv_streams=None,
    fm_weight=0.0,
    mask_nth_mgc_for_adv_loss=0,
    gan_type="lsgan",
    vuv_mask=False,
    cycle_weight=10.0,
    id_weight=5.0,
    use_id_loss=True,
):
    netG_A2B.train() if train else netG_A2B.eval()
    netG_B2A.train() if train else netG_B2A.eval()
    netD_A.train() if train else netD_A.eval()
    netD_B.train() if train else netD_B.eval()

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

    # Run forward A2B and B2A
    with autocast(enabled=grad_scaler is not None):
        pred_out_feats_A = netG_B2A(out_feats, lengths)
        pred_out_feats_B = netG_A2B(in_feats, lengths)

        # Cycle consistency loss
        loss_cycle = F.l1_loss(
            netG_A2B(pred_out_feats_A, lengths) * vuv, out_feats * vuv
        ) + F.l1_loss(netG_B2A(pred_out_feats_B, lengths) * vuv, in_feats * vuv)

        # Identity mapping loss
        if use_id_loss and id_weight > 0:
            loss_id = F.l1_loss(
                netG_A2B(out_feats, lengths) * vuv, out_feats * vuv
            ) + F.l1_loss(netG_B2A(in_feats, lengths) * vuv, in_feats * vuv)
        else:
            loss_id = torch.tensor(0.0).to(in_feats.device)

    real_netD_in_feats_A = select_streams(
        in_feats, model_config.stream_sizes, adv_streams
    )
    real_netD_in_feats_B = select_streams(
        out_feats, model_config.stream_sizes, adv_streams
    )
    fake_netD_in_feats_A = select_streams(
        pred_out_feats_A,
        model_config.stream_sizes,
        adv_streams,
    )
    fake_netD_in_feats_B = select_streams(
        pred_out_feats_B,
        model_config.stream_sizes,
        adv_streams,
    )

    # Ref: http://sython.org/papers/ASJ/saito2017asja.pdf
    # 0-th mgc with adversarial trainging affects speech quality
    # NOTE: assuming that the first stream contains mgc
    if mask_nth_mgc_for_adv_loss > 0:
        real_netD_in_feats_A = real_netD_in_feats_A[:, :, mask_nth_mgc_for_adv_loss:]
        real_netD_in_feats_B = real_netD_in_feats_B[:, :, mask_nth_mgc_for_adv_loss:]
        fake_netD_in_feats_A = fake_netD_in_feats_A[:, :, mask_nth_mgc_for_adv_loss:]
        fake_netD_in_feats_B = fake_netD_in_feats_B[:, :, mask_nth_mgc_for_adv_loss:]

    with autocast(enabled=grad_scaler is not None):
        # Real
        D_real_A = netD_A(real_netD_in_feats_A * vuv, in_feats, lengths)
        D_real_B = netD_B(real_netD_in_feats_B * vuv, in_feats, lengths)
        # Fake
        D_fake_det_A = netD_A(fake_netD_in_feats_A.detach() * vuv, in_feats, lengths)
        D_fake_det_B = netD_B(fake_netD_in_feats_B.detach() * vuv, in_feats, lengths)

    # Mask (B, T, 1)
    mask = make_non_pad_mask(lengths).unsqueeze(-1).to(in_feats.device)

    # Update discriminator
    eps = 1e-14
    loss_real = 0
    loss_fake = 0

    # A
    with autocast(enabled=grad_scaler is not None):
        for idx, (D_real_, D_fake_det_) in enumerate(zip(D_real_A, D_fake_det_A)):
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
                hasattr(netD_A, "downsample_scale")
                and mask.shape[1] // netD_A.downsample_scale == D_real_[-1].shape[1]
            ):
                D_mask = mask[:, :: netD_A.downsample_scale, :]
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

            log_metrics[f"Loss_Real_Scale{idx}_A"] = loss_real_.item()
            log_metrics[f"Loss_Fake_Scale{idx}_A"] = loss_fake_.item()

            loss_real += loss_real_
            loss_fake += loss_fake_

        # B
        for idx, (D_real_, D_fake_det_) in enumerate(zip(D_real_B, D_fake_det_B)):
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
                hasattr(netD_B, "downsample_scale")
                and mask.shape[1] // netD_B.downsample_scale == D_real_[-1].shape[1]
            ):
                D_mask = mask[:, :: netD_B.downsample_scale, :]
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

            log_metrics[f"Loss_Real_Scale{idx}_B"] = loss_real_.item()
            log_metrics[f"Loss_Fake_Scale{idx}_B"] = loss_fake_.item()

            loss_real += loss_real_
            loss_fake += loss_fake_

        loss_d = loss_real + loss_fake

    if train:
        optD.zero_grad()
        if grad_scaler is not None:
            grad_scaler.scale(loss_d).backward()
            grad_scaler.unscale_(optD)
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                netD_A.parameters(), optim_config.netD.clip_norm
            )
            log_metrics["GradNorm_D/netG_A2B"] = grad_norm_d
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                netD_B.parameters(), optim_config.netD.clip_norm
            )
            log_metrics["GradNorm_D/netG_B2A"] = grad_norm_d
            grad_scaler.step(optD)
        else:
            loss_d.backward()
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                netD_A.parameters(), optim_config.netD.clip_norm
            )
            log_metrics["GradNorm_D/netG_A2B"] = grad_norm_d
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                netD_B.parameters(), optim_config.netD.clip_norm
            )
            log_metrics["GradNorm_D/netG_B2A"] = grad_norm_d
            optD.step()

    # adversarial loss
    loss_adv = 0

    with autocast(enabled=grad_scaler is not None):
        # A
        D_fake_A = netD_A(fake_netD_in_feats_A * vuv, in_feats, lengths)
        for idx, D_fake_ in enumerate(D_fake_A):
            if gan_type == "lsgan":
                loss_adv_ = (1 - D_fake_[-1]) ** 2
            elif gan_type == "vanilla-gan":
                loss_adv_ = -torch.log(D_fake_[-1] + eps)
            elif gan_type == "hinge":
                loss_adv_ = -D_fake_[-1]
            else:
                raise ValueError(f"Unknown gan type: {gan_type}")

            if (
                hasattr(netD_A, "downsample_scale")
                and mask.shape[1] // netD_A.downsample_scale == D_fake_[-1].shape[1]
            ):
                D_mask = mask[:, :: netD_A.downsample_scale, :]
            else:
                if D_real_[-1].shape[1] == out_feats.shape[1]:
                    D_mask = mask
                else:
                    D_mask = None

            if D_mask is not None:
                loss_adv_ = loss_adv_.masked_select(D_mask).mean()
            else:
                loss_adv_ = loss_adv_.mean()

            log_metrics[f"Loss_Adv_Scale{idx}_A"] = loss_adv_.item()

            loss_adv += loss_adv_

        # B
        D_fake_B = netD_B(fake_netD_in_feats_B * vuv, in_feats, lengths)
        for idx, D_fake_ in enumerate(D_fake_B):
            if gan_type == "lsgan":
                loss_adv_ = (1 - D_fake_[-1]) ** 2
            elif gan_type == "vanilla-gan":
                loss_adv_ = -torch.log(D_fake_[-1] + eps)
            elif gan_type == "hinge":
                loss_adv_ = -D_fake_[-1]
            else:
                raise ValueError(f"Unknown gan type: {gan_type}")

            if (
                hasattr(netD_B, "downsample_scale")
                and mask.shape[1] // netD_B.downsample_scale == D_fake_[-1].shape[1]
            ):
                D_mask = mask[:, :: netD_B.downsample_scale, :]
            else:
                if D_real_[-1].shape[1] == out_feats.shape[1]:
                    D_mask = mask
                else:
                    D_mask = None

            if D_mask is not None:
                loss_adv_ = loss_adv_.masked_select(D_mask).mean()
            else:
                loss_adv_ = loss_adv_.mean()

            log_metrics[f"Loss_Adv_Scale{idx}_B"] = loss_adv_.item()

            loss_adv += loss_adv_

        # Feature matching loss
        loss_fm = torch.tensor(0.0).to(in_feats.device)
        if fm_weight > 0:
            for D_fake_, D_real_ in zip(D_fake_A, D_real_A):
                for fake_fmap, real_fmap in zip(D_fake_[:-1], D_real_[:-1]):
                    loss_fm += F.l1_loss(fake_fmap, real_fmap.detach())
            for D_fake_, D_real_ in zip(D_fake_B, D_real_B):
                for fake_fmap, real_fmap in zip(D_fake_[:-1], D_real_[:-1]):
                    loss_fm += F.l1_loss(fake_fmap, real_fmap.detach())

    loss = (
        adv_weight * loss_adv
        + cycle_weight * loss_cycle
        + id_weight * loss_id
        + fm_weight * loss_fm
    )

    if train:
        optG.zero_grad()
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optG)
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                netG_A2B.parameters(), optim_config.netG.clip_norm
            )
            log_metrics["GradNorm_G/netG_A2B"] = grad_norm_g
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                netG_B2A.parameters(), optim_config.netG.clip_norm
            )
            log_metrics["GradNorm_G/netG_B2A"] = grad_norm_g
            grad_scaler.step(optG)
        else:
            loss.backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                netG_A2B.parameters(), optim_config.netG.clip_norm
            )
            log_metrics["GradNorm_G/netG_A2B"] = grad_norm_g
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                netG_B2A.parameters(), optim_config.netG.clip_norm
            )
            log_metrics["GradNorm_G/netG_B2A"] = grad_norm_g
            optG.step()

    # NOTE: this shouldn't be called multiple times in a training step
    if train and grad_scaler is not None:
        grad_scaler.update()

    # Metrics
    distortions = compute_distortions(
        pred_out_feats_B, out_feats, lengths, out_scaler, model_config
    )
    log_metrics.update(distortions)
    log_metrics.update(
        {
            "Loss": loss.item(),
            "Loss_Adv_Total": loss_adv.item(),
            "Loss_Cycle": loss_cycle.item(),
            "Loss_Identity": loss_id.item(),
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
    netG_A2B,
    netG_B2A,
    optG,
    schedulerG,
    netD_A,
    netD_B,
    optD,
    schedulerD,
    grad_scaler,
    data_loaders,
    writer,
    out_scaler,
    use_mlflow,
):
    out_dir = Path(to_absolute_path(config.train.out_dir))
    best_dev_loss = torch.finfo(torch.float32).max
    last_dev_loss = torch.finfo(torch.float32).max

    adv_streams = config.train.adv_streams
    if len(adv_streams) != len(config.model.stream_sizes):
        raise ValueError("adv_streams must be specified for all streams")

    for epoch in tqdm(range(1, config.train.nepochs + 1)):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            running_loss = 0
            running_metrics = {}
            evaluated = False
            for in_feats, out_feats, lengths in data_loaders[phase]:
                # NOTE: This is needed for pytorch's PackedSequence
                lengths, indices = torch.sort(lengths, dim=0, descending=True)
                in_feats, out_feats = (
                    in_feats[indices].to(device),
                    out_feats[indices].to(device),
                )

                if (not train) and (not evaluated):
                    eval_spss_model(
                        epoch,
                        netG_A2B,
                        in_feats,
                        out_feats,
                        lengths,
                        config.model,
                        out_scaler,
                        writer,
                        sr=config.data.sample_rate,
                    )
                    evaluated = True

                if (
                    config.train.id_loss_until > 0
                    and epoch > config.train.id_loss_until
                ):
                    use_id_loss = False
                else:
                    use_id_loss = True

                loss, log_metrics = train_step(
                    model_config=config.model,
                    optim_config=config.train.optim,
                    netG_A2B=netG_A2B,
                    netG_B2A=netG_B2A,
                    optG=optG,
                    netD_A=netD_A,
                    netD_B=netD_B,
                    optD=optD,
                    grad_scaler=grad_scaler,
                    train=train,
                    in_feats=in_feats,
                    out_feats=out_feats,
                    lengths=lengths,
                    out_scaler=out_scaler,
                    adv_weight=config.train.adv_weight,
                    adv_streams=adv_streams,
                    fm_weight=config.train.fm_weight,
                    mask_nth_mgc_for_adv_loss=config.train.mask_nth_mgc_for_adv_loss,
                    gan_type=config.train.gan_type,
                    vuv_mask=config.train.vuv_mask,
                    cycle_weight=config.train.cycle_weight,
                    id_weight=config.train.id_weight,
                    use_id_loss=use_id_loss,
                )
                running_loss += loss.item()
                for k, v in log_metrics.items():
                    try:
                        running_metrics[k] += float(v)
                    except KeyError:
                        running_metrics[k] = float(v)

            ave_loss = running_loss / len(data_loaders[phase])
            logger.info("[%s] [Epoch %s]: loss %s", phase, epoch, ave_loss)

            for k, v in running_metrics.items():
                ave_v = v / len(data_loaders[phase])
                if writer is not None:
                    writer.add_scalar(f"{k}/{phase}", ave_v, epoch)
                if use_mlflow:
                    mlflow.log_metric(f"{phase}_{k}", ave_v, step=epoch)

            if not train:
                last_dev_loss = ave_loss
            if not train and ave_loss < best_dev_loss:
                best_dev_loss = ave_loss
                for model, opt, scheduler, postfix in [
                    (netG_A2B, optG, schedulerG, "_A2B"),
                    (netG_B2A, optG, schedulerG, "_B2A"),
                    (netD_A, optD, schedulerD, "_D_A"),
                    (netD_B, optD, schedulerD, "_D_B"),
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
                (netG_A2B, optG, schedulerG, "_A2B"),
                (netG_B2A, optG, schedulerG, "_B2A"),
                (netD_A, optD, schedulerD, "_D_A"),
                (netD_B, optD, schedulerD, "_D_B"),
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
        (netG_A2B, optG, schedulerG, "_A2B"),
        (netG_B2A, optG, schedulerG, "_B2A"),
        (netD_A, optD, schedulerD, "_D_A"),
        (netD_B, optD, schedulerD, "_D_B"),
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


@hydra.main(config_path="conf/train_postfilter_cycle", config_name="config")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        (netG_A2B, netG_B2A, optG, schedulerG),
        (netD_A, netD_B, optD, schedulerD),
        grad_scaler,
        data_loaders,
        writer,
        logger,
        _,
        out_scaler,
    ) = setup_cyclegan(config, device, collate_fn)

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
                netG_A2B,
                netG_B2A,
                optG,
                schedulerG,
                netD_A,
                netD_B,
                optD,
                schedulerD,
                grad_scaler,
                data_loaders,
                writer,
                out_scaler,
                use_mlflow,
            )
    else:
        save_configs(config)
        last_dev_loss = train_loop(
            config,
            logger,
            device,
            netG_A2B,
            netG_B2A,
            optG,
            schedulerG,
            netD_A,
            netD_B,
            optD,
            schedulerD,
            grad_scaler,
            data_loaders,
            writer,
            out_scaler,
            use_mlflow,
        )

    return last_dev_loss


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
