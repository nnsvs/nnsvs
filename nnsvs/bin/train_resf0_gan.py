from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from hydra.utils import to_absolute_path
from nnsvs.base import PredictionType
from nnsvs.multistream import (
    get_static_features,
    get_static_stream_sizes,
    select_streams,
)
from nnsvs.train_util import (
    check_resf0_config,
    compute_batch_pitch_regularization_weight,
    compute_distortions,
    eval_spss_model,
    log_params_from_omegaconf_dict,
    save_checkpoint,
    save_configs,
    setup_gan,
)
from nnsvs.util import PyTorchStandardScaler, make_non_pad_mask
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def train_step(
    model_config,
    optim_config,
    netG,
    optG,
    netD,
    optD,
    train,
    in_feats,
    out_feats,
    lengths,
    out_scaler,
    pitch_reg_dyn_ws,
    pitch_reg_weight=1.0,
    adv_weight=1.0,
    adv_streams=None,
    fm_weight=0.0,
    adv_use_static_feats_only=True,
    mask_nth_mgc_for_adv_loss=0,
    gan_type="lsgan",
    adv_segment_length=-1,
):
    optG.zero_grad()
    optD.zero_grad()
    log_metrics = {}

    criterion = nn.MSELoss(reduction="none")
    prediction_type = (
        netG.module.prediction_type()
        if isinstance(netG, nn.DataParallel)
        else netG.prediction_type()
    )
    # NOTE: it is not trivial to adapt GAN for probabilistic models
    assert prediction_type != PredictionType.PROBABILISTIC

    # Apply preprocess if required (e.g., FIR filter for shallow AR)
    # defaults to no-op
    if isinstance(netG, nn.DataParallel):
        out_feats = netG.module.preprocess_target(out_feats)
    else:
        out_feats = netG.preprocess_target(out_feats)

    # Run forward
    pred_out_feats, lf0_residual = netG(in_feats, lengths)

    # Multi-scale: multiple outputs or not.
    # Only the last output is used for computing adversarial loss
    # and discriminator loss.
    is_multiscale = isinstance(pred_out_feats, list)

    # Select streams for computing adversarial loss
    if adv_use_static_feats_only:
        real_netD_in_feats = torch.cat(
            get_static_features(
                out_feats,
                model_config.num_windows,
                model_config.stream_sizes,
                model_config.has_dynamic_features,
                adv_streams,
            ),
            dim=-1,
        )
        fake_netD_in_feats = torch.cat(
            get_static_features(
                pred_out_feats[-1] if is_multiscale else pred_out_feats,
                model_config.num_windows,
                model_config.stream_sizes,
                model_config.has_dynamic_features,
                adv_streams,
            ),
            dim=-1,
        )
    else:
        real_netD_in_feats = select_streams(
            out_feats, model_config.stream_sizes, adv_streams
        )
        fake_netD_in_feats = select_streams(
            pred_out_feats[-1] if is_multiscale else pred_out_feats,
            model_config.stream_sizes,
            adv_streams,
        )

    # Rather than classifying whole features as real or fake, use smaller segments
    # that should ease the training of GANs I guess
    # The idea is similar to the trick used for training neural vocoders. i.e.,
    # use small segments like 8000 or 24000 samples for training vocoders.
    if adv_segment_length > 0:
        if lengths.min() < adv_segment_length:
            real_netD_in_feats = real_netD_in_feats[:, : lengths.min(), :]
            fake_netD_in_feats = fake_netD_in_feats[:, : lengths.min(), :]
        else:
            si = np.random.randint(0, lengths.min() - adv_segment_length)
            real_netD_in_feats = real_netD_in_feats[:, si : si + adv_segment_length, :]
            fake_netD_in_feats = fake_netD_in_feats[:, si : si + adv_segment_length, :]
            in_feats = in_feats[:, si : si + adv_segment_length, :]

    # Ref: http://sython.org/papers/ASJ/saito2017asja.pdf
    # 0-th mgc with adversarial trainging affects speech quality
    # NOTE: assuming that the first stream contains mgc
    if mask_nth_mgc_for_adv_loss > 0:
        real_netD_in_feats = real_netD_in_feats[:, :, mask_nth_mgc_for_adv_loss:]
        fake_netD_in_feats = fake_netD_in_feats[:, :, mask_nth_mgc_for_adv_loss:]

    # Real
    D_real = netD(real_netD_in_feats, in_feats, lengths)
    # NOTE: must be list of list to support multi-scale discriminators
    assert isinstance(D_real, list) and isinstance(D_real[-1], list)
    # Fake
    D_fake_det = netD(fake_netD_in_feats.detach(), in_feats, lengths)

    # Mask (B, T, 1)
    mask = make_non_pad_mask(lengths).unsqueeze(-1).to(in_feats.device)

    # Update discriminator
    eps = 1e-14
    loss_real = 0
    loss_fake = 0
    for idx, (D_real_, D_fake_det_) in enumerate(zip(D_real, D_fake_det)):
        if gan_type == "lsgan":
            loss_real_ = (D_real_[-1] - 1) ** 2
            loss_fake_ = D_fake_det_[-1] ** 2
        elif gan_type == "vanilla-gan":
            loss_real_ = -torch.log(D_real_[-1] + eps)
            loss_fake_ = -torch.log(1 - D_fake_det_[-1] + eps)
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

        log_metrics[f"Loss_Real_scale{idx}"] = loss_real_.item()
        log_metrics[f"Loss_Fake_scale{idx}"] = loss_fake_.item()

        loss_real += loss_real_
        loss_fake += loss_fake_

    loss_d = loss_real + loss_fake

    if train:
        loss_d.backward()
        grad_norm_d = torch.nn.utils.clip_grad_norm_(
            netD.parameters(), optim_config.netD.clip_norm
        )
        log_metrics["GradNorm_D"] = grad_norm_d
        optD.step()

    # Update generator
    if is_multiscale:
        loss_feats = 0
        # Use the last feats loss only when adversarial training is enabled
        if adv_weight > 0:
            for idx, pred_out_feats_ in enumerate(pred_out_feats):
                loss_feats_ = criterion(
                    pred_out_feats_.masked_select(mask), out_feats.masked_select(mask)
                ).mean()
                log_metrics[f"Loss_Feats_scale{idx}"] = loss_feats_.item()
                loss_feats += loss_feats_
        else:
            for idx, pred_out_feats_ in enumerate(pred_out_feats[:-1]):
                loss_feats_ = criterion(
                    pred_out_feats_.masked_select(mask), out_feats.masked_select(mask)
                ).mean()
                log_metrics[f"Loss_Feats_scale{idx}"] = loss_feats_.item()
                loss_feats += loss_feats_
    else:
        loss_feats = criterion(
            pred_out_feats.masked_select(mask), out_feats.masked_select(mask)
        ).mean()

    # adversarial loss
    D_fake = netD(fake_netD_in_feats, in_feats, lengths)
    loss_adv = 0
    for idx, D_fake_ in enumerate(D_fake):
        if gan_type == "lsgan":
            loss_adv_ = (1 - D_fake_[-1]) ** 2
        elif gan_type == "vanilla-gan":
            loss_adv_ = -torch.log(D_fake_[-1] + eps)
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

        log_metrics[f"Loss_Adv_scale{idx}"] = loss_adv_.item()

        loss_adv += loss_adv_

    # Feature matching loss
    loss_fm = torch.tensor(0.0).to(in_feats.device)
    if fm_weight > 0:
        for D_fake_, D_real_ in zip(D_fake, D_real):
            for fake_fmap, real_fmap in zip(D_fake_[:-1], D_real_[:-1]):
                loss_fm += F.l1_loss(fake_fmap, real_fmap.detach())

    # Pitch regularization
    # NOTE: l1 loss seems to be better than mse loss in my experiments
    # we could use l2 loss as suggested in the sinsy's paper
    if isinstance(lf0_residual, list):
        loss_pitch = 0
        for idx, lf0_residual_ in enumerate(lf0_residual):
            loss_pitch_ = (
                (pitch_reg_dyn_ws * lf0_residual_.abs()).masked_select(mask).mean()
            )
            log_metrics[f"Loss_Pitch_scale{idx}"] = loss_pitch_.item()
            loss_pitch += loss_pitch_
    else:
        loss_pitch = (pitch_reg_dyn_ws * lf0_residual.abs()).masked_select(mask).mean()

    loss = (
        loss_feats
        + adv_weight * loss_adv
        + pitch_reg_weight * loss_pitch
        + fm_weight * loss_fm
    )

    if train:
        loss.backward()
        grad_norm_g = torch.nn.utils.clip_grad_norm_(
            netG.parameters(), optim_config.netG.clip_norm
        )
        log_metrics["GradNorm_G"] = grad_norm_g
        optG.step()

    # Metrics
    if is_multiscale:
        distortions = {}
        for idx, pred_out_feats_ in enumerate(pred_out_feats):
            distortions_ = compute_distortions(
                pred_out_feats_, out_feats, lengths, out_scaler, model_config
            )
            for k, v in distortions_.items():
                log_metrics[f"{k}_scale{idx}"] = v
            # Keep the last-scale distortion
            distortions = distortions_
    else:
        distortions = compute_distortions(
            pred_out_feats, out_feats, lengths, out_scaler, model_config
        )
    log_metrics.update(distortions)
    log_metrics.update(
        {
            "Loss": loss.item(),
            "Loss_Feats": loss_feats.item(),
            "Loss_Adv": loss_adv.item(),
            "Loss_Feature_Matching": loss_fm.item(),
            "Loss_Pitch": loss_pitch.item(),
            "Loss_Real": loss_real.item(),
            "Loss_Fake": loss_fake.item(),
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
    data_loaders,
    writer,
    in_scaler,
    out_scaler,
    use_mlflow,
):
    out_dir = Path(to_absolute_path(config.train.out_dir))
    best_dev_loss = torch.finfo(torch.float32).max
    last_dev_loss = torch.finfo(torch.float32).max

    in_lf0_idx = config.data.in_lf0_idx
    in_rest_idx = config.data.in_rest_idx
    if in_lf0_idx is None or in_rest_idx is None:
        raise ValueError("in_lf0_idx and in_rest_idx must be specified")
    pitch_reg_weight = config.train.pitch_reg_weight
    fm_weight = config.train.fm_weight
    adv_streams = config.train.adv_streams
    if len(adv_streams) != len(config.model.stream_sizes):
        raise ValueError("adv_streams must be specified for all streams")

    E_loss_feats = 1.0
    E_loss_adv = 1.0
    for epoch in tqdm(range(1, config.train.nepochs + 1)):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            netG.train() if train else netG.eval()
            netD.train() if train else netD.eval()
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
                        netG,
                        in_feats,
                        out_feats,
                        lengths,
                        config.model,
                        out_scaler,
                        writer,
                        sr=48000,
                    )
                    evaluated = True

                # Compute denormalized log-F0 in the musical scores
                # test - s.min_[in_lf0_idx]) / s.scale_[in_lf0_idx]
                lf0_score_denorm = (
                    in_feats[:, :, in_lf0_idx] - in_scaler.min_[in_lf0_idx]
                ) / in_scaler.scale_[in_lf0_idx]
                # Fill zeros for rest and padded frames
                lf0_score_denorm *= (in_feats[:, :, in_rest_idx] <= 0).float()
                for idx, length in enumerate(lengths):
                    lf0_score_denorm[idx, length:] = 0
                # Compute time-variant pitch regularization weight vector
                pitch_reg_dyn_ws = compute_batch_pitch_regularization_weight(
                    lf0_score_denorm
                )

                # Adv weight
                if config.train.adv_weight > 0 and config.train.dynamic_adv_weight:
                    adv_weight = config.train.adv_weight * np.clip(
                        E_loss_feats / E_loss_adv, 0, 1e3
                    )
                else:
                    adv_weight = config.train.adv_weight

                loss, log_metrics = train_step(
                    model_config=config.model,
                    optim_config=config.train.optim,
                    netG=netG,
                    optG=optG,
                    netD=netD,
                    optD=optD,
                    train=train,
                    in_feats=in_feats,
                    out_feats=out_feats,
                    lengths=lengths,
                    out_scaler=out_scaler,
                    pitch_reg_dyn_ws=pitch_reg_dyn_ws,
                    pitch_reg_weight=pitch_reg_weight,
                    adv_weight=adv_weight,
                    adv_streams=adv_streams,
                    fm_weight=fm_weight,
                    adv_segment_length=config.train.adv_segment_length,
                    adv_use_static_feats_only=config.train.adv_use_static_feats_only,
                    mask_nth_mgc_for_adv_loss=config.train.mask_nth_mgc_for_adv_loss,
                    gan_type=config.train.gan_type,
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
            # Update dynamic adv_weight parameters
            if train:
                N = len(data_loaders[phase])
                keys = sorted(
                    [k for k in running_metrics.keys() if "Loss_Feats_scale" in k]
                )
                # Multi-scale case: use the last scale's feats loss
                if len(keys) > 0:
                    E_loss_feats = running_metrics[keys[-1]] / N
                else:
                    E_loss_feats = running_metrics["Loss_Feats"] / N
                E_loss_adv = running_metrics["Loss_Adv"] / N
                logger.info(
                    "[%s] [Epoch %s]: dynamic adv weight %s",
                    phase,
                    epoch,
                    E_loss_feats / E_loss_adv,
                )
                if writer is not None:
                    writer.add_scalar(
                        f"Dynamic_Adv_Weight/{phase}",
                        np.clip(E_loss_feats / E_loss_adv, 0, 1e3),
                        epoch,
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


@hydra.main(config_path="conf/train_resf0_gan", config_name="config")
def my_app(config: DictConfig) -> None:
    # NOTE: set discriminator's in_dim automatically
    if config.model.netD.in_dim is None:
        if config.train.adv_use_static_feats_only:
            stream_sizes = get_static_stream_sizes(
                config.model.stream_sizes,
                config.model.has_dynamic_features,
                config.model.num_windows,
            )
        else:
            stream_sizes = np.asarray(config.model.stream_sizes)
        D_in_dim = int((stream_sizes * np.asarray(config.train.adv_streams)).sum())
        if config.train.mask_nth_mgc_for_adv_loss > 0:
            D_in_dim -= config.train.mask_nth_mgc_for_adv_loss
        config.model.netD.in_dim = D_in_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        (netG, optG, schedulerG),
        (netD, optD, schedulerD),
        data_loaders,
        writer,
        logger,
        in_scaler,
        out_scaler,
    ) = setup_gan(config, device)

    check_resf0_config(logger, netG, config, in_scaler, out_scaler)

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
                data_loaders,
                writer,
                in_scaler,
                out_scaler,
                use_mlflow,
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
            data_loaders,
            writer,
            in_scaler,
            out_scaler,
            use_mlflow,
        )

    return last_dev_loss


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
