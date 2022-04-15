from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from hydra.utils import to_absolute_path
from nnsvs.base import PredictionType
from nnsvs.bin.train_resf0 import (
    check_resf0_config,
    compute_batch_pitch_regularization_weight,
    compute_distortions,
)
from nnsvs.multistream import select_streams
from nnsvs.train_util import (
    log_params_from_omegaconf_dict,
    save_checkpoint,
    save_configs,
    setup_gan,
)
from nnsvs.util import PyTorchStandardScaler, make_non_pad_mask
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm


def train_step(
    model_config,
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

    # Select streams for computing adversarial loss
    real_netD_in_feats = select_streams(
        out_feats, model_config.stream_sizes, adv_streams
    )
    fake_netD_in_feats = select_streams(
        pred_out_feats, model_config.stream_sizes, adv_streams
    )

    # Real
    D_real = netD(real_netD_in_feats, lengths)
    # Fake
    D_fake_det = netD(fake_netD_in_feats.detach(), lengths)

    # Mask (B, T, 1)
    mask = make_non_pad_mask(lengths).unsqueeze(-1).to(in_feats.device)
    if hasattr(netD, "downsample_scale"):
        D_mask = mask[:, :: netD.downsample_scale, :]
    else:
        if D_real.shape[1] == real_netD_in_feats.shape[1]:
            D_mask = mask
        else:
            D_mask = None

    # Update discriminator
    loss_real = (D_real - 1) ** 2
    loss_fake = D_fake_det ** 2
    if D_mask is not None:
        loss_real = loss_real.masked_select(D_mask).mean()
        loss_fake = loss_fake.masked_select(D_mask).mean()
    else:
        loss_real = loss_real.mean()
        loss_fake = loss_fake.mean()
    loss_d = loss_real + loss_fake

    if train:
        loss_d.backward()
        optD.step()

    # Update generator
    loss_feats = criterion(
        pred_out_feats.masked_select(mask), out_feats.masked_select(mask)
    ).mean()

    # adversarial loss
    D_fake = netD(fake_netD_in_feats, lengths)
    loss_adv = (1 - D_fake) ** 2
    if D_mask is not None:
        loss_adv = loss_adv.masked_select(D_mask).mean()
    else:
        loss_adv = loss_adv.mean()

    # Pitch regularization
    # NOTE: l1 loss seems to be better than mse loss in my experiments
    # we could use l2 loss as suggested in the sinsy's paper
    loss_pitch = (pitch_reg_dyn_ws * lf0_residual.abs()).masked_select(mask).mean()

    loss = loss_feats + adv_weight * loss_adv + pitch_reg_weight * loss_pitch

    distortions = compute_distortions(
        pred_out_feats, out_feats, lengths, out_scaler, model_config
    )
    log_metrics.update(distortions)
    log_metrics.update(
        {
            "Loss": loss.item(),
            "Loss_Feats": loss_feats.item(),
            "Loss_Adv": loss_adv.item(),
            "Loss_Pitch": loss_pitch.item(),
            "Loss_Real": loss_real.item(),
            "Loss_Fake": loss_fake.item(),
            "Loss_D": loss_d.item(),
        }
    )

    if train:
        loss.backward()
        optG.step()

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
    adv_weight = config.train.adv_weight
    adv_streams = config.train.adv_streams
    if len(adv_streams) != len(config.model.stream_sizes):
        raise ValueError("adv_streams must be specified for all streams")

    for epoch in tqdm(range(1, config.train.nepochs + 1)):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            netG.train() if train else netG.eval()
            netD.train() if train else netD.eval()
            running_loss = 0
            running_metrics = {}
            for in_feats, out_feats, lengths in data_loaders[phase]:
                # NOTE: This is needed for pytorch's PackedSequence
                lengths, indices = torch.sort(lengths, dim=0, descending=True)
                in_feats, out_feats = (
                    in_feats[indices].to(device),
                    out_feats[indices].to(device),
                )
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

                loss, log_metrics = train_step(
                    config.model,
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
                    pitch_reg_weight,
                    adv_weight,
                    adv_streams,
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


@hydra.main(config_path="conf/train_resf0", config_name="config")
def my_app(config: DictConfig) -> None:
    # NOTE: set discriminator's in_dim automatically
    if config.model.netD.in_dim is None:
        D_in_dim = (
            np.asarray(config.model.stream_sizes) * np.asarray(config.train.adv_streams)
        ).sum()
        config.model.netD.in_dim = int(D_in_dim)

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
