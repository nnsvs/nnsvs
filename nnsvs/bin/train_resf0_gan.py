from pathlib import Path

import hydra
import librosa
import librosa.display
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pysptk
import pyworld
import torch
from hydra.utils import to_absolute_path
from nnsvs.base import PredictionType
from nnsvs.gen import gen_world_params, get_windows
from nnsvs.multistream import (
    get_static_features,
    get_static_stream_sizes,
    multi_stream_mlpg,
    select_streams,
    split_streams,
)
from nnsvs.train_util import (
    check_resf0_config,
    compute_batch_pitch_regularization_weight,
    compute_distortions,
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

    # Ref: http://sython.org/papers/ASJ/saito2017asja.pdf
    # 0-th mgc with adversarial trainging affects speech quality
    # NOTE: assuming that the first stream contains mgc
    if mask_nth_mgc_for_adv_loss > 0:
        real_netD_in_feats = real_netD_in_feats[:, :, mask_nth_mgc_for_adv_loss:]
        fake_netD_in_feats = fake_netD_in_feats[:, :, mask_nth_mgc_for_adv_loss:]

    # Real
    D_real = netD(real_netD_in_feats, in_feats, lengths)
    # Fake
    D_fake_det = netD(fake_netD_in_feats.detach(), in_feats, lengths)

    # Mask (B, T, 1)
    mask = make_non_pad_mask(lengths).unsqueeze(-1).to(in_feats.device)
    if hasattr(netD, "downsample_scale"):
        D_mask = mask[:, :: netD.downsample_scale, :]
    else:
        if D_real[-1].shape[1] == out_feats.shape[1]:
            D_mask = mask
        else:
            D_mask = None

    # Update discriminator
    eps = 1e-7
    if gan_type == "lsgan":
        loss_real = (D_real[-1] - 1) ** 2
        loss_fake = D_fake_det[-1] ** 2
    elif gan_type == "vanilla-gan":
        loss_real = -torch.log(D_real[-1] + eps)
        loss_fake = -torch.log(1 - D_fake_det[-1] + eps)
    else:
        raise ValueError(f"Unknown gan type: {gan_type}")
    if D_mask is not None:
        loss_real = loss_real.masked_select(D_mask).mean()
        loss_fake = loss_fake.masked_select(D_mask).mean()
    else:
        loss_real = loss_real.mean()
        loss_fake = loss_fake.mean()
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
        for idx, pred_out_feats_ in enumerate(pred_out_feats):
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
    if gan_type == "lsgan":
        loss_adv = (1 - D_fake[-1]) ** 2
    elif gan_type == "vanilla-gan":
        loss_adv = -torch.log(D_fake[-1] + eps)
    else:
        raise ValueError(f"Unknown gan type: {gan_type}")
    if D_mask is not None:
        loss_adv = loss_adv.masked_select(D_mask).mean()
    else:
        loss_adv = loss_adv.mean()

    # Feature matching loss
    loss_fm = 0
    for fake_fmap, real_fmap in zip(D_fake[:-1], D_real[:-1]):
        loss_fm += F.l1_loss(fake_fmap, real_fmap.detach())
    loss_fm /= len(D_fake[:-1])

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


@torch.no_grad()
def eval_model(
    step, netG, in_feats, out_feats, lengths, model_config, out_scaler, writer
):
    utt_indices = [-1, -2, -3]
    utt_indices = utt_indices[: min(3, len(in_feats))]

    if np.any(model_config.has_dynamic_features):
        static_stream_sizes = get_static_stream_sizes(
            model_config.stream_sizes,
            model_config.has_dynamic_features,
            model_config.num_windows,
        )
    else:
        static_stream_sizes = model_config.stream_sizes

    for utt_idx in utt_indices:
        out_feats_ = out_scaler.inverse_transform(
            out_feats[utt_idx, : lengths[utt_idx]].unsqueeze(0)
        )
        mgc, lf0, vuv, bap = get_static_features(
            out_feats_,
            model_config.num_windows,
            model_config.stream_sizes,
            model_config.has_dynamic_features,
        )[:4]
        mgc = mgc.squeeze(0).cpu().numpy()
        lf0 = lf0.squeeze(0).cpu().numpy()
        vuv = vuv.squeeze(0).cpu().numpy()
        bap = bap.squeeze(0).cpu().numpy()

        sr = 48000
        f0, spectrogram, aperiodicity = gen_world_params(mgc, lf0, vuv, bap, sr)
        wav = pyworld.synthesize(f0, spectrogram, aperiodicity, sr, 5)
        group = f"utt{np.abs(utt_idx)}_reference"
        wav = wav / np.abs(wav).max() if np.max(wav) > 1.0 else wav
        writer.add_audio(group, wav, step, sr)

        # Run forward
        pred_out_feats, _ = netG(
            in_feats[utt_idx, : lengths[utt_idx]].unsqueeze(0), [lengths[utt_idx]]
        )
        if not isinstance(pred_out_feats, list):
            pred_out_feats = [pred_out_feats]

        for idx, pred_out_feats_ in enumerate(pred_out_feats):
            pred_out_feats_ = pred_out_feats_.squeeze(0).cpu().numpy()
            pred_out_feats_denorm = (
                out_scaler.inverse_transform(
                    torch.from_numpy(pred_out_feats_).to(in_feats.device)
                )
                .cpu()
                .numpy()
            )
            if np.any(model_config.has_dynamic_features):
                # (T, D_out) -> (T, static_dim)
                pred_out_feats_denorm = multi_stream_mlpg(
                    pred_out_feats_denorm,
                    (out_scaler.scale_ ** 2).cpu().numpy(),
                    get_windows(model_config.num_windows),
                    model_config.stream_sizes,
                    model_config.has_dynamic_features,
                )
            pred_mgc, pred_lf0, pred_vuv, pred_bap = split_streams(
                pred_out_feats_denorm, static_stream_sizes
            )[:4]

            # Generated sample
            f0, spectrogram, aperiodicity = gen_world_params(
                pred_mgc, pred_lf0, pred_vuv, pred_bap, sr
            )
            wav = pyworld.synthesize(f0, spectrogram, aperiodicity, sr, 5)
            wav = wav / np.abs(wav).max() if np.max(wav) > 1.0 else wav
            group = f"utt{np.abs(utt_idx)}_scale{idx}_generated"
            writer.add_audio(group, wav, step, sr)

            group = f"utt{np.abs(utt_idx)}_scale{idx}"
            _plot_spss_params(
                step,
                writer,
                mgc,
                lf0,
                vuv,
                bap,
                pred_mgc,
                pred_lf0,
                pred_vuv,
                pred_bap,
                group=group,
            )


@torch.no_grad()
def _plot_spss_params(
    step, writer, mgc, lf0, vuv, bap, pred_mgc, pred_lf0, pred_vuv, pred_bap, group
):
    sr = 48000
    fftlen = pyworld.get_cheaptrick_fft_size(sr)
    alpha = pysptk.util.mcepalpha(sr)
    hop_length = int(sr * 0.005)

    # F0
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    timeaxis = np.arange(len(lf0)) * 0.005
    f0 = np.exp(lf0)
    f0[vuv < 0.5] = 0
    pred_f0 = np.exp(pred_lf0)
    pred_f0[pred_vuv < 0.5] = 0
    ax.plot(timeaxis, f0, linewidth=2, label="F0 of natural speech")
    ax.plot(timeaxis, pred_f0, "--", linewidth=2, label="F0 of generated speech")
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlim(timeaxis[0], timeaxis[-1])
    plt.legend()
    plt.tight_layout()
    writer.add_figure(f"{group}/F0", fig, step)
    plt.close()

    # Spectrogram
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha).T
    mesh = librosa.display.specshow(
        librosa.power_to_db(np.abs(spectrogram), ref=np.max),
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap="viridis",
        ax=ax[0],
    )
    fig.colorbar(mesh, ax=ax[0], format="%+2.f dB")
    pred_spectrogram = pysptk.mc2sp(pred_mgc, fftlen=fftlen, alpha=alpha).T
    mesh = librosa.display.specshow(
        librosa.power_to_db(np.abs(pred_spectrogram), ref=np.max),
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap="viridis",
        ax=ax[1],
    )
    fig.colorbar(mesh, ax=ax[1], format="%+2.f dB")
    for a in ax:
        a.set_ylim(0, 14000)
    plt.tight_layout()
    writer.add_figure(f"{group}/Spectrogram", fig, step)
    plt.close()

    # Aperiodicity
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), sr, fftlen).T
    mesh = librosa.display.specshow(
        20 * np.log10(aperiodicity),
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        cmap="viridis",
        ax=ax[0],
    )
    fig.colorbar(mesh, ax=ax[0], format="%+2.f dB")
    pred_aperiodicity = pyworld.decode_aperiodicity(
        pred_bap.astype(np.float64), sr, fftlen
    ).T
    mesh = librosa.display.specshow(
        20 * np.log10(pred_aperiodicity),
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        cmap="viridis",
        ax=ax[1],
    )
    fig.colorbar(mesh, ax=ax[1], format="%+2.f dB")
    for a in ax:
        a.set_ylim(0, 14000)
    plt.tight_layout()
    writer.add_figure(f"{group}/Aperiodicity", fig, step)
    plt.close()


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
                    eval_model(
                        epoch,
                        netG,
                        in_feats,
                        out_feats,
                        lengths,
                        config.model,
                        out_scaler,
                        writer,
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
                    config.model,
                    config.train.optim,
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
                    fm_weight,
                    config.train.adv_use_static_feats_only,
                    config.train.mask_nth_mgc_for_adv_loss,
                    config.train.gan_type,
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
