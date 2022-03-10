from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from nnsvs.base import PredictionType
from nnsvs.mdn import mdn_loss
from nnsvs.multistream import split_streams
from nnsvs.train_util import get_stream_weight, save_checkpoint, setup
from nnsvs.util import make_non_pad_mask
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm


def train_step(
    model,
    optimizer,
    train,
    in_feats,
    out_feats,
    lengths,
    stream_wise_loss=False,
    stream_weights=None,
    stream_sizes=None,
):
    optimizer.zero_grad()

    criterion = nn.MSELoss(reduction="none")

    # Apply preprocess if required (e.g., FIR filter for shallow AR)
    # defaults to no-op
    out_feats = model.preprocess_target(out_feats)

    # Run forward
    outs = model(in_feats, lengths)

    # Compute loss
    if model.prediction_type() == PredictionType.PROBABILISTIC:
        pi, sigma, mu = outs

        # (B, max(T)) or (B, max(T), D_out)
        mask = make_non_pad_mask(lengths).to(in_feats.device)
        mask = mask.unsqueeze(-1) if len(pi.shape) == 4 else mask

        # Compute loss and apply mask
        loss = mdn_loss(pi, sigma, mu, out_feats, reduce=False)
        loss = loss.masked_select(mask).mean()
    else:
        pred_out_feats = outs

        mask = make_non_pad_mask(lengths).unsqueeze(-1).to(in_feats.device)

        if stream_wise_loss:
            w = get_stream_weight(stream_weights, stream_sizes).to(in_feats.device)
            streams = split_streams(out_feats, stream_sizes)
            pred_streams = split_streams(pred_out_feats, stream_sizes)
            loss = 0
            for pred_stream, stream, sw in zip(pred_streams, streams, w):
                loss += (
                    sw
                    * criterion(
                        pred_stream.masked_select(mask), stream.masked_select(mask)
                    ).mean()
                )
        else:
            loss = criterion(
                pred_out_feats.masked_select(mask), out_feats.masked_select(mask)
            ).mean()

    if train:
        loss.backward()
        optimizer.step()

    return loss


def train_loop(
    config,
    logger,
    device,
    model,
    optimizer,
    lr_scheduler,
    data_loaders,
    writer,
    in_scaler,
    out_scaler,
):
    out_dir = Path(to_absolute_path(config.train.out_dir))
    best_loss = torch.finfo(torch.float32).max

    for epoch in tqdm(range(1, config.train.nepochs + 1)):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            model.train() if train else model.eval()
            running_loss = 0
            for in_feats, out_feats, lengths in data_loaders[phase]:
                # NOTE: This is needed for pytorch's PackedSequence
                lengths, indices = torch.sort(lengths, dim=0, descending=True)
                in_feats, out_feats = (
                    in_feats[indices].to(device),
                    out_feats[indices].to(device),
                )
                loss = train_step(model, optimizer, train, in_feats, out_feats, lengths)
                running_loss += loss.item()
            ave_loss = running_loss / len(data_loaders[phase])
            writer.add_scalar(f"Loss/{phase}", ave_loss, epoch)

            ave_loss = running_loss / len(data_loaders[phase])
            logger.info("[%s] [Epoch %s]: loss %s", phase, epoch, ave_loss)
            if not train and ave_loss < best_loss:
                best_loss = ave_loss
                save_checkpoint(
                    logger, out_dir, model, optimizer, lr_scheduler, epoch, is_best=True
                )

        lr_scheduler.step()
        if epoch % config.train.checkpoint_epoch_interval == 0:
            save_checkpoint(
                logger, out_dir, model, optimizer, lr_scheduler, epoch, is_best=False
            )

    save_checkpoint(
        logger, out_dir, model, optimizer, lr_scheduler, config.train.nepochs
    )
    logger.info("The best loss was %s", best_loss)


@hydra.main(config_path="conf/train", config_name="config")
def my_app(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        model,
        optimizer,
        lr_scheduler,
        data_loaders,
        writer,
        logger,
        in_scaler,
        out_scaler,
    ) = setup(config, device)
    train_loop(
        config,
        logger,
        device,
        model,
        optimizer,
        lr_scheduler,
        data_loaders,
        writer,
        in_scaler,
        out_scaler,
    )


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
