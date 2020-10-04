# coding: utf-8

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf
import numpy as np
from glob import glob
from tqdm import tqdm
from os.path import basename, splitext, exists, join
import os
import shutil
import torch
from torch import nn
from torch.utils import data as data_utils
from torch.nn import functional as F
from torch import optim
from torch.backends import cudnn
from nnmnkwii.datasets import FileDataSource, FileSourceDataset, MemoryCacheDataset
from nnsvs.util import make_non_pad_mask
from nnsvs.multistream import split_streams
from nnsvs.logger import getLogger

logger = None
use_cuda = torch.cuda.is_available()


class NpyFileSource(FileDataSource):
    def __init__(self, data_root):
        self.data_root = data_root

    def collect_files(self):
        files = sorted(glob(join(self.data_root, "*-feats.npy")))
        return files

    def collect_features(self, path):
        return np.load(path).astype(np.float32)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


def _pad_2d(x, max_len, b_pad=0, constant_values=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=constant_values)
    return x


def collate_fn(batch):
    """Create batch

    Args:
        batch(tuple): List of tuples
            - x[0] (ndarray,int) : list of (T, D_in)
            - x[1] (ndarray,int) : list of (T, D_out)
    Returns:
        tuple: Tuple of batch
            - x (FloatTensor) : Network inputs (B, max(T), D_in)
            - y (FloatTensor)  : Network targets (B, max(T), D_out)
            - lengths (LongTensor): Input lengths
    """
    input_lengths = [len(x[0]) for x in batch]
    max_len = max(input_lengths)

    x_batch = np.array([_pad_2d(x[0], max_len) for x in batch], dtype=np.float32)
    y_batch = np.array([_pad_2d(x[1], max_len) for x in batch], dtype=np.float32)

    x_batch = torch.from_numpy(x_batch)
    y_batch = torch.from_numpy(y_batch)
    input_lengths = torch.LongTensor(input_lengths)

    return x_batch, y_batch, input_lengths


def get_data_loaders(config):
    data_loaders = {}
    for phase in ["train_no_dev", "dev"]:
        in_dir = to_absolute_path(config.data[phase].in_dir)
        out_dir = to_absolute_path(config.data[phase].out_dir)
        train = phase.startswith("train")
        in_feats = FileSourceDataset(NpyFileSource(in_dir))
        out_feats = FileSourceDataset(NpyFileSource(out_dir))

        in_feats = MemoryCacheDataset(in_feats, cache_size=10000)
        out_feats = MemoryCacheDataset(out_feats, cache_size=10000)

        dataset = Dataset(in_feats, out_feats)
        data_loaders[phase] = data_utils.DataLoader(
            dataset, batch_size=config.data.batch_size, collate_fn=collate_fn,
            pin_memory=config.data.pin_memory,
            num_workers=config.data.num_workers, shuffle=train)

        for x, y, l in data_loaders[phase]:
            logger.info(f"{x.shape}, {y.shape}, {l.shape}")

    return data_loaders


def save_checkpoint(config, model, optimizer, lr_scheduler, epoch, stream_id=None):
    out_dir = to_absolute_path(config.train.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if stream_id is not None:
        checkpoint_path = join(out_dir, f"stream_{stream_id}_checkpoint_epoch{epoch:04d}.pth")
        lastest_path = join(out_dir, f"stream_{stream_id}_latest.pth")
    else:
        checkpoint_path = join(out_dir, f"checkpoint_epoch{epoch:04d}.pth")
        lastest_path = join(out_dir, "latest.pth")

    torch.save({
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict(),
    }, checkpoint_path)
    logger.info(f"Checkpoint is saved at {checkpoint_path}")
    shutil.copyfile(checkpoint_path, lastest_path)


def save_best_checkpoint(config, model, optimizer, best_loss, stream_id=None):
    out_dir = to_absolute_path(config.train.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if stream_id is not None:
        checkpoint_path = join(out_dir, f"stream_{stream_id}_best_loss.pth")
    else:
        checkpoint_path = join(out_dir, "best_loss.pth")
        
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, checkpoint_path)
    logger.info(f"[Best loss {best_loss}: checkpoint is saved at {checkpoint_path}")


def get_stream_weight(stream_weights, stream_sizes):
    if stream_weights is not None:
        assert len(stream_weights) == len(stream_sizes)
        return torch.tensor(stream_weights)

    S = sum(stream_sizes)
    w = torch.tensor(stream_sizes).float() / S
    return w


def train_loop(config, device, model, optimizer, lr_scheduler, data_loaders, stream_id=None):
    criterion = nn.MSELoss(reduction="none")
    logger.info("Start utterance-wise training...")

    stream_weights = get_stream_weight(
        config.model.stream_weights, config.model.stream_sizes).to(device)

    best_loss = 10000000
    for epoch in tqdm(range(1, config.train.nepochs + 1)):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            model.train() if train else model.eval()
            running_loss = 0
            for x, y, lengths in data_loaders[phase]:
                # Sort by lengths . This is needed for pytorch's PackedSequence
                sorted_lengths, indices = torch.sort(lengths, dim=0, descending=True)
                x, y = x[indices].to(device), y[indices].to(device)

                if config.model.stream_wise_training and stream_id is not None:
                    y = split_streams(y, config.model.stream_sizes)[stream_id]
                    
                optimizer.zero_grad()

                # Run forwaard
                y_hat = model(x, sorted_lengths)

                # Compute loss
                mask = make_non_pad_mask(sorted_lengths).unsqueeze(-1).to(device)

                if not config.model.stream_wise_training and config.train.stream_wise_loss:
                    # Strean-wise loss
                    streams = split_streams(y, config.model.stream_sizes)
                    streams_hat = split_streams(y_hat, config.model.stream_sizes)
                    loss = 0
                    for s_hat, s, sw in zip(streams_hat, streams, stream_weights):
                        s_hat_mask = s_hat.masked_select(mask)
                        s_mask = s.masked_select(mask)
                        loss += sw * criterion(s_hat_mask, s_mask).mean()
                else:
                    # Joint modeling
                    y_hat = y_hat.masked_select(mask)
                    y = y.masked_select(mask)
                    loss = criterion(y_hat, y).mean()

                if train:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
            ave_loss = running_loss / len(data_loaders[phase])
            logger.info(f"[{phase}] [Epoch {epoch}]: loss {ave_loss}")
            if not train and ave_loss < best_loss:
                best_loss = ave_loss
                save_best_checkpoint(config, model, optimizer, best_loss, stream_id)

        # step per each epoch (may consider updating per iter.)
        lr_scheduler.step()

        if epoch % config.train.checkpoint_epoch_interval == 0:
            save_checkpoint(config, model, optimizer, lr_scheduler, epoch, stream_id)

    # save at last epoch
    save_checkpoint(config, model, optimizer, lr_scheduler, config.train.nepochs, stream_id)
    logger.info(f"The best loss was {best_loss}")

    return model

def setup(config, device, stream_id=None):

    print(stream_id)
    if stream_id is not None:
        model = hydra.utils.instantiate(config.model.models[stream_id].netG).to(device)
    else:
        model = hydra.utils.instantiate(config.model.netG).to(device)
        
    # Optimizer
    optimizer_class = getattr(optim, config.optim.optimizer.name)
    optimizer = optimizer_class(model.parameters(), **config.optim.optimizer.params)

    # Scheduler
    lr_scheduler_class = getattr(optim.lr_scheduler, config.optim.lr_scheduler.name)
    lr_scheduler = lr_scheduler_class(optimizer, **config.optim.lr_scheduler.params)

    # Resume
    checkpoint=None
    if type(config.resume.checkpoint) is ListConfig and \
       len(config.resume.checkpoint) == len(config.model.stream_sizes):
        logger.info(f"Load weights from {config.resume.checkpoint[stream_id]}")
        checkpoint = torch.load(to_absolute_path(config.resume.checkpoint[stream_id]))
    elif type(config.resume.checkpoint) is str and \
         len(config.resume.checkpoint) > 0:
        logger.info(f"Load weights from {config.resume.checkpoint}")
        checkpoint = torch.load(to_absolute_path(config.resume.checkpoint))

    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"])

    if config.resume.load_optimizer:
        logger.info("Load optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])

    return model, optimizer, lr_scheduler

@hydra.main(config_path="conf/train/config.yaml")
def my_app(config : DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(config.pretty())

    if use_cuda:
        from torch.backends import cudnn
        cudnn.benchmark = config.cudnn.benchmark
        cudnn.deterministic = config.cudnn.deterministic
        logger.info(f"cudnn.deterministic: {cudnn.deterministic}")
        logger.info(f"cudnn.benchmark: {cudnn.benchmark}")

    device = torch.device("cuda" if use_cuda else "cpu")
    
    data_loaders = get_data_loaders(config)

    if config.model.stream_wise_training and \
    len(config.model.models) == len(config.model.stream_sizes):
        logger.info(f"stream-wise training is enabled")
        for stream_id in range(len(config.model.stream_sizes)):
            model, optimizer, lr_scheduler = setup(config, device, stream_id)
            # Run training loop
            train_loop(config, device, model, optimizer, lr_scheduler, data_loaders, stream_id)
            del model, optimizer, lr_scheduler
    else:
        model, optimizer, lr_scheduler = setup(config, device, None)
        # Run training loop
        train_loop(config, device, model, optimizer, lr_scheduler, data_loaders, None)
        
    # Save model definition
    out_dir = to_absolute_path(config.train.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(join(out_dir, "model.yaml"), "w") as f:
        OmegaConf.save(config.model, f)

def entry():
    my_app()


if __name__ == "__main__":
    my_app()
