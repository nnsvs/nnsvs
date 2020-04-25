# coding: utf-8

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import numpy as np
from glob import glob
from tqdm import tqdm
from os.path import basename, splitext, exists, join
import os
import torch
from torch import nn
from torch.utils import data as data_utils
from torch.nn import functional as F
from torch import optim
from torch.backends import cudnn
from nnmnkwii.datasets import FileDataSource, FileSourceDataset, MemoryCacheDataset
from dnnsvs.util import make_non_pad_mask
from dnnsvs.logger import getLogger

logger = None

use_cuda = torch.cuda.is_available()


class NpyFileSource(FileDataSource):
    def __init__(self, data_root):
        self.data_root = data_root

    def collect_files(self):
        files = sorted(glob(join(self.data_root, "*.npy")))
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
    utt_lengths = {}
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


def train_loop(config, device, model, optimizer, data_loaders):
    criterion = nn.MSELoss(reduction="none")
    logger.info("Start utterance-wise training...")
    for epoch in tqdm(range(config.train.nepochs)):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            model.train() if train else model.eval()
            running_loss = 0
            for x, y, lengths in data_loaders[phase]:
                # Sort by lengths . This is needed for pytorch's PackedSequence
                sorted_lengths, indices = torch.sort(lengths, dim=0, descending=True)
                x, y = x[indices].to(device), y[indices].to(device)
                # Make mask for padding
                mask = make_non_pad_mask(sorted_lengths).unsqueeze(-1).to(device)

                optimizer.zero_grad()

                # Run forwaard
                y_hat = model(x, sorted_lengths)

                # Compute loss
                if True:
                    y_hat = y_hat.masked_select(mask)
                    y = y.masked_select(mask)
                loss = criterion(y_hat, y).mean()
                if train:
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
            ave_loss = running_loss / len(data_loaders[phase])
            logger.info(f"[{phase}] [Epoch {epoch}]: loss {ave_loss}")

    return model


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

    model = hydra.utils.instantiate(config.model).to(device)
    optimizer_class = getattr(optim, config.optim.name)
    optimizer = optimizer_class(model.parameters(), **config.optim.params)
    data_loaders = get_data_loaders(config)
    train_loop(config, device, model, optimizer, data_loaders)

    # save last checkpoint
    out_dir = to_absolute_path(config.train.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = join(out_dir, "latest.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, checkpoint_path)
    logger.info(f"Checkpoint is saved at {checkpoint_path}")


def entry():
    my_app()


if __name__ == "__main__":
    my_app()