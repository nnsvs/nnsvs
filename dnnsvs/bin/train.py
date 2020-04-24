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

from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from nnmnkwii.datasets import PaddedFileSourceDataset, MemoryCacheDataset
from dnnsvs.logger import getLogger

logger = None

use_cuda = torch.cuda.is_available()


class NpyFileSource(FileDataSource):
    def __init__(self, data_root, train):
        self.data_root = data_root
        self.train = train

    def collect_files(self):
        files = sorted(glob(join(self.data_root, "*.npy")))
        return files

    def collect_features(self, path):
        return np.load(path).astype(np.float32)


class PyTorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, lengths):
        self.X = X
        self.Y = Y
        if isinstance(lengths, list):
            lengths = np.array(lengths)[:,None]
        elif isinstance(lengths, np.ndarray):
            lengths = lengths[:,None]
        self.lengths = lengths

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        # TODO: move normalization to another script
        #x = minmax_scale(x, self.X_min, self.X_max, feature_range=(0.01, 0.99))
        #y = scale(y, self.Y_mean, self.Y_scale)
        l = torch.from_numpy(self.lengths[idx])
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y, l.view(1,1)
    def __len__(self):
        return len(self.X)


def get_data_loaders(config):
    data_loaders = {}
    utt_lengths = {}
    for phase in ["train_no_dev", "dev"]:
        in_dir = to_absolute_path(config.data[phase].in_dir)
        out_dir = to_absolute_path(config.data[phase].out_dir)
        train = phase.startswith("train")
        in_feats = FileSourceDataset(NpyFileSource(in_dir, train))
        utt_lengths = [len(x) for x in in_feats]
        padded_length = max(utt_lengths)

        in_feats = PaddedFileSourceDataset(
                NpyFileSource(in_dir, train), padded_length)
        out_feats = PaddedFileSourceDataset(
                NpyFileSource(out_dir, train), padded_length)
        in_feats = MemoryCacheDataset(in_feats, cache_size=10000)
        out_feats = MemoryCacheDataset(out_feats, cache_size=10000)

        dataset = PyTorchDataset(in_feats, out_feats, utt_lengths)
        data_loaders[phase] = data_utils.DataLoader(
            dataset, batch_size=config.data.batch_size,
            pin_memory=config.data.pin_memory,
            num_workers=config.data.num_workers, shuffle=train)

        for x, y, l in data_loaders[phase]:
            logger.info(f"{x.shape}, {y.shape}, {l.shape}")

    return data_loaders


def train_loop(config, model, optimizer, data_loaders):
    if use_cuda:
        model = model.cuda()

    criterion = nn.MSELoss()
    logger.info("Start utterance-wise training...")
    for epoch in tqdm(range(config.train.nepochs)):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            model.train() if train else model.eval()
            running_loss = 0
            for x, y, lengths in data_loaders[phase]:
                # Sort by lengths . This is needed for pytorch's PackedSequence
                sorted_lengths, indices = torch.sort(lengths.view(-1), dim=0, descending=True)
                # Get sorted batch
                x, y = x[indices], y[indices]
                # Trim outputs with max length
                y = y[:, :sorted_lengths[0]]
                if use_cuda:
                    x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                y_hat = model(x, sorted_lengths)
                loss = criterion(y_hat, y)
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

    model = hydra.utils.instantiate(config.model)
    optimizer_class = getattr(optim, config.optim.name)
    optimizer = optimizer_class(model.parameters(), **config.optim.params)
    data_loaders = get_data_loaders(config)
    train_loop(config, model, optimizer, data_loaders)

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