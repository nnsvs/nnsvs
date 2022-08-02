import itertools
import shutil
from glob import glob
from os.path import join
from pathlib import Path

import hydra
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pysptk
import pyworld
import torch
from hydra.utils import get_original_cwd, to_absolute_path
from nnmnkwii import metrics
from nnmnkwii.datasets import FileDataSource, FileSourceDataset, MemoryCacheDataset
from nnsvs.base import PredictionType
from nnsvs.gen import gen_world_params, get_windows
from nnsvs.logger import getLogger
from nnsvs.mdn import mdn_get_most_probable_sigma_and_mu
from nnsvs.multistream import (
    get_static_features,
    get_static_stream_sizes,
    multi_stream_mlpg,
    split_streams,
)
from nnsvs.pitch import lowpass_filter, nonzero_segments
from nnsvs.util import MinMaxScaler, StandardScaler, init_seed, pad_2d
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.utils import data as data_utils
from torch.utils.tensorboard import SummaryWriter

plt.style.use("seaborn-whitegrid")


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                mlflow.log_param(f"{parent_name}.{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f"{parent_name}.{i}", v)


def num_trainable_params(model):
    """Count the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): Model to count the number of trainable parameters.

    Returns:
        int: Number of trainable parameters.
    """
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])


class NpyFileSource(FileDataSource):
    """Load numpy files from a directory

    Args:
        data_root (str): Path to the directory containing the numpy files.
        filter_long_segments (bool): Whether to filter out long segments.
        filter_num_frames (int): Minimum number of frames to keep a segment.

    """

    def __init__(
        self, data_root, logger, filter_long_segments=False, filter_num_frames=6000
    ):
        self.data_root = data_root
        self.logger = logger
        self.filter_long_segments = filter_long_segments
        self.filter_num_frames = filter_num_frames

    def collect_files(self):
        files = sorted(glob(join(self.data_root, "*-feats.npy")))

        if self.filter_long_segments:
            valid_files = []

            num_filtered = 0
            for path in files:
                length = len(np.load(path))
                if length < self.filter_num_frames:
                    valid_files.append(path)
                else:
                    self.logger.info(f"Filtered: {path} is too long: {length}")
                    num_filtered += 1
            if num_filtered > 0:
                self.logger.info(f"Filtered {num_filtered} files")

            # Print stats of lengths
            lengths = [len(np.load(f)) for f in files]
            self.logger.debug(f"[before] Size of dataset: {len(files)}")
            self.logger.debug(f"[before] maximum length: {max(lengths)}")
            self.logger.debug(f"[before] minimum length: {min(lengths)}")
            self.logger.debug(f"[before] mean length: {np.mean(lengths)}")
            self.logger.debug(f"[before] std length: {np.std(lengths)}")
            self.logger.debug(f"[before] median length: {np.median(lengths)}")

            files = valid_files

            lengths = [len(np.load(f)) for f in files]
            self.logger.debug(f"[after] Size of dataset: {len(files)}")
            self.logger.debug(f"[after] maximum length: {max(lengths)}")
            self.logger.debug(f"[after] minimum length: {min(lengths)}")
            self.logger.debug(f"[after] mean length: {np.mean(lengths)}")
            self.logger.debug(f"[after] std length: {np.std(lengths)}")
            self.logger.debug(f"[after] median length: {np.median(lengths)}")

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


def ensure_divisible_by(feats, N):
    """Ensure that the number of frames is divisible by N.

    Args:
        feats (np.ndarray): Input features.
        N (int): Target number of frames.

    Returns:
        np.ndarray: Input features with number of frames divisible by N.
    """
    if N == 1:
        return feats
    mod = len(feats) % N
    if mod != 0:
        feats = feats[: len(feats) - mod]
    return feats


def collate_fn_default(batch, reduction_factor=1):
    """Create batch

    Args:
        batch(tuple): List of tuples
            - x[0] (ndarray,int) : list of (T, D_in)
            - x[1] (ndarray,int) : list of (T, D_out)
        reduction_factor (int): Reduction factor.

    Returns:
        tuple: Tuple of batch
            - x (FloatTensor) : Network inputs (B, max(T), D_in)
            - y (FloatTensor)  : Network targets (B, max(T), D_out)
            - lengths (LongTensor): Input lengths
    """
    lengths = [len(ensure_divisible_by(x[0], reduction_factor)) for x in batch]
    max_len = max(lengths)
    x_batch = torch.stack(
        [
            torch.from_numpy(
                pad_2d(ensure_divisible_by(x[0], reduction_factor), max_len)
            )
            for x in batch
        ]
    )
    y_batch = torch.stack(
        [
            torch.from_numpy(
                pad_2d(ensure_divisible_by(x[1], reduction_factor), max_len)
            )
            for x in batch
        ]
    )
    l_batch = torch.tensor(lengths, dtype=torch.long)
    return x_batch, y_batch, l_batch


def collate_fn_random_segments(batch, max_time_frames=256):
    """Collate function with random segments

    Use segmented frames instead of padded entire frames. No padding is performed.

    .. warning::

        max_time_frames must be larger than the shortest sequence in the training data.

    Args:
        batch (tuple): tupls of lit
            - x[0] (ndarray,int) : list of (T, D_in)
            - x[1] (ndarray,int) : list of (T, D_out)
        max_time_frames (int, optional): Number of time frames. Defaults to 256.

    Returns:
        tuple: Tuple of batch
            - x (FloatTensor) : Network inputs (B, max(T), D_in)
            - y (FloatTensor)  : Network targets (B, max(T), D_out)
            - lengths (LongTensor): Input lengths
    """
    xs, ys = [b[0] for b in batch], [b[1] for b in batch]
    lengths = [len(x[0]) for x in batch]

    start_frames = np.array(
        [np.random.randint(0, xl - max_time_frames) for xl in lengths]
    )
    starts = start_frames
    ends = starts + max_time_frames
    x_cut = [torch.from_numpy(x[s:e]) for x, s, e in zip(xs, starts, ends)]
    y_cut = [torch.from_numpy(y[s:e]) for y, s, e in zip(ys, starts, ends)]

    x_batch = torch.stack(x_cut).float()
    y_batch = torch.stack(y_cut).float()
    # NOTE: we don't actually need lengths since we don't perform padding
    # but just for consistency with collate_fn_default
    l_batch = torch.tensor([max_time_frames] * len(lengths), dtype=torch.long)

    return x_batch, y_batch, l_batch


def get_data_loaders(data_config, collate_fn, logger):
    """Get data loaders for training and validation.

    Args:
        data_config (dict): Data configuration.
        collate_fn (callable): Collate function.
        logger (logging.Logger): Logger.

    Returns:
        dict: Data loaders.
    """
    if "filter_long_segments" not in data_config:
        logger.warning(
            "filter_long_segments is not found in the data config. Consider set it explicitly."
        )
        logger.info("Disable filtering for long segments.")
        filter_long_segments = False
    else:
        filter_long_segments = data_config.filter_long_segments

    if "filter_num_frames" not in data_config:
        logger.warning(
            "filter_num_frames is not found in the data config. Consider set it explicitly."
        )
        filter_num_frames = 6000
    else:
        filter_num_frames = data_config.filter_num_frames

    data_loaders = {}
    for phase in ["train_no_dev", "dev"]:
        in_dir = to_absolute_path(data_config[phase].in_dir)
        out_dir = to_absolute_path(data_config[phase].out_dir)
        train = phase.startswith("train")
        in_feats = FileSourceDataset(
            NpyFileSource(
                in_dir,
                logger,
                filter_long_segments=filter_long_segments,
                filter_num_frames=filter_num_frames,
            )
        )
        out_feats = FileSourceDataset(
            NpyFileSource(
                out_dir,
                logger,
                filter_long_segments=filter_long_segments,
                filter_num_frames=filter_num_frames,
            )
        )

        in_feats = MemoryCacheDataset(in_feats, cache_size=10000)
        out_feats = MemoryCacheDataset(out_feats, cache_size=10000)

        dataset = Dataset(in_feats, out_feats)
        data_loaders[phase] = data_utils.DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            collate_fn=collate_fn,
            pin_memory=data_config.pin_memory,
            num_workers=data_config.num_workers,
            shuffle=train,
        )

    return data_loaders


def set_epochs_based_on_max_steps_(train_config, steps_per_epoch, logger):
    """Set epochs based on max steps.

    Args:
        train_config (TrainConfig): Train config.
        steps_per_epoch (int): Number of steps per epoch.
        logger (logging.Logger): Logger.
    """
    if "max_train_steps" not in train_config:
        logger.warning("max_train_steps is not found in the train config.")
        return

    logger.info(f"Number of iterations per epoch: {steps_per_epoch}")

    if train_config.max_train_steps < 0:
        # Set max_train_steps based on nepochs
        max_train_steps = train_config.nepochs * steps_per_epoch
        train_config.max_train_steps = max_train_steps
        logger.info(
            "Number of max_train_steps is set based on nepochs: {}".format(
                max_train_steps
            )
        )
    else:
        # Set nepochs based on max_train_steps
        max_train_steps = train_config.max_train_steps
        epochs = int(np.ceil(max_train_steps / steps_per_epoch))
        train_config.nepochs = epochs
        logger.info(
            "Number of epochs is set based on max_train_steps: {}".format(epochs)
        )

    logger.info(f"Number of epochs: {train_config.nepochs}")
    logger.info(f"Number of iterations: {train_config.max_train_steps}")


def save_checkpoint(
    logger,
    out_dir,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    is_best=False,
    postfix="",
):
    """Save a checkpoint.

    Args:
        logger (logging.Logger): Logger.
        out_dir (str): Output directory.
        model (nn.Module): Model.
        optimizer (Optimizer): Optimizer.
        lr_scheduler (LRScheduler): Learning rate scheduler.
        epoch (int): Current epoch.
        is_best (bool, optional): Whether or not the current model is the best.
            Defaults to False.
        postfix (str, optional): Postfix. Defaults to "".
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    out_dir.mkdir(parents=True, exist_ok=True)
    if is_best:
        path = out_dir / f"best_loss{postfix}.pth"
    else:
        path = out_dir / "epoch{:04d}{}.pth".format(epoch, postfix)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "lr_scheduler_state": lr_scheduler.state_dict(),
        },
        path,
    )

    logger.info(f"Saved checkpoint at {path}")
    if not is_best:
        shutil.copyfile(path, out_dir / f"latest{postfix}.pth")


def get_stream_weight(stream_weights, stream_sizes):
    if stream_weights is not None:
        assert len(stream_weights) == len(stream_sizes)
        return torch.tensor(stream_weights)

    S = sum(stream_sizes)
    w = torch.tensor(stream_sizes).float() / S
    return w


def _instantiate_optim(optim_config, model):
    # Optimizer
    optimizer_class = getattr(optim, optim_config.optimizer.name)
    optimizer = optimizer_class(model.parameters(), **optim_config.optimizer.params)

    # Scheduler
    lr_scheduler_class = getattr(optim.lr_scheduler, optim_config.lr_scheduler.name)
    lr_scheduler = lr_scheduler_class(optimizer, **optim_config.lr_scheduler.params)

    return optimizer, lr_scheduler


def _instantiate_optim_cyclegan(optim_config, netG_A2B, netG_B2A):
    # Optimizer
    optimizer_class = getattr(optim, optim_config.optimizer.name)

    optimizer = optimizer_class(
        itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
        **optim_config.optimizer.params,
    )

    # Scheduler
    lr_scheduler_class = getattr(optim.lr_scheduler, optim_config.lr_scheduler.name)
    lr_scheduler = lr_scheduler_class(optimizer, **optim_config.lr_scheduler.params)

    return optimizer, lr_scheduler


def _resume(logger, resume_config, model, optimizer, lr_scheduler):
    if resume_config.checkpoint is not None and len(resume_config.checkpoint) > 0:
        logger.info("Load weights from %s", resume_config.checkpoint)
        checkpoint = torch.load(to_absolute_path(resume_config.checkpoint))
        model.load_state_dict(checkpoint["state_dict"])
        if resume_config.load_optimizer:
            logger.info("Load optimizer state")
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])


def setup(config, device, collate_fn=collate_fn_default):
    """Setup for training

    Args:
        config (dict): configuration for training
        device (torch.device): device to use for training
        collate_fn (callable, optional): collate function. Defaults to collate_fn_default.

    Returns:
        (tuple): tuple containing model, optimizer, learning rate scheduler,
            data loaders, tensorboard writer, logger, and scalers.
    """
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    logger.info(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        from torch.backends import cudnn

        cudnn.benchmark = config.train.cudnn.benchmark
        cudnn.deterministic = config.train.cudnn.deterministic
        logger.info(f"cudnn.deterministic: {cudnn.deterministic}")
        logger.info(f"cudnn.benchmark: {cudnn.benchmark}")
        if torch.backends.cudnn.version() is not None:
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")

    logger.info(f"Random seed: {config.seed}")
    init_seed(config.seed)

    if config.train.use_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        logger.info("Set to use torch.autograd.detect_anomaly")

    if "use_amp" in config.train and config.train.use_amp:
        logger.info("Use mixed precision training")
        grad_scaler = GradScaler()
    else:
        grad_scaler = None

    # Model
    model = hydra.utils.instantiate(config.model.netG).to(device)
    logger.info(
        "Number of trainable params: {:.3f} million".format(
            num_trainable_params(model) / 1000000.0
        )
    )
    logger.info(model)

    # Optimizer
    optimizer_class = getattr(optim, config.train.optim.optimizer.name)
    optimizer = optimizer_class(
        model.parameters(), **config.train.optim.optimizer.params
    )

    # Scheduler
    lr_scheduler_class = getattr(
        optim.lr_scheduler, config.train.optim.lr_scheduler.name
    )
    lr_scheduler = lr_scheduler_class(
        optimizer, **config.train.optim.lr_scheduler.params
    )

    # DataLoader
    data_loaders = get_data_loaders(config.data, collate_fn, logger)

    set_epochs_based_on_max_steps_(
        config.train, len(data_loaders["train_no_dev"]), logger
    )

    # Resume
    if (
        config.train.resume.checkpoint is not None
        and len(config.train.resume.checkpoint) > 0
    ):
        logger.info("Load weights from %s", config.train.resume.checkpoint)
        checkpoint = torch.load(to_absolute_path(config.train.resume.checkpoint))
        model.load_state_dict(checkpoint["state_dict"])
        if config.train.resume.load_optimizer:
            logger.info("Load optimizer state")
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])

    if config.data_parallel:
        model = nn.DataParallel(model)

    # Mlflow
    if config.mlflow.enabled:
        mlflow.set_tracking_uri("file://" + get_original_cwd() + "/mlruns")
        mlflow.set_experiment(config.mlflow.experiment)
        # NOTE: disable tensorboard if mlflow is enabled
        writer = None
        logger.info("Using mlflow instead of tensorboard")
    else:
        # Tensorboard
        writer = SummaryWriter(to_absolute_path(config.train.log_dir))

    # Scalers
    if "in_scaler_path" in config.data and config.data.in_scaler_path is not None:
        in_scaler = joblib.load(to_absolute_path(config.data.in_scaler_path))
        in_scaler = MinMaxScaler(
            in_scaler.min_, in_scaler.scale_, in_scaler.data_min_, in_scaler.data_max_
        )
    else:
        in_scaler = None
    if "out_scaler_path" in config.data and config.data.out_scaler_path is not None:
        out_scaler = joblib.load(to_absolute_path(config.data.out_scaler_path))
        out_scaler = StandardScaler(
            out_scaler.mean_, out_scaler.var_, out_scaler.scale_
        )
    else:
        out_scaler = None

    return (
        model,
        optimizer,
        lr_scheduler,
        grad_scaler,
        data_loaders,
        writer,
        logger,
        in_scaler,
        out_scaler,
    )


def setup_gan(config, device, collate_fn=collate_fn_default):
    """Setup for training GAN

    Args:
        config (dict): configuration for training
        device (torch.device): device to use for training
        collate_fn (callable, optional): collate function. Defaults to collate_fn_default.

    Returns:
        (tuple): tuple containing model, optimizer, learning rate scheduler,
            data loaders, tensorboard writer, logger, and scalers.
    """
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    logger.info(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        from torch.backends import cudnn

        cudnn.benchmark = config.train.cudnn.benchmark
        cudnn.deterministic = config.train.cudnn.deterministic
        logger.info(f"cudnn.deterministic: {cudnn.deterministic}")
        logger.info(f"cudnn.benchmark: {cudnn.benchmark}")
        if torch.backends.cudnn.version() is not None:
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")

    logger.info(f"Random seed: {config.seed}")
    init_seed(config.seed)

    if config.train.use_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        logger.info("Set to use torch.autograd.detect_anomaly")

    if "use_amp" in config.train and config.train.use_amp:
        logger.info("Use mixed precision training")
        grad_scaler = GradScaler()
    else:
        grad_scaler = None

    # Model G
    netG = hydra.utils.instantiate(config.model.netG).to(device)
    logger.info(
        "[Generator] Number of trainable params: {:.3f} million".format(
            num_trainable_params(netG) / 1000000.0
        )
    )
    logger.info(netG)
    # Optimizer and LR scheduler for G
    optG, schedulerG = _instantiate_optim(config.train.optim.netG, netG)

    # Model D
    netD = hydra.utils.instantiate(config.model.netD).to(device)
    logger.info(
        "[Discriminator] Number of trainable params: {:.3f} million".format(
            num_trainable_params(netD) / 1000000.0
        )
    )
    logger.info(netD)
    # Optimizer and LR scheduler for D
    optD, schedulerD = _instantiate_optim(config.train.optim.netD, netD)

    # DataLoader
    data_loaders = get_data_loaders(config.data, collate_fn, logger)

    set_epochs_based_on_max_steps_(
        config.train, len(data_loaders["train_no_dev"]), logger
    )

    # Resume
    _resume(logger, config.train.resume.netG, netG, optG, schedulerG)
    _resume(logger, config.train.resume.netD, netD, optD, schedulerD)

    if config.data_parallel:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    # Mlflow
    if config.mlflow.enabled:
        mlflow.set_tracking_uri("file://" + get_original_cwd() + "/mlruns")
        mlflow.set_experiment(config.mlflow.experiment)
        # NOTE: disable tensorboard if mlflow is enabled
        writer = None
        logger.info("Using mlflow instead of tensorboard")
    else:
        # Tensorboard
        writer = SummaryWriter(to_absolute_path(config.train.log_dir))

    # Scalers
    if "in_scaler_path" in config.data and config.data.in_scaler_path is not None:
        in_scaler = joblib.load(to_absolute_path(config.data.in_scaler_path))
        if isinstance(in_scaler, SKMinMaxScaler):
            in_scaler = MinMaxScaler(
                in_scaler.min_,
                in_scaler.scale_,
                in_scaler.data_min_,
                in_scaler.data_max_,
            )
    else:
        in_scaler = None
    if "out_scaler_path" in config.data and config.data.out_scaler_path is not None:
        out_scaler = joblib.load(to_absolute_path(config.data.out_scaler_path))
        out_scaler = StandardScaler(
            out_scaler.mean_, out_scaler.var_, out_scaler.scale_
        )
    else:
        out_scaler = None

    return (
        (netG, optG, schedulerG),
        (netD, optD, schedulerD),
        grad_scaler,
        data_loaders,
        writer,
        logger,
        in_scaler,
        out_scaler,
    )


def setup_cyclegan(config, device, collate_fn=collate_fn_default):
    """Setup for training CycleGAN

    Args:
        config (dict): configuration for training
        device (torch.device): device to use for training
        collate_fn (callable, optional): collate function. Defaults to collate_fn_default.

    Returns:
        (tuple): tuple containing model, optimizer, learning rate scheduler,
            data loaders, tensorboard writer, logger, and scalers.
    """
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    logger.info(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        from torch.backends import cudnn

        cudnn.benchmark = config.train.cudnn.benchmark
        cudnn.deterministic = config.train.cudnn.deterministic
        logger.info(f"cudnn.deterministic: {cudnn.deterministic}")
        logger.info(f"cudnn.benchmark: {cudnn.benchmark}")
        if torch.backends.cudnn.version() is not None:
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")

    logger.info(f"Random seed: {config.seed}")
    init_seed(config.seed)

    if config.train.use_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        logger.info("Set to use torch.autograd.detect_anomaly")

    if "use_amp" in config.train and config.train.use_amp:
        logger.info("Use mixed precision training")
        grad_scaler = GradScaler()
    else:
        grad_scaler = None

    # Model G
    netG_A2B = hydra.utils.instantiate(config.model.netG).to(device)
    netG_B2A = hydra.utils.instantiate(config.model.netG).to(device)
    logger.info(
        "[Generator] Number of trainable params: {:.3f} million".format(
            num_trainable_params(netG_A2B) / 1000000.0
        )
    )
    logger.info(netG_A2B)
    # Optimizer and LR scheduler for G
    optG, schedulerG = _instantiate_optim_cyclegan(
        config.train.optim.netG, netG_A2B, netG_B2A
    )

    # Model D
    netD_A = hydra.utils.instantiate(config.model.netD).to(device)
    netD_B = hydra.utils.instantiate(config.model.netD).to(device)
    logger.info(
        "[Discriminator] Number of trainable params: {:.3f} million".format(
            num_trainable_params(netD_A) / 1000000.0
        )
    )
    logger.info(netD_A)
    # Optimizer and LR scheduler for D
    optD, schedulerD = _instantiate_optim_cyclegan(
        config.train.optim.netD, netD_A, netD_B
    )

    # DataLoader
    data_loaders = get_data_loaders(config.data, collate_fn, logger)

    set_epochs_based_on_max_steps_(
        config.train, len(data_loaders["train_no_dev"]), logger
    )

    # Resume
    # TODO
    # _resume(logger, config.train.resume.netG, netG, optG, schedulerG)
    # _resume(logger, config.train.resume.netD, netD, optD, schedulerD)

    if config.data_parallel:
        netG_A2B = nn.DataParallel(netG_A2B)
        netG_B2A = nn.DataParallel(netG_B2A)
        netD_A = nn.DataParallel(netD_A)
        netD_B = nn.DataParallel(netD_B)

    # Mlflow
    if config.mlflow.enabled:
        mlflow.set_tracking_uri("file://" + get_original_cwd() + "/mlruns")
        mlflow.set_experiment(config.mlflow.experiment)
        # NOTE: disable tensorboard if mlflow is enabled
        writer = None
        logger.info("Using mlflow instead of tensorboard")
    else:
        # Tensorboard
        writer = SummaryWriter(to_absolute_path(config.train.log_dir))

    # Scalers
    if "in_scaler_path" in config.data and config.data.in_scaler_path is not None:
        in_scaler = joblib.load(to_absolute_path(config.data.in_scaler_path))
        if isinstance(in_scaler, SKMinMaxScaler):
            in_scaler = MinMaxScaler(
                in_scaler.min_,
                in_scaler.scale_,
                in_scaler.data_min_,
                in_scaler.data_max_,
            )
    else:
        in_scaler = None
    if "out_scaler_path" in config.data and config.data.out_scaler_path is not None:
        out_scaler = joblib.load(to_absolute_path(config.data.out_scaler_path))
        out_scaler = StandardScaler(
            out_scaler.mean_, out_scaler.var_, out_scaler.scale_
        )
    else:
        out_scaler = None

    return (
        (netG_A2B, netG_B2A, optG, schedulerG),
        (netD_A, netD_B, optD, schedulerD),
        grad_scaler,
        data_loaders,
        writer,
        logger,
        in_scaler,
        out_scaler,
    )


def save_configs(config):
    out_dir = Path(to_absolute_path(config.train.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.yaml", "w") as f:
        OmegaConf.save(config.model, f)
    with open(out_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)


def check_resf0_config(logger, model, config, in_scaler, out_scaler):
    logger.info("Checking model configs for residual F0 prediction")
    if in_scaler is None or out_scaler is None:
        raise ValueError("in_scaler and out_scaler must be specified")

    if isinstance(model, nn.DataParallel):
        model = model.module

    in_lf0_idx = config.data.in_lf0_idx
    in_rest_idx = config.data.in_rest_idx
    out_lf0_idx = config.data.out_lf0_idx
    if in_lf0_idx is None or in_rest_idx is None or out_lf0_idx is None:
        raise ValueError("in_lf0_idx, in_rest_idx and out_lf0_idx must be specified")

    logger.info("in_lf0_idx: %s", in_lf0_idx)
    logger.info("in_rest_idx: %s", in_rest_idx)
    logger.info("out_lf0_idx: %s", out_lf0_idx)

    ok = True
    if hasattr(model, "in_lf0_idx"):
        if model.in_lf0_idx != in_lf0_idx:
            logger.warning(
                "in_lf0_idx in model and data config must be same",
                model.in_lf0_idx,
                in_lf0_idx,
            )
            ok = False
    if hasattr(model, "out_lf0_idx"):
        if model.out_lf0_idx != out_lf0_idx:
            logger.warning(
                "out_lf0_idx in model and data config must be same",
                model.out_lf0_idx,
                out_lf0_idx,
            )
            ok = False

    if hasattr(model, "in_lf0_min") and hasattr(model, "in_lf0_max"):
        # Inject values from the input scaler
        if model.in_lf0_min is None or model.in_lf0_max is None:
            model.in_lf0_min = in_scaler.data_min_[in_lf0_idx]
            model.in_lf0_max = in_scaler.data_max_[in_lf0_idx]

        logger.info("in_lf0_min: %s", model.in_lf0_min)
        logger.info("in_lf0_max: %s", model.in_lf0_max)
        if not np.allclose(model.in_lf0_min, in_scaler.data_min_[model.in_lf0_idx]):
            logger.warning(
                f"in_lf0_min is set to {model.in_lf0_min}, "
                f"but should be {in_scaler.data_min_[model.in_lf0_idx]}"
            )
            ok = False
        if not np.allclose(model.in_lf0_max, in_scaler.data_max_[model.in_lf0_idx]):
            logger.warning(
                f"in_lf0_max is set to {model.in_lf0_max}, "
                f"but should be {in_scaler.data_max_[model.in_lf0_idx]}"
            )
            ok = False

    if hasattr(model, "out_lf0_mean") and hasattr(model, "out_lf0_scale"):
        # Inject values from the output scaler
        if model.out_lf0_mean is None or model.out_lf0_scale is None:
            model.out_lf0_mean = out_scaler.mean_[out_lf0_idx]
            model.out_lf0_scale = out_scaler.scale_[out_lf0_idx]

        logger.info("model.out_lf0_mean: %s", model.out_lf0_mean)
        logger.info("model.out_lf0_scale: %s", model.out_lf0_scale)
        if not np.allclose(model.out_lf0_mean, out_scaler.mean_[model.out_lf0_idx]):
            logger.warning(
                f"out_lf0_mean is set to {model.out_lf0_mean}, "
                f"but should be {out_scaler.mean_[model.out_lf0_idx]}"
            )
            ok = False
        if not np.allclose(model.out_lf0_scale, out_scaler.scale_[model.out_lf0_idx]):
            logger.warning(
                f"out_lf0_scale is set to {model.out_lf0_scale}, "
                f"but should be {out_scaler.scale_[model.out_lf0_idx]}"
            )
            ok = False

    if not ok:
        if (
            model.in_lf0_idx == in_lf0_idx
            and hasattr(model, "in_lf0_min")
            and hasattr(model, "out_lf0_mean")
        ):
            logger.info(
                f"""
If you are 100% sure that you set model.in_lf0_idx and model.out_lf0_idx correctly,
Please consider the following parameters in your model config:

    in_lf0_idx: {model.in_lf0_idx}
    out_lf0_idx: {model.out_lf0_idx}
    in_lf0_min: {in_scaler.data_min_[model.in_lf0_idx]}
    in_lf0_max: {in_scaler.data_max_[model.in_lf0_idx]}
    out_lf0_mean: {out_scaler.mean_[model.out_lf0_idx]}
    out_lf0_scale: {out_scaler.scale_[model.out_lf0_idx]}
"""
            )
        raise ValueError("The model config has wrong configurations.")

    # Overwrite the parameters to the config
    for key in ["in_lf0_min", "in_lf0_max", "out_lf0_mean", "out_lf0_scale"]:
        if hasattr(model, key):
            config.model.netG[key] = float(getattr(model, key))


def note_segments(lf0_score_denorm):
    """Compute note segments (start and end indices) from log-F0

    Note that unvoiced frames must be set to 0 in advance.

    Args:
        lf0_score_denorm (Tensor): (B, T)

    Returns:
        list: list of note (start, end) indices
    """
    segments = []
    for s, e in nonzero_segments(lf0_score_denorm):
        out = torch.sign(torch.abs(torch.diff(lf0_score_denorm[s : e + 1])))
        transitions = torch.where(out > 0)[0]
        note_start, note_end = s, -1
        for pos in transitions:
            note_end = int(s + pos)
            segments.append((note_start, note_end))
            note_start = note_end + 1

        # Handle last note
        while (
            note_start < len(lf0_score_denorm) - 1 and lf0_score_denorm[note_start] <= 0
        ):
            note_start += 1
        note_end = note_start + 1
        while note_end < len(lf0_score_denorm) - 1 and lf0_score_denorm[note_end] > 0:
            note_end += 1

        if note_end != note_start + 1:
            segments.append((note_start, note_end))

    return segments


def compute_pitch_regularization_weight(segments, N, decay_size=25, max_w=0.5):
    """Compute pitch regularization weight given note segments

    Args:
        segments (list): list of note (start, end) indices
        N (int): number of frames
        decay_size (int): size of the decay window
        max_w (float): maximum weight

    Returns:
        Tensor: weights of shape (N,)
    """
    w = torch.zeros(N)

    for s, e in segments:
        L = e - s
        w[s:e] = max_w
        if L > decay_size * 2:
            w[s : s + decay_size] *= torch.arange(decay_size) / decay_size
            w[e - decay_size : e] *= torch.arange(decay_size - 1, -1, -1) / decay_size

    return w


def compute_batch_pitch_regularization_weight(lf0_score_denorm, decay_size):
    """Batch version of computing pitch regularization weight

    Args:
        lf0_score_denorm (Tensor): (B, T)

    Returns:
        Tensor: weights of shape (B, N, 1)
    """
    B, T = lf0_score_denorm.shape
    w = torch.zeros_like(lf0_score_denorm)
    for idx in range(len(lf0_score_denorm)):
        segments = note_segments(lf0_score_denorm[idx])
        w[idx, :] = compute_pitch_regularization_weight(
            segments, N=T, decay_size=decay_size
        ).to(w.device)

    return w.unsqueeze(-1)


@torch.no_grad()
def compute_distortions(pred_out_feats, out_feats, lengths, out_scaler, model_config):
    """Compute distortion measures between predicted and ground-truth acoustic features


    Args:
        pred_out_feats (nn.Tensor): predicted acoustic features
        out_feats (nn.Tensor): ground-truth acoustic features
        lengths (nn.Tensor): lengths of the sequences
        out_scaler (nn.Module): scaler to denormalize features
        model_config (dict): model configuration

    Returns:
        dict: a dict that includes MCD for mgc/bap, V/UV error and F0 RMSE
    """
    out_feats = out_scaler.inverse_transform(out_feats)
    pred_out_feats = out_scaler.inverse_transform(pred_out_feats)
    out_streams = get_static_features(
        out_feats,
        model_config.num_windows,
        model_config.stream_sizes,
        model_config.has_dynamic_features,
    )
    pred_out_streams = get_static_features(
        pred_out_feats,
        model_config.num_windows,
        model_config.stream_sizes,
        model_config.has_dynamic_features,
    )

    assert len(out_streams) >= 4
    mgc, lf0, vuv, bap = out_streams[0], out_streams[1], out_streams[2], out_streams[3]
    pred_mgc, pred_lf0, pred_vuv, pred_bap = (
        pred_out_streams[0],
        pred_out_streams[1],
        pred_out_streams[2],
        pred_out_streams[3],
    )

    # binarize vuv
    vuv, pred_vuv = (vuv > 0.5).float(), (pred_vuv > 0.5).float()

    dist = {
        "ObjEval_MGC_MCD": metrics.melcd(
            mgc[:, :, 1:], pred_mgc[:, :, 1:], lengths=lengths
        ),
        "ObjEval_BAP_MCD": metrics.melcd(bap, pred_bap, lengths=lengths) / 10.0,
        "ObjEval_VUV_ERR": metrics.vuv_error(vuv, pred_vuv, lengths=lengths),
    }

    try:
        f0_mse = metrics.lf0_mean_squared_error(
            lf0, vuv, pred_lf0, pred_vuv, lengths=lengths, linear_domain=True
        )
        dist["ObjEval_F0_RMSE"] = np.sqrt(f0_mse)
    except ZeroDivisionError:
        pass

    return dist


@torch.no_grad()
def eval_spss_model(
    phase,
    step,
    netG,
    in_feats,
    out_feats,
    lengths,
    model_config,
    out_scaler,
    writer,
    sr,
    lf0_score_denorm=None,
    trajectory_smoothing=True,
    trajectory_smoothing_cutoff=50,
    trajectory_smoothing_cutoff_f0=20,
):
    # make sure to be in eval mode
    netG.eval()
    is_autoregressive = (
        netG.module.is_autoregressive()
        if isinstance(netG, nn.DataParallel)
        else netG.is_autoregressive()
    )
    prediction_type = (
        netG.module.prediction_type()
        if isinstance(netG, nn.DataParallel)
        else netG.prediction_type()
    )
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
        out_feats_denorm_ = out_scaler.inverse_transform(
            out_feats[utt_idx, : lengths[utt_idx]].unsqueeze(0)
        )
        mgc, lf0, vuv, bap = get_static_features(
            out_feats_denorm_,
            model_config.num_windows,
            model_config.stream_sizes,
            model_config.has_dynamic_features,
        )[:4]
        mgc = mgc.squeeze(0).cpu().numpy()
        lf0 = lf0.squeeze(0).cpu().numpy()
        vuv = vuv.squeeze(0).cpu().numpy()
        bap = bap.squeeze(0).cpu().numpy()
        if lf0_score_denorm is not None:
            lf0_score_denorm_ = (
                lf0_score_denorm[utt_idx, : lengths[utt_idx]].cpu().numpy().reshape(-1)
            )
        else:
            lf0_score_denorm_ = None

        f0, spectrogram, aperiodicity = gen_world_params(mgc, lf0, vuv, bap, sr)
        wav = pyworld.synthesize(f0, spectrogram, aperiodicity, sr, 5)
        group = f"{phase}_utt{np.abs(utt_idx)}_reference"
        wav = wav / np.abs(wav).max() if np.max(wav) > 1.0 else wav
        writer.add_audio(group, wav, step, sr)

        # Run forward
        if is_autoregressive:
            outs = netG(
                in_feats[utt_idx, : lengths[utt_idx]].unsqueeze(0),
                [lengths[utt_idx]],
                out_feats[utt_idx, : lengths[utt_idx]].unsqueeze(0),
            )
        else:
            outs = netG(
                in_feats[utt_idx, : lengths[utt_idx]].unsqueeze(0), [lengths[utt_idx]]
            )

        # ResF0 case
        if isinstance(outs, tuple) and len(outs) == 2:
            outs, _ = outs

        if prediction_type == PredictionType.PROBABILISTIC:
            pi, sigma, mu = outs
            pred_out_feats = mdn_get_most_probable_sigma_and_mu(pi, sigma, mu)[1]
        else:
            pred_out_feats = outs
        # NOTE: multiple outputs
        if isinstance(pred_out_feats, list):
            pred_out_feats = pred_out_feats[-1]
        if isinstance(pred_out_feats, tuple):
            pred_out_feats = pred_out_feats[0]

        if not isinstance(pred_out_feats, list):
            pred_out_feats = [pred_out_feats]

        # Run inference
        if prediction_type == PredictionType.PROBABILISTIC:
            if isinstance(netG, nn.DataParallel):
                inference_out_feats, _ = netG.module.inference(
                    in_feats[utt_idx, : lengths[utt_idx]].unsqueeze(0),
                    [lengths[utt_idx]],
                )
            else:
                inference_out_feats, _ = netG.inference(
                    in_feats[utt_idx, : lengths[utt_idx]].unsqueeze(0),
                    [lengths[utt_idx]],
                )
        else:
            if isinstance(netG, nn.DataParallel):
                inference_out_feats = netG.module.inference(
                    in_feats[utt_idx, : lengths[utt_idx]].unsqueeze(0),
                    [lengths[utt_idx]],
                )
            else:
                inference_out_feats = netG.inference(
                    in_feats[utt_idx, : lengths[utt_idx]].unsqueeze(0),
                    [lengths[utt_idx]],
                )
        pred_out_feats.append(inference_out_feats)

        assert len(pred_out_feats) == 2
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

            # Remove high-frequency components of mgc/bap
            # NOTE: It seems to be effective to suppress artifacts of GAN-based post-filtering
            if trajectory_smoothing:
                modfs = int(1 / 0.005)
                pred_lf0[:, 0] = lowpass_filter(
                    pred_lf0[:, 0], modfs, cutoff=trajectory_smoothing_cutoff_f0
                )
                for d in range(pred_mgc.shape[1]):
                    pred_mgc[:, d] = lowpass_filter(
                        pred_mgc[:, d], modfs, cutoff=trajectory_smoothing_cutoff
                    )
                for d in range(pred_bap.shape[1]):
                    pred_bap[:, d] = lowpass_filter(
                        pred_bap[:, d], modfs, cutoff=trajectory_smoothing_cutoff
                    )

            # Generated sample
            f0, spectrogram, aperiodicity = gen_world_params(
                pred_mgc, pred_lf0, pred_vuv, pred_bap, sr
            )
            wav = pyworld.synthesize(f0, spectrogram, aperiodicity, sr, 5)
            wav = wav / np.abs(wav).max() if np.max(wav) > 1.0 else wav
            if idx == 1:
                group = f"{phase}_utt{np.abs(utt_idx)}_inference"
            else:
                group = f"{phase}_utt{np.abs(utt_idx)}_forward"
            writer.add_audio(group, wav, step, sr)
            plot_spsvs_params(
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
                lf0_score=lf0_score_denorm_,
                group=group,
                sr=sr,
            )


@torch.no_grad()
def plot_spsvs_params(
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
    lf0_score,
    group,
    sr,
):
    """Plot acoustic parameters of parametric SVS

    Args:
        step (int): step of the current iteration
        writer (tensorboard.SummaryWriter): tensorboard writer
        mgc (np.ndarray): mgc
        lf0 (np.ndarray): lf0
        vuv (np.ndarray): vuv
        bap (np.ndarray): bap
        pred_mgc (np.ndarray): predicted mgc
        pred_lf0 (np.ndarray): predicted lf0
        pred_vuv (np.ndarray): predicted vuv
        pred_bap (np.ndarray): predicted bap
        f0_score (np.ndarray): lf0 score
        group (str): group name
        sr (int): sampling rate
    """
    fftlen = pyworld.get_cheaptrick_fft_size(sr)
    alpha = pysptk.util.mcepalpha(sr)
    hop_length = int(sr * 0.005)

    # Log-F0
    if lf0_score is not None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        timeaxis = np.arange(len(lf0)) * 0.005
        ax.plot(timeaxis, lf0, linewidth=1.5, color="tab:blue", label="Target log-F0")
        ax.plot(
            timeaxis,
            pred_lf0,
            linewidth=1.5,
            color="tab:orange",
            label="Predicted log-F0",
        )
        ax.plot(
            timeaxis,
            lf0_score,
            "--",
            color="gray",
            linewidth=1.3,
            label="Note log-F0",
        )
        ax.set_xlabel("Time [sec]")
        ax.set_ylabel("Log-frequency [Hz]")
        ax.set_xlim(timeaxis[0], timeaxis[-1])
        ax.set_ylim(
            min(min(lf0_score[lf0_score > 0]), min(lf0), min(pred_lf0)) - 0.1,
            max(max(lf0_score), max(lf0), max(pred_lf0)) + 0.1,
        )
        plt.legend(loc="upper right", borderaxespad=0, ncol=3)
        plt.tight_layout()
        writer.add_figure(f"{group}/ContinuousLogF0", fig, step)
        plt.close()

        f0_score = lf0_score.copy()
        note_indices = f0_score > 0
        f0_score[note_indices] = np.exp(lf0_score[note_indices])

        # F0
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        timeaxis = np.arange(len(lf0)) * 0.005
        f0 = np.exp(lf0)
        f0[vuv < 0.5] = 0
        pred_f0 = np.exp(pred_lf0)
        pred_f0[pred_vuv < 0.5] = 0
        ax.plot(
            timeaxis,
            f0,
            linewidth=1.5,
            color="tab:blue",
            label="Target F0",
        )
        ax.plot(
            timeaxis,
            pred_f0,
            linewidth=1.5,
            color="tab:orange",
            label="Predicted F0",
        )
        ax.plot(timeaxis, f0_score, "--", linewidth=1.3, color="gray", label="Note F0")
        ax.set_xlabel("Time [sec]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlim(timeaxis[0], timeaxis[-1])
        ax.set_ylim(
            min(min(f0_score[f0_score > 0]), min(np.exp(lf0)), min(np.exp(pred_lf0)))
            - 10,
            max(max(f0_score), max(np.exp(lf0)), max(np.exp(pred_lf0))) + 10,
        )
        plt.legend(loc="upper right", borderaxespad=0, ncol=3)
        plt.tight_layout()
        writer.add_figure(f"{group}/F0", fig, step)
        plt.close()

    # V/UV
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    timeaxis = np.arange(len(lf0)) * 0.005
    ax.plot(timeaxis, vuv, linewidth=2, label="Target V/UV")
    ax.plot(timeaxis, pred_vuv, "--", linewidth=2, label="Predicted V/UV")
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("V/UV")
    ax.set_xlim(timeaxis[0], timeaxis[-1])
    plt.legend(loc="upper right", borderaxespad=0, ncol=2)
    plt.tight_layout()
    writer.add_figure(f"{group}/VUV", fig, step)
    plt.close()

    # Spectrogram
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].set_title("Reference spectrogram")
    ax[1].set_title("Predicted spectrogram")
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
    pred_spectrogram = pysptk.mc2sp(
        np.ascontiguousarray(pred_mgc), fftlen=fftlen, alpha=alpha
    ).T
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
        a.set_ylim(0, sr // 2)
    plt.tight_layout()
    writer.add_figure(f"{group}/Spectrogram", fig, step)
    plt.close()

    # Aperiodicity
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].set_title("Reference aperiodicity")
    ax[1].set_title("Predicted aperiodicity")
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
        np.ascontiguousarray(pred_bap).astype(np.float64), sr, fftlen
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
        a.set_ylim(0, sr // 2)
    plt.tight_layout()
    writer.add_figure(f"{group}/Aperiodicity", fig, step)
    plt.close()

    # GV for mgc
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(np.var(mgc, axis=0), "--", linewidth=2, label="Natural: global variances")
    ax.plot(np.var(pred_mgc, axis=0), linewidth=2, label="Generated: global variances")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("Dimension of mgc")
    min_ = min(np.var(mgc, axis=0).min(), np.var(pred_mgc, axis=0).min(), 1e-4)
    ax.set_ylim(min_)
    plt.tight_layout()
    writer.add_figure(f"{group}/GV_mgc", fig, step)
    plt.close()

    # GV for bap
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(np.var(bap, axis=0), "--", linewidth=2, label="Natural: global variances")
    ax.plot(np.var(pred_bap, axis=0), linewidth=2, label="Generated: global variances")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("Dimension of bap")
    min_ = min(np.var(bap, axis=0).min(), np.var(pred_bap, axis=0).min(), 10)
    ax.set_ylim(min_)
    plt.tight_layout()
    writer.add_figure(f"{group}/GV_bap", fig, step)
    plt.close()
