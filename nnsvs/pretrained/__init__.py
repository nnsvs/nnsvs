import os
import shutil
import tarfile
from os.path import join
from pathlib import Path
from urllib.request import urlretrieve

from nnsvs.util import dynamic_import
from tqdm.auto import tqdm

DEFAULT_CACHE_DIR = join(os.path.expanduser("~"), ".cache", "nnsvs")
CACHE_DIR = os.environ.get("NNSVS_CACHE_DIR", DEFAULT_CACHE_DIR)

model_registry = {
    "r9y9/yoko_latest": {
        "url": "https://www.dropbox.com/s/k8mya65yt52m0ps/yoko_latest.tar.gz?dl=1",
        "_target_": "nnsvs.svs:SPSVS",
    },
    "r9y9/20220322_yoko_timelag_mdn_duration_mdn_acoustic_resf0conv": {
        "url": "https://www.dropbox.com/s/olsfyqol9ryk5kx/"
        "20220322_yoko_timelag_mdn_duration_mdn_acoustic_resf0conv.tar.gz?dl=1",
        "_target_": "nnsvs.svs:SPSVS",
    },
}


def create_svs_engine(name, *args, **kwargs):
    """Create SVS engine from pretrained models.

    Args:
        name (str): Pre-trained model name
        args (list): Additional args for instantiation
        kwargs (dict): Additional kwargs for instantiation

    Returns:
        object: instance of SVS engine
    """
    if name not in model_registry:
        s = ""
        for model_id in get_available_model_ids():
            s += f"'{model_id}'\n"
        raise ValueError(
            f"""
Pretrained model '{name}' does not exist!

Available models:
{s[:-1]}"""
        )

    # download if not exists
    model_dir = retrieve_pretrained_model(name)

    # create an instance
    return dynamic_import(model_registry[name]["_target_"])(model_dir, *args, **kwargs)


def get_available_model_ids():
    """Get available pretrained model names.

    Returns:
        list: List of available pretrained model names.
    """
    return list(model_registry.keys())


# https://github.com/tqdm/tqdm#hooks-and-callbacks
class _TqdmUpTo(tqdm):  # type: ignore
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)


def is_pretrained_model_ready(name):
    out_dir = Path(CACHE_DIR) / name
    if out_dir.exists() and len(list(out_dir.glob("*.pth"))) == 0:
        return False
    return out_dir.exists()


def retrieve_pretrained_model(name):
    """Retrieve pretrained model from local cache or download from GitHub.

    Args:
        name (str): Name of pretrained model.

    Returns:
        str: Path to the pretrained model.

    Raises:
        ValueError: If the pretrained model is not found.
    """
    global model_registry
    if name not in model_registry:
        s = ""
        for model_id in get_available_model_ids():
            s += f"'{model_id}'\n"
        raise ValueError(
            f"""
Pretrained model '{name}' does not exist!

Available models:
{s[:-1]}"""
        )

    url = model_registry[name]["url"]
    # NOTE: assuming that filename and extracted is the same
    out_dir = Path(CACHE_DIR) / name
    model_dir = out_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(CACHE_DIR) / f"{name}.tar.gz"

    # re-download models
    if out_dir.exists() and len(list(out_dir.glob("*.pth"))) == 0:
        shutil.rmtree(out_dir)

    if not out_dir.exists():
        print('Downloading: "{}"'.format(url))
        with _TqdmUpTo(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=f"{name}.tar.gz",
        ) as t:  # all optional kwargs
            urlretrieve(url, filename, reporthook=t.update_to)
            t.total = t.n
        with tarfile.open(filename, mode="r|gz") as f:
            f.extractall(path=model_dir)
        os.remove(filename)

    return out_dir
