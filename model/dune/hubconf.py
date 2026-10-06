import torch
from pathlib import Path

from model.dune import load_dune_encoder_from_checkpoint, load_dune_from_checkpoint


URL_DICT = {
    "vitbase_14_448_paper": "https://download.europe.naverlabs.com/dune/dune_vitbase14_448_paper.pth",
    "vitbase_14_448": "https://download.europe.naverlabs.com/dune/dune_vitbase14_448.pth",
    "vitbase_14_336": "https://download.europe.naverlabs.com/dune/dune_vitbase14_336.pth",
    "vitsmall_14_448": "https://download.europe.naverlabs.com/dune/dune_vitsmall14_448.pth",
    "vitsmall_14_336": "https://download.europe.naverlabs.com/dune/dune_vitsmall14_336.pth",
}


def _load_dune_model_from_url(model_name, encoder_only=False):
    if model_name not in URL_DICT:
        raise ValueError("Model name '{}' is not recognized.".format(model_name))

    cache_dir = Path(torch.hub.get_dir())
    cache_dir = cache_dir / "checkpoints" / "dune"
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = URL_DICT[model_name]
    ckpt_fname = cache_dir / Path(url).name
    if not ckpt_fname.exists():
        torch.hub.download_url_to_file(url, ckpt_fname)

    if encoder_only:
        return load_dune_encoder_from_checkpoint(ckpt_fname)[0]
    else:
        return load_dune_from_checkpoint(ckpt_fname)[0]


def dune_vitbase_14_448_paper_encoder():
    return _load_dune_model_from_url("vitbase_14_448_paper", encoder_only=True)


def dune_vitbase_14_448_paper():
    return _load_dune_model_from_url("vitbase_14_448_paper", encoder_only=False)


def dune_vitbase_14_448_encoder():
    return _load_dune_model_from_url("vitbase_14_448", encoder_only=True)


def dune_vitbase_14_448():
    return _load_dune_model_from_url("vitbase_14_448", encoder_only=False)


def dune_vitbase_14_336_encoder():
    return _load_dune_model_from_url("vitbase_14_336", encoder_only=True)


def dune_vitbase_14_336():
    return _load_dune_model_from_url("vitbase_14_336", encoder_only=False)


def dune_vitsmall_14_448_encoder():
    return _load_dune_model_from_url("vitsmall_14_448", encoder_only=True)


def dune_vitsmall_14_448():
    return _load_dune_model_from_url("vitsmall_14_448", encoder_only=False)
