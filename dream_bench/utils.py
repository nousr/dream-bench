import os
from urllib.request import urlretrieve
from click import secho
from importlib import import_module
from torch import cuda, load as torch_load

CACHE_FOLDER = "~/.cache/draw_bench"
DEFAULT_PRIOR_STATE_URL = "https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/prior/latest.pth"
DEFAULT_PRIOR_CONFIG_URL = "https://huggingface.co/laion/DALLE2-PyTorch/raw/main/prior/prior_config.json"

def exists(x):
    return x is not None

def filename_from_path(path):
    return path.split("/")[-1]

def import_or_print_error(pkg_name, err_str=None, **kwargs):
    try:
        return import_module(pkg_name)
    except ModuleNotFoundError as _:
        if exists(err_str):
            secho(err_str, **kwargs)
        exit()


def _load_open_clip(clip_model, use_jit=True, device="cuda"):
    open_clip = import_or_print_error(
        "open_clip",
        err_str="You have requested an open-clip model but do not have the library installed.",
        fg="red",
    )

    pretrained = dict(open_clip.list_pretrained())
    checkpoint = pretrained[clip_model]
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained=checkpoint, device=device, jit=use_jit
    )

    return model, preprocess


def load_clip(clip_model, use_jit=True):
    device = "cuda" if cuda.is_available() else "cpu"

    if clip_model.startswith("open_clip:"):
        clip_model = clip_model[len("open_clip:") :]
        return _load_open_clip(clip_model, use_jit, device)

    else:
        clip = import_or_print_error(
            "clip",
            err_str="You have requested an openai-clip model but do not have the library installed.",
            fg="red",
        )

        model, preprocess = clip.load(clip_model, device=device, jit=use_jit)

    return model, preprocess


def _download_prior(state_url: str, config_url:str):
    """
    Download a diffusion prior and configuration file from a url
    """

    # create cache folders

    cache_folder = os.path.join(CACHE_FOLDER, "prior")
    prior_file = os.path.join(cache_folder, filename_from_path(state_url))
    config_file = os.path.join(cache_folder, filename_from_path(config_url))

    # download checkpoint & model to cache

    urlretrieve(config_url, config_file)
    urlretrieve(state_url, prior_file)

    # return files

    return prior_file, config_file

def load_prior(prior_config_path, checkpoint_path: str, default=True):
    device = "cuda" if cuda.is_available() else "cpu"

    dalle2_train_config = import_or_print_error(
        "dalle2_pytorch.train_configs",
        err_str="dalle2_pytorch is a required package to load diffusion_prior models.",
        fg="red",
    )

    # download from url if default

    # TODO

    # load configuration from path

    prior_config = dalle2_train_config.TrainDiffusionPriorConfig.from_json_path(
        prior_config_path
    )
    prior_config = prior_config["prior"]

    # create model from config

    diffusion_prior = prior_config.create()
    state_dict = torch_load(checkpoint_path, map_location=device)
    diffusion_prior.load_state_dict(state_dict)
    diffusion_prior.eval()
    diffusion_prior.to(device)

    if device == "cpu":
        diffusion_prior.float()

    return diffusion_prior