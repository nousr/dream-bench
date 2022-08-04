import os
from dream_bench.helpers import (
    import_or_print_error,
    exists,
    filename_from_path,
    is_url,
)
from urllib.request import urlretrieve
from torch import cuda, load as torch_load, nn
from os.path import expanduser

CACHE_FOLDER = os.path.join(os.path.expanduser("~"), ".cache", "dream_bench")
DEFAULT_PRIOR_STATE_URL = (
    "https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/prior/latest.pth"
)
DEFAULT_PRIOR_CONFIG_URL = (
    "https://huggingface.co/laion/DALLE2-PyTorch/raw/main/prior/prior_config.json"
)


def _load_open_clip(clip_model, use_jit=True, device="cuda"):
    "Load a clip model from the open-clip library"

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
    "Load a clip model from openai or open-clip"

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


def _download_prior(state_url: str, config_url: str):
    "Download a diffusion prior and configuration file from a url"

    # create cache folders

    cache_folder = os.path.join(CACHE_FOLDER, "prior")
    os.makedirs(cache_folder, exist_ok=True)

    prior_file = os.path.join(cache_folder, filename_from_path(state_url))
    config_file = os.path.join(cache_folder, filename_from_path(config_url))

    # download checkpoint & model to cache if they don't exist

    if not os.path.isfile(prior_file):
        prior_file, _ = urlretrieve(state_url, prior_file)

    if not os.path.isfile(config_file):
        config_file, _ = urlretrieve(config_url, config_file)

    # return files

    return prior_file, config_file


def load_prior(checkpoint_path: str, config_path: str):
    "Load a dalle2-pytorch diffusion prior model"

    device = "cuda" if cuda.is_available() else "cpu"

    dalle2_train_config = import_or_print_error(
        "dalle2_pytorch.train_configs",
        err_str="dalle2_pytorch is a required package to load diffusion_prior models.",
        fg="red",
    )

    # if both params are none, then replace with defaults

    if not exists(checkpoint_path) and not exists(config_path):
        checkpoint_path = DEFAULT_PRIOR_STATE_URL
        config_path = DEFAULT_PRIOR_CONFIG_URL

    # check if the paths are actually urls to download from

    if is_url(checkpoint_path) and is_url(config_path):
        checkpoint_path, config_path = _download_prior(checkpoint_path, config_path)

    # check if the path exists

    assert os.path.exists(checkpoint_path) and os.path.exists(
        config_path
    ), "Files must exist and be visible if loading from a local path."

    # load configuration from path

    prior_config = dalle2_train_config.TrainDiffusionPriorConfig.from_json_path(
        config_path
    )
    prior_config = prior_config.prior

    # create model from config

    diffusion_prior = prior_config.create()
    state_dict = torch_load(checkpoint_path, map_location=device)
    diffusion_prior.load_state_dict(state_dict)
    diffusion_prior.eval()
    diffusion_prior.to(device)

    if device == "cpu":
        diffusion_prior.float()

    return diffusion_prior


def get_aesthetic_model(clip_model="ViT-L/14"):
    """load the aethetic model"""
    if clip_model == "ViT-L/14":
        model_file = "vit_l_14"
    elif clip_model == "ViT-B/32":
        model_file = "vit_b_32"
    else:
        raise NotImplementedError(
            "No aesthetic model has been trained on that architecture."
        )

    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_" + model_file + "_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"
            + clip_model
            + "_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "ViT-L/14":
        m = nn.Linear(768, 1)
    elif clip_model == "ViT-B/32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch_load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m
