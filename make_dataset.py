#! python

import os
import json
import click
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.nn import Module
from torch.cuda import is_available
from webdataset import ShardWriter
from dream_bench.load_models import load_clip, load_prior, filename_from_path
from dream_bench.tokenizer import tokenizer
from dream_bench.helpers import print_ribbon

DEVICE = "cuda" if is_available() else "cpu"
TOKENIZER_CONTEXT_LENGTH = 77
TOKENIZER_TRUNCATE_TEXT = True


def load_prompts(filepath, prompt_repeat):
    """
    Load the prompts from a file
    """

    if filepath.endswith(".json"):
        with open(filepath, "r") as f:
            return np.array(json.load(f)).repeat(prompt_repeat)
    elif filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
        try:
            return np.array(df["caption"].values).repeat(prompt_repeat)
        except KeyError as e:
            print(e)
            print("There is no `caption` key in your dataframe")
    else:
        print("This filetype is not supported")
        exit(1)


def load_images(filepath, resize=None):
    images = []

    for file in os.listdir(filepath):
        image = Image.open(os.path.join(filepath, file))

        if resize is not None:
            image = image.resize((resize, resize), Image.Resampling.LANCZOS)

        images.append(np.array(image).transpose(2, 0, 1))

    return images


def txt2embeddings(model: Module, architecture: str, text: torch.Tensor, batch_size: int):
    """
    Predict embeddings for captions using a given batch size.

    Returns:
        - All predicted embeddings batched along the first dimension and converted to numpy format.
    """
    click.secho(print_ribbon(f"Creating Embeddings With: {architecture}"), fg="green")

    embeds = list()

    for i in range(0, len(text), batch_size):
        model_input = text[i : i + batch_size]

        if architecture == "prior":
            embeds.append(model.sample(model_input).detach().cpu().numpy())
        elif architecture == "clip":
            embeds.append(model.encode_text(model_input).detach().cpu().numpy())
        else:
            click.secho(f"The {architecture} architecture is not yet supported.", fg="red")
            exit(1)

    return np.concatenate(embeds, axis=0)


def wds_create(
    prompts: list,
    images: list,
    batch_size: int,
    output_folder: str,
    filename: str,
    clip_model: Module,
    prior_model: Module,
    predict_image: bool,
    embed_text: bool,
    tokenize_text: bool,
):
    """
    Create a webdataset for the prompt list.

    # TODO: docstring
    """
    filename = filename.split(".")[0]

    # precompute embeddings for all requested formats
    tokenized_text = tokenizer.tokenize(
        prompts,
        context_length=TOKENIZER_CONTEXT_LENGTH,
        truncate_text=TOKENIZER_TRUNCATE_TEXT,
    ).to(DEVICE)
    clip_text_embeddings = (
        txt2embeddings(
            model=clip_model,
            text=tokenized_text,
            batch_size=batch_size,
            architecture="clip",
        )
        if embed_text
        else None
    )
    prior_image_embeddings = (
        txt2embeddings(
            model=prior_model,
            text=tokenized_text,
            batch_size=batch_size,
            architecture="prior",
        )
        if predict_image
        else None
    )

    with ShardWriter(pattern=f"{output_folder}/{filename}-%04d.tar") as sink:
        for idx, caption in enumerate(prompts):
            tar_keys = {"__key__": "%04d" % idx, "caption.txt": caption}

            if predict_image:
                tar_keys["prior_image_embedding.npy"] = prior_image_embeddings[idx].detach().cpu().numpy()
            if embed_text:
                tar_keys["clip_text_embedding.npy"] = clip_text_embeddings[idx].detach().cpu().numpy()
            if tokenize_text:
                tar_keys["tokenized_text.npy"] = tokenized_text[idx].detach().cpu().numpy()
            if images is not None:
                tar_keys["real_image.npy"] = images[idx]

            sink.write(tar_keys)


@click.command()
@click.option(
    "--prompt-list",
    type=str,
    help="A json or parquet file containing a list of prompts.",
    required=True,
)
@click.option(
    "--real-images",
    type=str,
    help="A path to a folder containing real images",
    default=None,
)
@click.option(
    "--resize",
    type=int,
    help="What size to reisze to when loading real images.",
)
@click.option(
    "--output-folder",
    type=str,
    help="parent folder of the dataset to be created",
    required=True,
)
@click.option(
    "--batch-size",
    default=512,
    type=int,
    help="Number of prompts to process in parallel.",
)
@click.option(
    "--prompt-repeat",
    default=1,
    type=int,
    help="Number of times to run inference for a prompt.",
)
@click.option(
    "--embed-text",
    is_flag=True,
    help="Wether to use clip to embed the prompts as text embeddings",
)
@click.option(
    "--tokenize-text",
    is_flag=True,
    help="Wether to save the tokenized the text.",
)
@click.option(
    "--predict-image",
    is_flag=True,
    help="Wether to use a diffusion prior to create an image embedding from each prompt.",
)
@click.option(
    "--clip-model",
    default=None,
    type=str,
    help="Clip model to use for embedding images.",
)
@click.option(
    "--prior-checkpoint",
    default=None,
    type=str,
    help="A local path or web url to a prior config file.",
)
@click.option(
    "--prior-config",
    default=None,
    type=str,
    help="A local path or web url to a prior checkpoint.",
)
@click.option(
    "--override-folder",
    is_flag=True,
    help="wether to overrite the output folder if it exists",
)
def main(
    prompt_list,
    real_images,
    resize,
    output_folder,
    batch_size,
    prompt_repeat,
    embed_text,
    tokenize_text,
    predict_image,
    clip_model,
    prior_checkpoint,
    prior_config,
    override_folder,
):
    filename = filename_from_path(prompt_list)
    options = {
        "tokenized_text": tokenize_text,
        "clip_text_embed": embed_text,
        "prior_image_embed": predict_image,
    }

    # TODO: asserts & ensure that clip model & prior model align

    click.secho(
        f"Generating benchmark dataset for: {filename}",
        fg="bright_cyan",
        underline=True,
    )

    for k, v in options.items():
        click.secho(f"\t* {k} : {v}", fg="green" if v else "red")

    # check that the output folder can be created

    try:
        os.makedirs(output_folder, exist_ok=override_folder)
        click.secho(f"Dataset will be created at: {output_folder}", fg="magenta")
    except OSError:
        click.secho(
            f"WARNING: The folder you specified already exists. This is a safety measure to avoid destroying existing work. If you wish to override it, please re-run this script with the --override-folder flag.",
            fg="red",
        )

        exit(1)

    # load prompt list
    prompts = load_prompts(prompt_list, prompt_repeat)

    # load real images
    images = load_images(real_images, resize) if real_images is not None else None

    # grab models
    clip_model = None

    if embed_text:
        clip_model = load_clip(clip_model=clip_model, device=DEVICE)

    prior_model = None

    if predict_image:
        prior_model = load_prior(checkpoint_path=prior_checkpoint, config_path=prior_config, device=DEVICE)

    # begin creating webdataset
    wds_create(
        prompts=prompts,
        images=images,
        batch_size=batch_size,
        output_folder=output_folder,
        filename=filename,
        clip_model=clip_model,
        prior_model=prior_model,
        predict_image=predict_image,
        embed_text=embed_text,
        tokenize_text=tokenize_text,
    )

    click.secho(f"All done!", fg="green")


if __name__ == "__main__":
    main()
