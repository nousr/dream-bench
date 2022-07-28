#! python

import os
import json
import click
from torch.nn import Module
from torch.cuda import is_available
from webdataset import ShardWriter
from dream_bench.load_models import load_clip, load_prior, filename_from_path
from dream_bench.tokenizer import tokenizer

DEVICE = "cuda" if is_available() else "cpu"
TOKENIZER_CONTEXT_LENGTH = 77
TOKENIZER_TRUNCATE_TEXT = True


def wds_create(
    prompts: list,
    output_folder: str,
    filename: str,
    clip_model: Module,
    prior_model: Module,
    options: dict,
):
    """
    Create a webdataset for the prompt list.

    # TODO: docstring
    """
    filename = filename.split(".")[0]
    with ShardWriter(pattern=f"{output_folder}/{filename}-%04d.tar") as sink:
        for idx, prompt in enumerate(prompts):
            tar_keys = {"__key__": "%04d" % idx}

            # always save original caption as string & tokenize for simplicity

            tar_keys["caption.txt"] = prompt
            tokenized_prompt = tokenizer.tokenize(
                prompt,
                context_length=TOKENIZER_CONTEXT_LENGTH,
                truncate_text=TOKENIZER_TRUNCATE_TEXT,
            ).to(DEVICE)

            for option, in_use in options.items():
                if not in_use:
                    continue

                if option == "tokenized_text":
                    tar_keys[option + ".npy"] = tokenized_prompt.detach().cpu().numpy()

                elif option == "clip_text_embed":
                    tar_keys[option + ".npy"] = (
                        clip_model.encode_text(tokenized_prompt).detach().cpu().numpy()
                    )

                elif option == "prior_image_embed":
                    tar_keys[option + ".npy"] = (
                        prior_model.sample(tokenized_prompt).detach().cpu().numpy()
                    )

            sink.write(tar_keys)


@click.command()
@click.option(
    "--prompt-list", help="A json file containing a list of prompts.", required=True
)
@click.option(
    "--output-folder", help="parent folder of the dataset to be created", required=True
)
@click.option(
    "--override-folder",
    default=False,
    help="wether to overrite the output folder if it exists",
)
@click.option(
    "--embed-text",
    default=False,
    help="Wether to use clip to embed the prompts as text embeddings",
)
@click.option("--tokenize-text", default=True, help="Wether to tokenize the text.")
@click.option(
    "--predict-image",
    default=True,
    help="Wether to use a diffusion prior to create an image embedding from each prompt.",
)
@click.option(
    "--clip-model", default=None, help="Clip model to use for embedding images."
)
@click.option(
    "--prior-checkpoint",
    default=None,
    help="A local path or web url to a prior config file.",
)
@click.option(
    "--prior-config",
    default=None,
    help="A local path or web url to a prior checkpoint.",
)
def main(
    prompt_list,
    output_folder,
    override_folder,
    embed_text,
    tokenize_text,
    predict_image,
    clip_model,
    prior_checkpoint,
    prior_config,
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
            f"WARNING: The folder you specified already exists. This is a safety measure to avoid destroying existing work. If you wish to override it, please re-run this script with --override-folder=True.",
            fg="red",
        )

        exit(1)

    # load prompt list

    with open(prompt_list, "r") as f:
        prompts = json.load(f)

    # grab models

    clip_model = None

    if embed_text:
        clip_model = load_clip(clip_model=clip_model)

    prior_model = None

    if predict_image:
        prior_model = load_prior(
            checkpoint_path=prior_checkpoint, config_path=prior_config
        )

    # begin creating webdataset

    wds_create(
        prompts=prompts,
        output_folder=output_folder,
        filename=filename,
        clip_model=clip_model,
        prior_model=prior_model,
        options=options,
    )

    click.secho(f"All done!", fg="green")


if __name__ == "__main__":
    main()
