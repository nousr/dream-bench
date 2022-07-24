import os
import json
import click
import torch
import webdataset as wds


@click.command()
@click.option("--prompt-list", help="A json file containing a list of prompts.")
@click.option("--embed-text", default=False, help="Wether to use clip to embed the prompts as text embeddings")
@click.option("--tokenize-text", default=True, help="Wether to tokenize the text.")
@click.option("--predict-image", default=True, help="Wether to use a diffusion prior to create an image embedding from each prompt.")
# @click.option("--clip-model", help="Clip model to use for embedding images.")
@click.option("--prior-config", default="default", help="A local path or web url to a prior config file.")
@click.option("--prior-checkpoint", default="default", help="A local path or web url to a prior checkpoint.")
def main(prompt_list, embed_text, tokenize_text, predict_image):
    file_name = prompt_list.split("/")[-1]
    options = {"embedding text": embed_text, "tokenizing text": tokenize_text, "predicting images": predict_image}

    click.secho(f"Generating benchmark dataset for: {file_name}", fg="bright_cyan", underline=True)

    for k,v in options.items():
        click.secho(f"\t* {k} : {v}", fg="green" if v else "red")

    # grab models

if __name__ == "__main__":
    main()