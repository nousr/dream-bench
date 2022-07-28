# Dream Bench Pre-Processing Guide

## Overview

`dream_bench` currently provides two major methods for supplying your model with input each with three sub-types for you to choose from.

    1. Text
       1. clip text embeddings (tensor)
       2. tokenized text (tensor)
       3. raw text (string)
    2. Images
       1. clip image embeddings (tensor)
       2. diffusion prior embeddings (tensor)
       3. raw RGB-images (tensor)

## Getting Started

The easiest way to get started is to have a model that can generate images from text, as well as a list of prompts you wish to utilize as your benchmark. For convinience, this repository hosts popular benchmarking prompts, more info [here](../../prompts/README.md).

To turn your prompt list into a dataset, you can simply utilize the provided `prompts2datset.py` script to format everything nicely.

```bash
python prompts2datset.py --prompt-list drawbench.json --output-folder benchmarks/drawbench --tokenize-text True --predict-image True
```

Here we call the prompt-processing script and pass it four arguments:
* First, the path to our prompt list (in this case, we use the provided drawbench list).
* Then, we specify the root folder we'd like to store the resulting benchmark.
* Next, we specify that we'd prefer tokenized text (as opposed to raw strings of text).
* Finally, we specify that we would also like to leverage a diffusion prior to invert the text, and generate a synthetic image embedding for each caption.

At this point, the prompts are stored in webdataset format and can be read by the package (for reference a snippit is included to visualize the dataset)

```python
import webdataset as wds


url = "benchmarks/drawbench/drawbench-0000.tar"

dataset = (
    wds.WebDataset(url)
    .decode()
    .to_tuple("caption.txt", "tokenized_text.npy", "prior_image_embed.npy")
)

loader = wds.WebLoader(dataset)

caption, tokens, prior_embed = next(iter(loader))
```