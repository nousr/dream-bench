import os
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd
import torch
import wandb
from torch.nn.functional import cosine_similarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_pil_image

from dream_bench.helpers import exists
from dream_bench.load_models import get_aesthetic_model, load_clip
from dream_bench.tokenizer import tokenizer

TOKENIZER_CONTEXT_LENGTH = 77
TOKENIZER_TRUNCATE_TEXT = True

# Custom Types

METRICS = Literal["FID", "Aesthetic", "ClipScore"]

# decorator


def convert_and_place_input(fn):
    "Cast all numpy arrays to torch tensors"

    def inner(*args, **kwargs):
        model_input: dict = kwargs.pop("model_input") if "model_input" in kwargs else None

        device = kwargs["device"] if "device" in kwargs else None

        assert exists(device), "A device must be present to ensure proper resource usage."

        # cast all numpy arrays to torch tensors
        if exists(model_input):
            for k, v in model_input.items():
                if not k.endswith(".npy"):
                    continue

                if isinstance(k, np.ndarray):
                    model_input[k] = torch.from_numpy(v)  # pylint: disable=E1137

                # place on correct device
                if isinstance(model_input[k], torch.Tensor):
                    model_input[k].to(device)

                else:
                    raise AssertionError(f"Input is not a torch tensor or numpy array. Got {type(model_input[k])}")

            # replace variable
            kwargs["model_input"] = model_input

        # cast model output
        if "model_output" in kwargs and isinstance(kwargs["model_output"], torch.Tensor):
            kwargs["model_output"] = kwargs["model_output"].to(device)
        else:
            raise AssertionError("Model output is not a torch tensor.")

        return fn(*args, **kwargs)

    return inner


class FID:
    """
    The FID metric computes the Frechet Inception Distance of predicted images given a distribution of real images.
    """

    # FIXME: implement upfront cache-ing of the real images to save on time

    METRIC_NAME = "FID"
    USES_INPUT = True

    def __init__(self, feature=64) -> None:
        self.fid = FrechetInceptionDistance(feature=feature)

    def _reset(self):
        self.fid.reset()

    @convert_and_place_input
    def compute(
        self,
        model_input: Dict[str, torch.Tensor],
        model_output: torch.Tensor,
        device: str,
    ):
        """Compute FID"""
        # reset model
        self._reset()

        # place model on device
        self.fid.to(device)

        # ensure the metric can be computed
        assert exists(model_input["raw_image.npy"]), "You must provide a distribution of real images to compute FID."

        real_images = model_input["raw_image.npy"].mul(255).add(0.5).clamp(0, 255).type(torch.uint8)

        model_output = model_output.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)

        # Update model
        self.fid.update(imgs=real_images, real=True)
        self.fid.update(imgs=model_output, real=False)

        # Compute and return FID
        return self.fid.compute().detach().cpu().numpy()


class Aesthetic:
    """
    Predict the aesthetic quality of images using a pre-trained aesthetic rating model.
    """

    METRIC_NAME = "Aesthetic"
    USES_INPUT = False

    def __init__(self, clip_model, clip_preprocess, clip_architecture) -> None:
        # get models
        self.aesthetic_model = get_aesthetic_model(clip_model=clip_architecture)
        self.clip_model, self.preprocess = clip_model, clip_preprocess

    def _embed(self, images: torch.Tensor):
        "Embed an image with clip and return the encoded images."
        device = images.device

        processed_images = self.preprocess(images).to(device)

        return self.clip_model.encode_image(processed_images)

    @convert_and_place_input
    def compute(self, model_output: torch.Tensor, device: str):
        # place models on proper device
        self.clip_model.to(device)
        self.aesthetic_model.to(device)

        embeddings = self._embed(model_output).float()

        return self.aesthetic_model(embeddings).detach().cpu().numpy()


class ClipScore:
    """
    Compute the clip-similarity of an image and caption using a pre-trained CLIP model
    """

    METRIC_NAME = "ClipScore"
    USES_INPUT = True

    def __init__(self, clip_model, clip_preprocess) -> None:
        self.clip_model, self.preprocess = clip_model, clip_preprocess

    @convert_and_place_input
    def compute(
        self,
        model_input: Dict[str, torch.Tensor],
        model_output: torch.Tensor,
        device: str,
    ):

        "Compute the clip score of a given image/caption pair."

        # place models on proper device
        self.clip_model.to(device)

        # unpack model input
        if "tokenized_text.npy" not in model_input:
            captions = model_input["caption.txt"]
            model_input["tokenized_text.npy"] = tokenizer.tokenize(
                captions, context_length=TOKENIZER_CONTEXT_LENGTH, truncate_text=TOKENIZER_TRUNCATE_TEXT
            )

        tokenized_text = model_input["tokenized_text.npy"].to(device)

        images = self.preprocess(model_output).to(device)

        image_embeddings = self.clip_model.encode_image(images)
        text_embeddings = self.clip_model.encode_text(tokenized_text)

        cos_similarities = cosine_similarity(image_embeddings, text_embeddings, dim=1)

        return cos_similarities.detach().cpu().numpy()


class Evaluator:
    """
    A class that facilliatates calculating various metrics given input from a model
    """

    def __init__(
        self,
        metrics: List[METRICS],
        save_path: str,
        device="cpu",
        clip_architecture=None,
    ) -> None:
        self.data: List[Any] = []
        self.num_entries: int = 0
        self.evaluation_iteration: int = 0
        self.metric_names: List[str] = list(metrics)
        self.metrics: set = self._metric_factory(metrics, clip_architecture=clip_architecture, device=device)

        self.device = device

        self.save_path = Path(save_path)
        os.makedirs(save_path, exist_ok=True)

        self.prediction_table = wandb.Table(columns=["Key", "Captions", "Predictions"])
        self.metric_table = wandb.Table(columns=["Key", *self.metric_names])

    def _record_predictions(self, model_input: dict, model_output: torch.Tensor):
        """
        Record model output and save to disk, then track it in a table using wandb.
        """
        captions = model_input["caption.txt"]
        keys = model_input["__key__"]

        for key, caption, prediction in zip(keys, captions, model_output):
            # save image to disk
            pil_image = to_pil_image(prediction, mode="RGB")
            prediction_path = Path(f"{self.save_path}/prediction_{self.num_entries:06d}.jpg")
            pil_image.save(prediction_path)

            # create associated wandb.Image and save to wandb as a file
            wandb_image = wandb.Image(str(prediction_path), caption=caption)
            self.prediction_table.add_data(key, caption, wandb_image)
            wandb.save(str(prediction_path))

            self.num_entries += 1

    def _record_metrics(self, model_input: dict, model_output: torch.Tensor):
        """
        Evaluate a model's output and record the results to a table.
        """

        scores = {
            "Key": model_input["__key__"],
        }

        # loop through each metric and compute the results
        for metric in self.metrics:
            scores[metric.METRIC_NAME] = (
                metric.compute(
                    model_input=model_input,
                    model_output=model_output,
                    device=self.device,
                )
                if metric.USES_INPUT
                else metric.compute(model_output=model_output, device=self.device)
            ).squeeze()

        # cast dict to pandas array for easy adding
        for row in pd.DataFrame.from_dict(scores)[["Key", *self.metric_names]].to_numpy():
            self.metric_table.add_data(*row)

    def evaluate(self, model_input: dict, model_output: torch.Tensor):
        """
        Record the model's output and incrementally evaluate it.
        """
        self._record_predictions(model_input=model_input, model_output=model_output)
        self._record_metrics(model_input=model_input, model_output=model_output)
        print(f"Evaluated Up To #{self.num_entries:06d}")

    def log(self):
        """
        Log all the results to wandb.
        """

        # join both tables and publish

        master_table = wandb.JoinedTable(self.prediction_table, self.metric_table, join_key="Key")

        # log the tables
        wandb.log(
            {
                "Metric Table": self.metric_table,
                "Predictions Table": self.prediction_table,
                "Evaluation Report": master_table,
            }
        )

        # compute basic average of each metric
        average_scores = {}

        for metric in self.metric_names:
            average_scores[f"Average {metric}"] = np.array(self.metric_table.get_column(metric)).mean()

        wandb.log(average_scores, step=self.evaluation_iteration)

    def _metric_factory(self, metrics: List[METRICS], clip_architecture, device="cpu"):
        "Create an iterable of metric classes for the evaluator to use, given a list of metric names."

        metric_set: set = set()

        # a clip model to use across all metrics
        clip_model, clip_preprocess = None, None

        for metric in metrics:
            if metric == "FID":
                metric_set.add(FID())

            elif metric == "Aesthetic":
                assert exists(clip_architecture), "A clip architecture is required to use the Aesthetic Metric."
                if not exists(clip_model):
                    clip_model, clip_preprocess = load_clip(clip_model=clip_architecture, device=device)

                metric_set.add(
                    Aesthetic(
                        clip_model=clip_model,
                        clip_preprocess=clip_preprocess,
                        clip_architecture=clip_architecture,
                    )
                )

            elif metric == "ClipScore":
                assert exists(clip_architecture), "A clip architecture is required to use the Clip Score."
                if not exists(clip_model):
                    clip_model, clip_preprocess = load_clip(clip_model=clip_architecture, device=device)
                metric_set.add(ClipScore(clip_model=clip_model, clip_preprocess=clip_preprocess))

            else:
                raise KeyError("That metric does not exist, or is mispelled.")

        return metric_set
