from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import torch
import wandb
from toma import toma
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from webdataset import WebDataset

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

    METRIC_NAME = "FID"
    USES_INPUT = True

    def __init__(self, feature=2048) -> None:
        self.fid = FrechetInceptionDistance(feature=feature)

    def format_tensor(self, x):
        """Format the tensor to be FID friendly (uint8)."""

        # check if images are normalized
        if x.max <= 1.0:
            x = x.mul(255).add(0.5).clamp(0, 255)

        return x.type(torch.uint8)

    @toma.batch(initial_batchsize=512)
    @convert_and_place_input
    def compute(
        self,
        batch_size: int,
        dataset: WebDataset,
        model_output: torch.Tensor,
        device: str,
    ):
        """Compute FID"""
        # place model on device
        self.fid.to(device)

        for model_input in DataLoader(dataset=dataset, batch_size=batch_size):
            real_images = self.format_tensor(model_input["real_image.npy"])
            model_output = self.format_tensor(model_output)

            self.fid.update(imgs=real_images, real=True)
            self.fid.update(imgs=model_output, real=False)

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
        dataset: WebDataset,
        device="cpu",
        clip_architecture=None,
        default_batch_size=128,
    ) -> None:
        self.device: torch.device = device
        self.dataset: WebDataset = dataset
        self.default_batch_size: int = default_batch_size
        self.metric_names: List[str] = list(metrics)
        self.metrics: set = self._metric_factory(metrics, clip_architecture=clip_architecture, device=device)

        self.predictions: List[torch.Tensor] = []

    def record_predictions(self, model_output: torch.Tensor):
        """
        Save the model output.
        """
        self.predictions.append(*model_output.detach().cpu())

    def evaluate(self):
        """
        Evaluate the model's output once all the data has been collected.
        """

        results = {}
        summaries = {}

        for metric in self.metrics:
            kwargs = {"batchsize": self.default_batch_size, "predictions": self.predictions, "device": self.device}
            kwargs.update({"dataset": self.dataset}) if metric.USES_INPUT else kwargs.update({})

            result = metric(**kwargs)

            if metric.IS_SUMMARY:
                summaries[metric] = result
            else:
                results[metric] = result
                summaries[metric] = result.mean()

    def _log(self, results, summaries):
        """
        Log all the results to wandb:
            - Summaries get logged as charts.
            - Results is converted to a `wandb.Table` through `pd.DataFrame`.
        """

        results_table = wandb.Table(dataframe=pd.DataFrame(data=results))
        report = summaries.update({"Evaluation Report": results_table})
        wandb.log(report)

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
