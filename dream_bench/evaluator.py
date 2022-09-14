from typing import List, Literal

import numpy as np
import pandas as pd
import torch
import wandb
from toma import toma
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from webdataset import WebDataset

from dream_bench.helpers import exists
from dream_bench.load_models import get_aesthetic_model, load_clip
from dream_bench.tokenizer import tokenizer

TOKENIZER_CONTEXT_LENGTH = 77
TOKENIZER_TRUNCATE_TEXT = True

# Custom Types

METRICS = Literal["FID", "Aesthetic", "ClipScore"]

"""
TODO:
Overhaul device placement.
"""

# helpers


def make_loaders(batchsize: int, model_input: WebDataset, model_output: List[torch.Tensor]):
    """
    Create Dataloaders from the two inputs with a given batchsize.

    Return:
        - A zipped version of both datasets to sample from.
    """

    model_input = DataLoader(dataset=model_input, batch_size=batchsize)
    model_output = DataLoader(dataset=TensorDataset(torch.as_tensor(model_output)), batch_size=batchsize)

    return zip(model_input, model_output)


class FID:
    """
    The FID metric computes the Frechet Inception Distance of predicted images given a distribution of real images.
    """

    METRIC_NAME = "FID"
    USES_INPUT = True
    IS_SUMMARY = True

    def __init__(self, feature=2048) -> None:
        self.fid = FrechetInceptionDistance(feature=feature)

    def format_tensor(self, x):
        """Format the tensor to be FID friendly (uint8)."""

        # check if images are normalized

        if x.max().item() <= 1.0:
            x = x.float()
            x = x.mul(255).add(0.5).clamp(0, 255)

        return x.type(torch.uint8)

    def compute(
        self,
        dataset: WebDataset,
        predictions: torch.Tensor,
        device: str,
    ):
        @toma.execute.batch(initial_batchsize=512)
        def _block(batchsize):
            # place model on device
            self.fid.to(device)

            # create dataloader
            loader = make_loaders(batchsize=batchsize, model_input=dataset, model_output=predictions)

            for model_input, model_output in loader:
                real_images = self.format_tensor(model_input["real_image.npy"])
                model_output = self.format_tensor(model_output[0])

                real_images = real_images.to(device)
                model_output = model_output.to(device)

                self.fid.update(imgs=real_images, real=True)
                self.fid.update(imgs=model_output, real=False)

            results = self.fid.compute().detach().cpu().numpy().squeeze()

            # move model back to cpu
            self.fid = self.fid.to("cpu")

            return results

        return _block


class Aesthetic:
    """
    Predict the aesthetic quality of images using a pre-trained aesthetic rating model.
    """

    METRIC_NAME = "Aesthetic"
    USES_INPUT = False
    IS_SUMMARY = False

    def __init__(self, clip_model, clip_preprocess, clip_architecture) -> None:
        # get models
        self.aesthetic_model = get_aesthetic_model(clip_model=clip_architecture)
        self.clip_model, self.preprocess = clip_model, clip_preprocess

    def _embed(self, images: torch.Tensor):
        "Embed an image with clip and return the encoded images."
        device = images.device
        processed_images = self.preprocess(images).to(device)

        return self.clip_model.encode_image(processed_images)

    def compute(self, predictions: torch.Tensor, device: str):
        @toma.execute.batch(initial_batchsize=512)
        def _block(batchsize):
            # place models on proper device
            self.clip_model.to(device)
            self.aesthetic_model.to(device)

            all_aesthetic_scores = []

            for idx in range(0, len(predictions), batchsize):
                sample = predictions[idx : idx + batchsize]
                sample = sample.to(device)

                embeddings = self._embed(sample).float()

                all_aesthetic_scores += [*self.aesthetic_model(embeddings).detach().cpu().numpy()]

            # move models back to cpu
            self.clip_model.to("cpu")
            self.aesthetic_model.to("cpu")

            return np.array(all_aesthetic_scores).squeeze()

        return _block


class ClipScore:
    """
    Compute the clip-similarity of an image and caption using a pre-trained CLIP model
    """

    METRIC_NAME = "ClipScore"
    USES_INPUT = True
    IS_SUMMARY = False

    def __init__(self, clip_model, clip_preprocess) -> None:
        self.clip_model, self.preprocess = clip_model, clip_preprocess

    def compute(
        self,
        dataset: WebDataset,
        predictions: torch.Tensor,
        device: str,
    ):
        "Compute the clip score of a given image/caption pair."

        @toma.execute.batch(initial_batchsize=512)
        def _block(batchsize):
            # place models on proper device
            self.clip_model.to(device)

            loader = make_loaders(batchsize=batchsize, model_input=dataset, model_output=predictions)

            all_similarities = []

            for model_input, model_output in loader:
                if "tokenized_text.npy" not in model_input:
                    captions = model_input["caption.txt"]
                    model_input["tokenized_text.npy"] = tokenizer.tokenize(
                        captions, context_length=TOKENIZER_CONTEXT_LENGTH, truncate_text=TOKENIZER_TRUNCATE_TEXT
                    )

                model_output = model_output[0].to(device)
                tokenized_text = model_input["tokenized_text.npy"].to(device)

                images = self.preprocess(model_output).to(device)

                image_embeddings = self.clip_model.encode_image(images)
                text_embeddings = self.clip_model.encode_text(tokenized_text)

                cos_similarities = cosine_similarity(image_embeddings, text_embeddings, dim=1)

                all_similarities += [*cos_similarities.detach().cpu().numpy()]

            # move model back to cpu
            self.clip_model.to("cpu")

            return np.array(all_similarities).squeeze()

        return _block


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
        Save the model output along with the prompt.
        """
        self.predictions += [model_output.detach().cpu()]

    def evaluate(self):
        """
        Evaluate the model's output once all the data has been collected.
        """
        tensor_predictions = torch.concat(self.predictions, dim=0)

        # keep track of results and summaries
        results = {}
        summaries = {}

        # compute the requested metrics
        for metric in self.metrics:
            kwargs = {"predictions": tensor_predictions, "device": self.device}
            kwargs.update({"dataset": self.dataset}) if metric.USES_INPUT else kwargs.update({})

            result = metric.compute(**kwargs)

            if metric.IS_SUMMARY:
                summaries[metric.METRIC_NAME] = result
            else:
                results[metric.METRIC_NAME] = result
                summaries[metric.METRIC_NAME] = result.mean()

        self._log(results, summaries)

    def _to_wandb_image(self, images):
        """
        Convert a list of images into a list of wandb images.
        """

        wandb_images = []

        for image in images:
            wandb_images.append(wandb.Image(data_or_path=image.permute(1, 2, 0).detach().cpu().numpy()))

        return wandb_images

    def _log(self, results, summaries):
        """
        Log all the results to wandb:
            - Summaries get logged as charts.
            - Results is converted to a `wandb.Table` through `pd.DataFrame`.
        """
        results_df = pd.DataFrame.from_dict(data=results)
        results_table = wandb.Table(dataframe=results_df)

        results_table.add_column(name="Caption", data=self._get_captions())
        results_table.add_column(name="Prediction", data=self._to_wandb_image(torch.concat(self.predictions, dim=0)))
        summaries.update({f"Evaluation Report: #{wandb.run.step}": results_table})

        wandb.log(summaries)

    def _get_captions(self):
        """
        Extract the captions as a list of strings from the dataset.
        """

        captions = []

        for item in self.dataset:
            captions.append(item["caption.txt"])

        return captions

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
