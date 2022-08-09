import wandb
import torch
import numpy as np
import pandas as pd
from dream_bench.helpers import exists
from dream_bench.load_models import get_aesthetic_model, load_clip
from torchmetrics.image.fid import FrechetInceptionDistance
from typing import List, Any, Dict, Literal
from torchvision.transforms.functional import to_pil_image

# TODO: fix devices
# TODO: fix table formatting

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

        if "model_output" in kwargs and type(kwargs["model_output"] is torch.Tensor):
            kwargs["model_output"].to(device)
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

    def __init__(self, clip_architecture: str = "ViT-L/14", clip_model=None) -> None:
        assert clip_architecture in [
            "ViT-L/14",
            "ViT-B/32",
        ], f"You must choose from the available aesthetic models ViT-L/14 or ViT-B/32 (got: {clip_architecture})"

        # get models
        self.aesthetic_model = get_aesthetic_model(clip_model=clip_architecture)
        self.clip_model, self.preprocess = clip_model if exists(clip_model) else load_clip(clip_model=clip_architecture)

    def _embed(self, images: torch.Tensor):
        "Embed an image with clip and return the encoded images."
        device = images.device

        processed_images = torch.cat(
            [self.preprocess(to_pil_image(img)).unsqueeze(0) for img in images],
            dim=0,
        ).float()

        processed_images = processed_images.to(device)

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

    def __init__(self, clip_architecture: str, clip_model=None) -> None:
        self.clip_model, self.preprocess = clip_model if exists(clip_model) else load_clip(clip_model=clip_architecture)

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
        tokenized_text = model_input["tokenized_text.npy"]

        images = self.preprocess(model_output)
        image_logits, _ = self.clip_model(images, tokenized_text)

        return image_logits.softmax(dim=-1).detach().cpu().numpy()


class Evaluator:
    """
    A class that facilliatates calculating various metrics given input from a model
    """

    def __init__(self, metrics: List[METRICS]) -> None:
        self.data: List[Any] = []
        self.num_entries: int = 0
        self.metric_names: List[str] = list(metrics)
        self.metrics: set = self._metric_factory(metrics)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # wandb tools
        self.table_artifact: wandb.Artifact = wandb.Artifact(
            name="Evaluation Table",
            type="Table",
            description="A persistent table used to log evaluation data.",
        )
        self.table: wandb.Table = wandb.Table(columns=["Key", "Captions", "Predictions"] + list(metrics))

        # add table to artifact to be logged
        self.table_artifact.add(self.table, name="Evaluation Table")

    def evaluate(self, model_input: dict, model_output: torch.Tensor):
        """
        Takes model input and computes metrics for each model output.
        """
        self.num_entries += 1

        scores = {
            "Key": model_input["__key__"],
            "Captions": model_input["caption.txt"],
            "Predictions": [wandb.Image(img) for img in model_output],
        }

        for metric in self.metrics:
            scores[metric.METRIC_NAME] = (
                metric().compute(
                    model_input=model_input,
                    model_output=model_output,
                    device=self.device,
                )
                if metric.USES_INPUT
                else metric().compute(model_output=model_output, device=self.device)
            ).squeeze()

        # cast dict to pandas array for easy adding
        entries = pd.DataFrame.from_dict(scores)[["Key", "Captions", "Predictions", *self.metric_names]].to_numpy()

        for row in entries:
            self.table.add_data(*row)

        wandb.log({"Evaluation Batch": self.table})
        scores.pop("Predictions")
        print(f"Logged!:\n\t{scores}")

    def add_pairs(self, captions: list, images: torch.Tensor):
        """Add caption/image pairs to the table"""

        assert len(captions) == len(images), "Images and captions do not align along first dimension"
        wandb_images = [wandb.Image(img.permute(1, 2, 0).cpu().detach().numpy()) for img in images]
        self.data += list(zip(captions, wandb_images))

    def log_table(self):
        wandb.log({"predictions": wandb.Table(columns=["caption", "image"], data=self.data)})

        print("logged!")

    def _metric_factory(self, metrics: List[METRICS]):
        "Create an iterable of metric classes for the evaluator to use, given a list of metric names."

        metric_set: set = set()

        for metric in metrics:
            if metric == "Aesthetic":
                metric_set.add(Aesthetic)
            elif metrics == "FID":
                metric_set.add(FID)
            elif metrics == "ClipScore":
                metric_set.add(ClipScore)
            else:
                raise KeyError("That metric does not exist, or is mispelled.")

        return metric_set
