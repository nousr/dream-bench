import wandb
import torch
from dream_bench.helpers import exists
from dream_bench.load_models import get_aesthetic_model, load_clip
from torchmetrics.image.fid import FrechetInceptionDistance
from typing import List, Any

# TODO: think about relaxing the constraint that metrics must return a single float

# decorators


def with_uint8(fn):
    def inner(images: torch.Tensor, real_images: torch.Tensor, *args, **kwargs):
        images = images.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)
        real_images = real_images.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)

        return fn(images, real_images, *args, **kwargs)

    return inner


class FID:
    """
    The FID metric computes the Frechet Inception Distance of predicted images given a distribution of real images.
    """

    # FIXME: implement upfront cache-ing of the real images to save on time

    METRIC_NAME = "FID"

    def __init__(self, feature=64) -> None:
        self.fid = FrechetInceptionDistance(feature=feature)

    def _reset(self):
        self.fid.reset()

    @with_uint8
    def compute(self, images: torch.Tensor, real_images: torch.Tensor):
        """Compute FID"""
        # reset model
        self._reset()

        assert exists(
            real_images
        ), "You must provide a distribution of real images to compute FID."

        # Update model
        self.fid.update(imgs=real_images, real=True)
        self.fid.update(imgs=images, real=False)

        # Compute and return FID
        return self.fid.compute().cpu().item()


class Aesthetic:
    """
    Predict the aesthetic quality of images using a pre-trained aesthetic rating model.
    """

    METRIC_NAME = "Aesthetic"

    def __init__(self, clip_architecture: str = "ViT-L/14", clip_model=None) -> None:
        assert clip_model in [
            "vit_l_14",
            "vit_b_32",
        ], "You must choose from the available aesthetic models ViT-L/14 or ViT-B/32"

        # get models
        self.aesthetic_model = get_aesthetic_model(clip_model=clip_architecture)

        self.clip_model, self.preprocess = (
            clip_model
            if exists(clip_model)
            else load_clip(clip_model=clip_architecture)
        )

    def _embed(self, images: torch.Tensor, *args, **kwargs):
        images = self.preprocess(images, *args, **kwargs)
        return self.clip_model.encode_image(images, *args, **kwargs)

    def compute(self, images: torch.Tensor, *args, **kwargs):
        embeddings = self._embed(images, *args, **kwargs)
        return self.aesthetic_model(embeddings, *args, **kwargs).cpu().mean().item()


class ClipScore:
    """
    Compute the clip-similarity of an image and caption using a pre-trained CLIP model
    """

    METRIC_NAME = "ClipScore"

    def __init__(self, clip_architecture: str, clip_model=None) -> None:
        self.clip_model, self.preprocess = (
            clip_model
            if exists(clip_model)
            else load_clip(clip_model=clip_architecture)
        )

    def compute(
        self, images: torch.Tensor, tokenized_text: torch.Tensor, *args, **kwargs
    ):
        images = self.preprocess(images, *args, **kwargs)
        image_logits, _ = self.clip_model(images, tokenized_text, *args, **kwargs)

        return image_logits.softmax(dim=-1).cpu().mean().item()

class WandbTable:
    """
    Generate a WANDB caption/image table given a list of images and captions
    """

    def __init__(self) -> None:
        pass

class Evaluator:
    """
    A class that facilliatates calculating various metrics given input from a model
    """

    def __init__(self) -> None:
        self.data: List[Any] = []

    def add_pairs(self, captions: list, images: torch.Tensor):
        """Add caption/image pairs to the table"""

        assert len(captions) == len(
            images
        ), "Images and captions do not align along first dimension"
        wandb_images = [
            wandb.Image(img.permute(1, 2, 0).cpu().detach().numpy()) for img in images
        ]
        self.data += list(zip(captions, wandb_images))

    def log_table(self):
        wandb.log(
            {"predictions": wandb.Table(columns=["caption", "image"], data=self.data)}
        )

        print("logged!")
