import wandb
import torch
from dream_bench.helpers import exists
from dream_bench.load_models import get_aesthetic_model, load_clip
from torchmetrics.image.fid import FrechetInceptionDistance

# TODO: think about relaxing the constraint that metrics must return a single float


class Metric:
    """
    A metric is an abstract class that defines an interface for the evaluator to recieve a value.

    All metrics should implement the `compute()` function which does (at least) the following:
        - Input: A `torch.Tensor` object of the shape [B, C, H, W].
        - Returns: A float value representing the "score".
    """

    METRIC_NAME = None

    def __init__(self) -> None:
        assert exists(self.METRIC_NAME), "You must supply a name for this metric."

    def compute(self, images: torch.Tensor, *args, **kwargs):
        raise NotImplementedError


class FID(Metric):
    # FIXME: implement upfront cache-ing of the real images to save on time

    METRIC_NAME = "FID"

    def __init__(self, feature=64) -> None:
        super().__init__()
        self.fid = FrechetInceptionDistance(feature=feature)

    def with_uint8(fn):
        def inner(images: torch.Tensor, real_images: torch.Tensor, *args, **kwargs):
            images = images.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)
            real_images = real_images.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)

            return fn(images, real_images, *args, **kwargs)

        return inner

    def _reset(self):
        self.fid.reset()

    @with_uint8
    def compute(self, images: torch.Tensor, real_images: torch.Tensor, *args, **kwargs):
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


class Aesthetic(Metric):
    METRIC_NAME = "Aesthetic"

    def __init__(
        self, clip_architecture: str = "ViT-L/14", clip_model=None, *args, **kwargs
    ) -> None:
        super().__init__()
        assert clip_model in [
            "vit_l_14",
            "vit_b_32",
        ], "You must choose from the available aesthetic models ViT-L/14 or ViT-B/32"

        # get models
        self.aesthetic_model = get_aesthetic_model(
            clip_model=clip_architecture, *args, **kwargs
        )
        self.clip_model, self.preprocess = (
            clip_model
            if exists(clip_model)
            else load_clip(clip_model=clip_architecture, *args, **kwargs)
        )

    def _embed(self, images: torch.Tensor, *args, **kwargs):
        images = self.preprocess(images, *args, **kwargs)
        return self.clip_model.encode_image(images, *args, **kwargs)

    def compute(self, images: torch.Tensor, *args, **kwargs):
        embeddings = self._embed(images, *args, **kwargs)
        return self.aesthetic_model(embeddings, *args, **kwargs).cpu().mean().item()


class ClipScore(Metric):
    METRIC_NAME = "ClipScore"

    def __init__(
        self, clip_architecture: str, clip_model=None, *args, **kwargs
    ) -> None:
        super().__init__()

        self.clip_model, self.preprocess = (
            clip_model
            if exists(clip_model)
            else load_clip(clip_model=clip_architecture, *args, **kwargs)
        )

    def compute(
        self, images: torch.Tensor, tokenized_text: torch.Tensor, *args, **kwargs
    ):
        # preprocess images for clip
        images = self.preprocess(images, *args, **kwargs)

        # compute logits
        image_logits, _ = self.clip_model(images, tokenized_text, *args, **kwargs)

        return image_logits.softmax(dim=-1).cpu().mean().item()


class Evaluator:
    """
    A class that facilliatates calculating various metrics given input from a model
    """

    def __init__(self) -> None:
        self.data = list()

    def add_pairs(self, captions: list, images: torch.tensor):
        """Add caption/image pairs to the table"""

        assert len(captions) == len(
            images
        ), "Images and captions do not align along first dimension"
        images = [
            wandb.Image(img.permute(1, 2, 0).cpu().detach().numpy()) for img in images
        ]
        self.data += list(zip(captions, images))

    def log_table(self):
        wandb.log(
            {"predictions": wandb.Table(columns=["caption", "image"], data=self.data)}
        )

        print("logged!")
