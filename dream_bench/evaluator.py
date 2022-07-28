import wandb
import torch


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
        images = [wandb.Image(img.permute(1,2,0).cpu().detach().numpy()) for img in images]
        self.data += list(zip(captions, images))

    def log_table(self):
        wandb.log(
            {"predictions": wandb.Table(columns=["caption", "image"], data=self.data)}
        )

        print("logged!")