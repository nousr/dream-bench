import wandb
import torch


class Evaluator:
    """
    A class that facilliatates calculating various metrics given input from a model
    """

    def __init__(self) -> None:
        self.data = list()

    def add_pair(self, captions: torch.tensor, images: str):
        """Add an caption/image pair to the table"""

        assert len(captions) == len(
            images
        ), "Images and captions do not align along first dimension"

        self.data.append(zip(captions, images))

    def log_table(self):
        wandb.log(
            {"predictions": wandb.Table(columns=["caption", "image"], data=self.data)}
        )