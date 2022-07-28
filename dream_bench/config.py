from pydantic import BaseModel
from typing import List, Literal
from webdataset import WebDataset
from torch.utils.data import DataLoader

AVAILABLE_METRICS = Literal["table"]
AVAILABLE_KEYS = Literal[
    "tokenized_text.npy",
    "clip_text_embed.npy",
    "raw_image.npy",
    "clip_image_embed.npy",
    "prior_image_embed.npy",
]


class DatasetConfig(BaseModel):
    path: str
    keys: List[AVAILABLE_KEYS]
    batch_size: int

    def load(self):
        dataset = WebDataset(self.path).decode().to_tuple("caption.txt", *self.keys)
        return DataLoader(dataset=dataset, batch_size=self.batch_size)


class WandbConfig(BaseModel):
    entity: str = None  # your wandb username
    name: str = None    # name of the run
    project: str = None # project name


class DreamBenchConfig(BaseModel):
    dataset: DatasetConfig
    metrics: List[AVAILABLE_METRICS] = ["table"]
    wandb: WandbConfig = None
