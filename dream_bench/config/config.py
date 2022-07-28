from pydantic import BaseModel
from typing import List, Literal
from webdataset import WebDataset
from torch.utils.data import DataLoader

AVAILABLE_METRICS = Literal["table"]
AVAILABLE_KEYS = Literal[
    "raw_text",
    "tokenized_text",
    "clip_text_embed",
    "raw_image",
    "clip_image_embed",
    "prior_image_embed",
]


class DatasetConfig(BaseModel):
    path: str
    keys: List[AVAILABLE_KEYS]
    batch_size: int

    def load(self):
        dataset = WebDataset(self.path).decode().to_tuple(*self.keys)
        return DataLoader(dataset=dataset, batch_size=self.batch_size)


class WandbConfig(BaseModel):
    entity: str = None  # your wandb username
    name: str = None  # name of the run
    project: bool = None  # project name


class DreamBenchConfig(BaseModel):
    dataset: DatasetConfig
    metrics: List[AVAILABLE_METRICS] = ["table"]
    wandb: WandbConfig = None