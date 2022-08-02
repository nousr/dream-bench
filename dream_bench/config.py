from json import load
from pydantic import BaseModel
from typing import List, Literal
from webdataset import WebDataset
from torch.utils.data import DataLoader

AVAILABLE_METRICS = Literal["table"]


class DatasetConfig(BaseModel):
    path: str
    batch_size: int

    def load(self):
        dataset = WebDataset(self.path).decode()
        return DataLoader(dataset=dataset, batch_size=self.batch_size)


class WandbConfig(BaseModel):
    entity: str  # your wandb username
    project: str  # project name
    name: str = None  # name of the run


class DreamBenchConfig(BaseModel):
    dataset: DatasetConfig
    wandb: WandbConfig
    metrics: List[AVAILABLE_METRICS] = ["table"]

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            config = load(f)
        return cls(**config)
