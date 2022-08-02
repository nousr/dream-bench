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
    entity: str = None  # your wandb username
    name: str = None    # name of the run
    project: str = None # project name


class DreamBenchConfig(BaseModel):
    dataset: DatasetConfig
    metrics: List[AVAILABLE_METRICS] = ["table"]
    wandb: WandbConfig = None
