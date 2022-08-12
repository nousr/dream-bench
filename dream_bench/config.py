from json import load
from pydantic import BaseModel
from typing import List, Optional
from webdataset import WebDataset
from torch.utils.data import DataLoader
from dream_bench.evaluator import Evaluator, METRICS


class DatasetConfig(BaseModel):
    path: str
    batch_size: int

    def load(self):
        dataset = WebDataset(self.path).decode()
        return DataLoader(dataset=dataset, batch_size=self.batch_size)


class WandbConfig(BaseModel):
    entity: str  # your wandb username
    project: str  # project name
    name: Optional[str] = None

    class Config:
        # Each individual log type has it's own arguments that will be passed through the config
        extra = "allow"


class EvaluatorConfig(BaseModel):
    save_path: str = "predictions"
    metrics: List[METRICS] = ["Aesthetic"]
    device: str = "cpu"

    def load(self):
        return Evaluator(metrics=self.metrics, save_path=self.save_path, device=self.device)  # pylint: disable=E1121


class DreamBenchConfig(BaseModel):
    dataset: DatasetConfig
    wandb: WandbConfig
    evaluator: EvaluatorConfig

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path, encoding="utf-8") as f:
            config = load(f)
        return cls(**config)
