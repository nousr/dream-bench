from json import load
from typing import List, Optional

from pydantic import BaseModel
from webdataset import WebDataset

from dream_bench.evaluator import METRICS, Evaluator


class DatasetConfig(BaseModel):
    path: str
    batch_size: int

    def load(self):
        return WebDataset(self.path).decode()


class WandbConfig(BaseModel):
    entity: str  # your wandb username
    project: str  # project name
    name: Optional[str] = None

    class Config:
        extra = "allow"


class EvaluatorConfig(BaseModel):
    save_path: str = "predictions"
    metrics: List[METRICS] = ["Aesthetic"]
    device: str = "cpu"
    clip_architecture: Optional[str]
    batch_size: int = 128

    def load(self, dataset):
        return Evaluator(
            metrics=self.metrics,
            save_path=self.save_path,
            device=self.device,
            clip_architecture=self.clip_architecture,
            dataset=dataset,
            batch_size=self.batch_size,
        )


class DreamBenchConfig(BaseModel):
    dataset: DatasetConfig
    wandb: WandbConfig
    evaluator: EvaluatorConfig

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path, encoding="utf-8") as f:
            config = load(f)
        return cls(**config)
