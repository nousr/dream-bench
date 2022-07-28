import torch
import wandb
from dream_bench.evaluator import Evaluator
from dream_bench.config import DreamBenchConfig


def evaluate(images: torch.Tensor, input: tuple, metrics: list, evaluator: Evaluator):
    """
    Evaluate the predicted images from a model
    """

    # log the caption image pairs to wandb

    captions = input[0]

    evaluator.add_pairs(captions=captions, images=images)

    # TODO: other metrics computation


def benchmark(adapter, config: DreamBenchConfig):
    """
    Benchmark a model
    """

    # load the dataset from the config
    dataloader = config.dataset.load()

    # init the user's wandb
    wandb.init(
        project=config.wandb.project, entity=config.wandb.entity, name=config.wandb.name
    )

    evaluator = Evaluator()

    # begin benchmarking
    for input in dataloader:
        images = adapter(*input)

        evaluate(
            images=images, input=input, metrics=config.metrics, evaluator=evaluator
        )

    # upload to wandb
    evaluator.log_table()