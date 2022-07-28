import torch
from dream_bench.config import DreamBenchConfig
from dream_bench.evaluator import Evaluator


def evaluate(images: torch.Tensor, input: tuple, metrics: list, evaluator: Evaluator):
    """
    Evaluate the predicted images from a model
    """

    # log the caption image pairs to wandb

    # TODO: determine best way to log batched input & image pairs

def benchmark(adapter: function, config: DreamBenchConfig):
    """
    Benchmark a model
    """

    # load the dataset from the config
    dataloader = config.data.load()

    evaluator = Evaluator()

    # begin benchmarking
    for input in dataloader:
        images = adapter(**input)

        evaluate(images=images, input=input, metrics=config.metrics, evaluator=evaluator)
