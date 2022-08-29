import wandb
from torch.utils.data import DataLoader

from dream_bench.config import DreamBenchConfig


def benchmark(adapter, config: DreamBenchConfig):
    """
    Benchmark a model
    """

    # load the dataset from the config
    dataset = config.dataset.load()

    # init the user's wandb
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, name=config.wandb.name)

    evaluator = config.evaluator.load(dataset=dataset)

    # begin benchmarking
    for input_dict in DataLoader(dataset=dataset, batch_size=config.dataset.batch_size):
        images = adapter(input_dict)
        evaluator.evaluate(model_input=input_dict, model_output=images)

    evaluator.log()
