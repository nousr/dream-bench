import wandb
from dream_bench.config import DreamBenchConfig


def benchmark(adapter, config: DreamBenchConfig):
    """
    Benchmark a model
    """

    # load the dataset from the config
    dataloader = config.dataset.load()

    # init the user's wandb
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, name=config.wandb.name)

    evaluator = config.evaluator.load()

    # begin benchmarking
    for input_dict in dataloader:
        images = adapter(input_dict)
        evaluator.evaluate(model_input=input_dict, model_output=images)

    evaluator.log()
