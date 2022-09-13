from torch.utils.data import DataLoader

from dream_bench.config import DreamBenchConfig


def benchmark(model, adapter, config: DreamBenchConfig):
    """
    Benchmark a model
    """

    # load the dataset from the config
    dataset = config.dataset.load()

    # init the user's wandb
    config.wandb.init()

    evaluator = config.evaluator.load(dataset=dataset)

    # begin benchmarking
    for input_dict in DataLoader(dataset=dataset, batch_size=config.dataset.batch_size):
        images = adapter(model, input_dict)
        evaluator.record_predictions(model_output=images)

    # compute benchmarks
    evaluator.evaluate()
