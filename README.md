<h1 align="center"> 🖼️ Dream Bench 📊</h1>


## What does it do?

`dream_bench` provides a simplified interface for benchmarking your image-generation.

This repository also hosts common prompt-lists for benchmarking, as well as instructions for preparing a dataset to perform a more comprehensive evaluation evaluation.

## How does it work?

To start off, you will need to create an evaluation adapter for you model so that it can interface with `dream_bench`. This function will be called by `dream_bench` and expect an image in return.

```python
from dream_bench.utils import check_args

class DALLE2():
    def __init__(self, *args, **kwargs):
        super().__init__()
        ...

    def sample(self, img_emb, txt_emb):
        ...

    def my_evaluation_harness(self, conditioning_args):

        # extract what you need to from the conditioning arguments

        img_emb = conditioning_args["prior_image_embedding.npy"]
        txt_emb = conditioning_args["clip_text_embedding.npy"]

        # sample with your model's function

        predicted_image = self.sample(img_emb=img_emb, txt_emb=txt_emb)

        # return the image(s) to be evaluated by dream bench

        return predicted_image
```

Once you have a function that will accept `conditioning_args` and return an `image`, you can pass this function to `dream_bench` to handle the rest!

```python
from dream_bench import benchmark

# specify what you'd like to benchmark

benchmark_config = {
    "drawbench" : {
         "metrics": ["fid", "clip_sim", "aesthetic_rating"],
        "dataset": {
            "path": "benchmarks/dream_bench/dream_bench-0000.tar",
            "batch_size" 8
        }
    }
}

def train(model, dataloader, epochs):
    for epoch in range(epochs):
        for x,y in dataloader:
            # do training

            ...

            # benchmark on some interval

            if time_to_benchmark:
                benchmark(adapter=model.my_evaluation_harness, **benchmark_config)
```

If you're done training and would like to benchmark a pre-trained model it can be done in the following way.

```python
from dream_bench import benchmark

# specify what you'd like to benchmark

benchmark_config = {
    "benchmarks": ["drawbench", "dalle_mini", "simulacra"],
    "metrics": ["fid", "clip_sim", "aesthetic_rating"],
    "inputs": ["text", "image_embedding"]
}

# if your model doesn't have an adapter, you can create one now

def my_evaluation_harness(self, conditioning_args):

        # extract what you need to from the conditioning arguments

        img_emb = conditioning_args["prior_image_embedding.npy"]
        txt_emb = conditioning_args["clip_text_embedding.npy"]

        # sample with your model's function

        predicted_image = self.sample(img_emb=img_emb, txt_emb=txt_emb)

        # return the image(s) to be evaluated by dream bench

        return predicted_image

def main():
    # load your model

    model = load_model()

    # call benchmark outside of training

    benchmark(adapter=model.my_evaluation_harness, **benchmark_config)

if __name__ == "__main__":
    main()
```

## Setup

`dream_bench` works on the `Webdataset` format.

Before you run evaluation, you must preprocess your dataset/prompt-list so that is compatible with `dream_bench`.

For more information on how this can be done, take a look at the [dedicated readme](dream_bench/preprocessing/README.md)

## Tracking

To track your experiments, `dream_bench` utilizes [weights & biases](https://wandb.ai). With `wandb` it is possible to easily view your generations in an easy-to-read format, compile reports, and query against/across your runs.

Additional support may be added for other trackers in the future.

---

## ToDo
- [x] Determine dataset format
- [x] Provide scripts for generating datasets from prompt lists
- [ ] Provide guide/scripts for formatting other datasets/url-lists
- [ ] Build out benchmarking interface for clip-based models
- [ ] Complete wandb integration
- [ ] Formalize configuration & available arguments
- [ ] Publish to PyPi
- [ ] Add support for distributed benchmarking