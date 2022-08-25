# Configuration

Dream bench utilizes pydantic classes to manage configuration of your benchmarking run. You can view an example of it [here](example_config.json)

## Dataset Config

| key        | explanation                                     |
|------------|-------------------------------------------------|
| path       | path to the webdataset.                         |
| batch_size | the batch size for your image generation model. |

## Wandb

| key     | explanation                |
|---------|----------------------------|
| entity  | wandb entity to use        |
| project | the wandb project name     |
| name    | (optional) name of the run |

## Evaluation
| key               | explanation                                                            |
|-------------------|------------------------------------------------------------------------|
| save_path         | Path to save predicted outputs too.                                    |
| metrics           | A list of metrics to use in your run ["FID", "Aesthetic", "ClipScore"] |
| device            | torch device ["cuda:?", "cpu"] to use for evaluation                   |
| clip_architecture | Clip architecture to use for evaluation ["ViT-L/14", "ViT-B/32"]       |
