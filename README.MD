### setup

```bash
export NEPTUNE_TOKEN
export NEPTUNE_PROJECT
```


### Train a model

```bash

python trainer.py --config experiment_configs/base.yaml --trainer.max_epochs 10  --trainer.limit_train_batches 10

```


