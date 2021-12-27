import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger

from package_for_torchserve import load_model_from_config


def test_can_train_model_from_configs():
    neptune_logger = NeptuneLogger(
        offline_mode=True

    )
    config = '../experiment_configs/base.yaml'
    model, data = load_model_from_config(config)
    trainer = pl.Trainer(max_steps=10, logger=neptune_logger, max_epochs=1)
    trainer.fit(model, data)
