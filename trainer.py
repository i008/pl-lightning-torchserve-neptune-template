import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.utilities.cli import LightningCLI

NEPTUNE_KEY = os.environ.get('NEPTUNE_TOKEN')
NEPTUNE_PROJECT = 'i008/demo'


class CLI(LightningCLI):

    def before_fit(self) -> None:
        if not isinstance(self.trainer.logger, DummyLogger):
            self.trainer.logger.log_text("dataset_type", str(self.datamodule))
            self.trainer.logger.log_artifact(self.datamodule.path_to_labels_df)
            self.trainer.logger.log_text("classes", str(self.datamodule.encoder.classes_))
            self.trainer.logger.log_artifact(self.config['config'][0].abs_path, 'config.yaml')


if __name__ == '__main__':
    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_KEY,
        project_name=NEPTUNE_PROJECT,
        upload_source_files=["**/*.py", "**/*.ipynb"],
        experiment_name='/home/i008/demo_exps'
    )

    callbacks = [LearningRateMonitor()]

    train_defaults = {

        'logger': neptune_logger,
        'callbacks': callbacks,
        'devices': [0],
        'accelerator': 'gpu',
        # 'auto_scale_batch_size': True,
        # 'auto_lr_find': True,
        'precision': 16

    }

    cli = CLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        trainer_defaults=train_defaults,
        save_config_overwrite=True,
        save_config_callback=None,
        seed_everything_default=1245,

    )
