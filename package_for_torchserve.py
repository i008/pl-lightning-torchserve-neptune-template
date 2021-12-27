import argparse
import json
import os
import pathlib
import tempfile
from argparse import Namespace
from typing import Optional

import neptune.new as neptune
import pytorch_lightning as pl
import torch
from jsonargparse import ArgumentParser
from model_archiver.model_packaging import package_model
from model_archiver.model_packaging_utils import ModelExportUtils
from torch.hub import load_state_dict_from_url

DEVICE = 'cpu'
NEPTUNE_KEY = os.environ.get('NEPTUNE_TOKEN')
TRACE_INPUT = torch.randn(1, 3, 224, 224)


def load_model_from_config(config_path: str, checkpoint_path: str = None):
    """

    Given pl.Lightning yaml file and optional checkpoint returns the loaded model and data (if present)

    :param config_path:
    :param checkpoint_path:
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument('--model', type=pl.LightningModule)
    parser.add_argument('--data', type=pl.LightningDataModule)
    config = parser.parse_path(config_path)
    cls = parser.instantiate_classes(config)
    model, data = cls['model'], cls['data']
    if checkpoint_path:
        if 'http' in checkpoint_path:
            state_dict = load_state_dict_from_url(checkpoint_path, map_location=DEVICE)['state_dict']
        else:
            state_dict = torch.load(checkpoint_path, map_location=DEVICE)['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
    return model, data


def package_pl_model(project_name: str,
                     pl_checkpoints_path: str,
                     neptune_experiment_name: str,
                     model_store_path: str,
                     checkpoint_name):
    """

    Packages a pytorch_lightning model from a Neptune experiment

    :param project_name: Neptune project name.
    :param pl_checkpoints_path: Checkpoint paths (where checkpoints are stored)
    :param neptune_experiment_name: Neptune Experiment name
    :param checkpoint_name: Exact checkpoint file name
    :param model_store_path: Where the packaged model will be stored
    :return:
    """
    pathlib.Path(model_store_path).mkdir(exist_ok=True)
    run = neptune.init(project_name, run=neptune_experiment_name, api_token=NEPTUNE_KEY)

    with tempfile.TemporaryDirectory() as td:
        run['artifacts/config.yaml'].download(td)

        serialized_file = f'/{td}/{neptune_experiment_name}.ts'

        if checkpoint_name:
            full_checkpoint_path = f'{pl_checkpoints_path}/{neptune_experiment_name}/checkpoints/{checkpoint_name}'
        else:
            run['artifacts/best.pt'].download(td)
            full_checkpoint_path = f'{td}/best.pt'

        model, data = load_model_from_config(f'{td}/config.yaml', full_checkpoint_path)
        model.to_torchscript(serialized_file,
                             example_inputs=TRACE_INPUT,
                             method='trace',
                             strict=False)

        args = Namespace(
            **{
                'model_file': None,
                'serialized_file': serialized_file,
                'handler': 'torchserve_custom_handler.py',
                'model_name': neptune_experiment_name,
                'version': 'v2',
                'export_path': model_store_path,
                'force': True,
                'extra_files': 'pre_post_processing.py',
                'runtime': 'python',
                'archive_format': 'default',
                'requirements_file': None,
            })

        manifest = ModelExportUtils.generate_manifest_json(args)
        manifest = json.loads(manifest)
        manifest['base_model'] = model.hparams.base_model
        manifest = json.dumps(manifest)
        package_model(args, manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default='i008/demo')
    parser.add_argument('--experiment_name')
    parser.add_argument('--pl_checkpoints_path', required=False)
    parser.add_argument('--checkpoint_name',  required=False)
    parser.add_argument('--model_store_path', required=True)

    args = parser.parse_args()

    package_pl_model(args.project_name,
                     args.pl_checkpoints_path,
                     args.experiment_name,
                     args.model_store_path,
                     args.checkpoint_name,
                     )
