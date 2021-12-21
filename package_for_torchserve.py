import argparse
import json
import os
import pathlib
import tempfile
from argparse import Namespace

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


def load_model_from_config(config_path, checkpoint_path: str):
    parser = ArgumentParser()
    parser.add_argument('--model', type=pl.LightningModule)
    parser.add_argument('--data', type=pl.LightningDataModule)
    config = parser.parse_path(config_path)
    cls = parser.instantiate_classes(config)
    model, data = cls['model'], cls['data']
    if 'http' in checkpoint_path:
        state_dict = load_state_dict_from_url(checkpoint_path, map_location=DEVICE)['state_dict']
    else:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    return model, data


def package_pl_model(project_name, pl_checkpoints_path, experiment_name, checkpoint_name, model_store_path):
    pathlib.Path(model_store_path).mkdir(exist_ok=True)
    run = neptune.init(project_name, run=experiment_name, api_token=NEPTUNE_KEY)

    with tempfile.TemporaryDirectory() as td:
        run['artifacts/config.yaml'].download(td)

        serialized_file = f'/{td}/{experiment_name}.ts'

        if checkpoint_name:
            full_checkpoint_path = f'{pl_checkpoints_path}/{experiment_name}/checkpoints/{checkpoint_name}'
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
                'model_name': experiment_name,
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
        manifest['normalization'] = data.hparams.normalize
        manifest = json.dumps(manifest)
        package_model(args, manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default='i008/demo')
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--pl_checkpoints_path', type=str)
    parser.add_argument('--checkpoint_name', type=str)
    parser.add_argument('--model_store_path', type=str)

    args = parser.parse_args()

    package_pl_model(args.project_name,
                     args.pl_checkpoints_path,
                     args.experiment_name,
                     args.checkpoint_name,
                     args.model_store_path)
