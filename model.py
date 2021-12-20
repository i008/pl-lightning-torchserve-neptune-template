from typing import Any, Optional

import efficientnet_pytorch
import numpy as np
import pandas as pd
import pretrainedmodels
import pytorch_lightning as pl
import torch
from efficientnet_pytorch import EfficientNet
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim.lr_scheduler import CyclicLR, StepLR
from torchmetrics import Accuracy, F1
from transformers import ViTModel


def build_classification_head(in_features,
                              out_features,
                              strategy='standard'):
    if strategy == 'fastai':
        return fastai_classification_head(in_features, out_features)

    else:
        return nn.Linear(in_features, out_features)


def fastai_classification_head(in_features, out_features):
    """
    Improved classification head
    """
    return nn.Sequential(
        nn.BatchNorm1d(in_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.25, inplace=False),
        nn.Linear(in_features=in_features, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=512, out_features=out_features, bias=True),
    )


def calculate_clf_metric(y_true, y_pred):
    clf_report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return clf_report, cm


def _make_trainable(module):
    """Unfreeze a given module.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module, train_bn=True):
    """Freeze the layers of a given module.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
        Otherwise, they will be set to eval mode along with the other modules.
    """

    BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)

    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module, n=-1, train_bn=True):
    """Freeze the layers up to index n.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    n : int
        By default, all the layers will be frozen. Otherwise, an integer
        between 0 and `len(module.children())` must be given.
    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
    """
    idx = 0
    children = list(module.children())
    n_max = len(children) if n == -1 else int(n)
    for child in children:
        if idx < n_max:
            _recursive_freeze(module=child, train_bn=train_bn)
        else:
            _make_trainable(module=child)


def torch2np(results_dict: dict):
    for k, v in results_dict.items():
        results_dict[k] = v.detach().cpu().numpy()
    return results_dict


class AdaptiveConcatPooler(nn.Module):
    def __init__(self, how='both', flatten=True):
        super().__init__()
        self.how = how
        self.flatten = flatten
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        if self.how == 'both':
            avg = self.avg_pool(x)
            mx = self.max_pool(x)
            feats = torch.cat([avg, mx], axis=1)
        elif self.how == 'avg':
            feats = self.avg_pool(x)
        elif self.how == 'max':
            feats = self.max_pool(x)
        if self.flatten:
            feats = feats.flatten(start_dim=1)
        return feats


class UnifiedBackboneBuilder(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.normalization = 'standard'
        self.is_pretrained_models = False

        if model_name in pretrainedmodels.__dict__:
            self.is_pretrained_models = True  # coming from pretraeindmodels package
            if pretrained:
                pretrained = 'imagenet'
            else:
                pretrained = None
            self.backbone = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
            self.num_feature_planes = self.backbone.last_linear.in_features

        elif 'efficientnet' in model_name:
            if self.pretrained:
                self.backbone = EfficientNet.from_pretrained(model_name)
                self.num_feature_planes = self.backbone._fc.in_features

            else:
                raise NotImplemented("Only pretrained efficientnet supported")

        elif 'vit' in model_name:
            self.backbone = ViTModel.from_pretrained(model_name)
            self.normalization = 'vit'
            self.num_feature_planes = 768

    def features_forward(self, x):
        if isinstance(self.backbone, efficientnet_pytorch.model.EfficientNet):
            return self.backbone.extract_features(x)
        elif 'vit' in self.model_name:
            output = self.backbone(pixel_values=x)
            output = output.last_hidden_state[:, 0, :]
            output = output[:, :, None, None]  # add 2 dimensiosn at the end to be coherent with other models before GAP
            return output

        else:
            return self.backbone.features(x)


class PlClassificationModel(pl.LightningModule):
    def __init__(self,
                 base_model: str,
                 pooling: str,
                 backbone_lr: float,
                 classifier_lr: float,
                 num_classes: int,
                 optimizing_strategy: str,
                 clf_head: str = 'standard',  # or fastai,
                 label_smoothing=0.0,
                 checkpoint_monitor: str = 'val/f1_macro',
                 backbone_pretrained: bool = True,  # or None
                 freeze_backbone=False,
                 upload_best=True,
                 normalize_pool: bool = False):

        super().__init__()
        self.base_model = base_model
        self.pooling = pooling
        self.normalize_pool = normalize_pool
        self.num_classes = num_classes
        self.backbone_pretrained = backbone_pretrained
        self.clf_head = clf_head
        self.backbone_lr = backbone_lr
        self.classifier_lr = classifier_lr
        self.checkpoint_monitor = checkpoint_monitor
        self.optimizing_strategy = optimizing_strategy
        self.label_smoothing = label_smoothing
        self.freeze_backbone = freeze_backbone
        self.upload_best = True

        self.last_best_model = ''

        self.save_hyperparameters()

        self.backbone = UnifiedBackboneBuilder(base_model, pretrained=backbone_pretrained)  # pretrainedmodels.__dict__[base_model](pretrained=backbone_pretrained)

        if self.freeze_backbone:
            freeze(self.backbone)

        self.pooler = AdaptiveConcatPooler(how=pooling)
        self.num_feature_planes = self.backbone.num_feature_planes
        if self.pooling == 'both':
            self.num_feature_planes = int(2 * self.num_feature_planes)

        self.classifier = build_classification_head(
            self.num_feature_planes,
            self.num_classes,
            strategy=self.clf_head
        )
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.f1_val = F1(num_classes=self.num_classes, average='weighted')
        self.f1_train = F1(num_classes=self.num_classes, average='weighted')
        self.f1_val_macro = F1(num_classes=self.num_classes, average='macro')
        self.f1_train_macro = F1(num_classes=self.num_classes, average='macro')

        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def configure_callbacks(self):
        return [ModelCheckpoint(save_top_k=3,
                                monitor=self.checkpoint_monitor,
                                mode='max')]

    def forward(self, x):
        features = self.backbone.features_forward(x)
        pooled_features = self.pooler(features)
        logits = self.classifier(pooled_features)

        return {'logits': logits}

    def calc_loss(self, labels, logits):
        losses = {}
        losses['clf_loss'] = self.classification_loss(logits, labels)

        return losses

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        outputs = self(images)
        outputs['labels'] = labels

        self.log("val/acc", self.val_accuracy(outputs['logits'], labels))
        self.log("val/f1_avg", self.f1_val(outputs['logits'], labels))
        self.log("val/f1_macro", self.f1_val_macro(outputs['logits'], labels))

        return torch2np(outputs)

    def training_step(self, batch, batch_idx):

        images, labels = batch
        outputs = self(images)
        labels = labels.long()
        outputs['labels'] = labels

        loss_dict = self.calc_loss(labels, outputs['logits'])
        total_loss = sum(loss_dict.values())
        for k, v in loss_dict.items():
            self.log(k, v)

        self.log("train/acc", self.train_accuracy(outputs['logits'], labels))
        self.log("train/f1_avg", self.f1_train(outputs['logits'], labels))
        self.log('train/f1_macro', self.f1_train_macro(outputs['logits'], labels))

        return {'loss': total_loss, 'outputs': torch2np(outputs)}

    def __upload_checkpoints(self):
        if self.upload_best:
            current_best = self.trainer.callbacks[-1].best_model_path
            if not self.last_best_model and current_best:
                self.last_best_model = current_best

            if self.last_best_model != current_best:
                print(current_best)
                self.logger.log_artifact(current_best, 'best.pt')
                self.last_best_model = current_best

    def __process_output_preds(self, outputs):
        logits = np.concatenate([o['logits'] for o in outputs])
        y_true = np.concatenate([o['labels'] for o in outputs])
        preds = logits.argmax(axis=1)

        preds_labels = self.trainer.datamodule.encoder.inverse_transform(preds)
        labels = self.trainer.datamodule.encoder.inverse_transform(y_true)

        return preds_labels, labels

    def __clf_report(self, preds_labels, labels):
        x = [v for k, v in classification_report(preds_labels, labels, output_dict=True).items()]
        ks = [k for k, v in classification_report(preds_labels, labels, output_dict=True).items()]

        df_res = pd.DataFrame(x[:-3])
        df_res['cls'] = ks[:-3]
        df_res = df_res.sort_values('f1-score')

        with open(f'temp/{self.current_epoch}_scores.html', 'w') as fo:
            fo.write(df_res.to_html())
            self.logger.log_artifact(f'temp/{self.current_epoch}_scores.html')

    def validation_epoch_end(self, outputs) -> None:

        preds_labels, labels = self.__process_output_preds(outputs)
        self.__clf_report(preds_labels, labels)
        self.__upload_checkpoints()

    def configure_optimizers(self):

        if self.optimizing_strategy == 'regular1':
            params = [
                {'params': self.backbone.parameters(), 'lr': self.backbone_lr},
                {'params': self.classifier.parameters(), 'lr': self.classifier_lr},
            ]
            optimizer = torch.optim.AdamW(params)

            return [optimizer], [StepLR(optimizer, step_size=30, gamma=0.1)]

        if self.optimizing_strategy == 'cycling':
            params = [{'params': self.backbone.parameters(),
                       'lr': self.backbone_lr,
                       'weight_decay': 0.0001},
                      {'params': self.classifier.parameters(),
                       'lr': self.classifier_lr,
                       'weight_decay': 0.0001}
                      ]
            optimizer = torch.optim.SGD(params=params, lr=self.backbone_lr)
            scheduler = CyclicLR(optimizer, base_lr=self.backbone_lr, max_lr=0.1, step_size_up=3000,
                                 step_size_down=3000)

            return [optimizer], [{'scheduler': scheduler,
                                  'interval': 'step', 'frequency': 1}]

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        if len(batch) >= 2:
            batch = batch[0]
        outputs = self(batch)
        return torch2np(outputs)
