import pathlib
from abc import abstractmethod

import PIL
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from albumentations.core.composition import Compose
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from pre_post_processing import build_eval_transform, build_post_transform, build_training_transform


class ImageClassificationDataset(Dataset):
    def __init__(self, df, base_path,
                 transform: Compose,
                 file_column='file_name',
                 label_column='label_encoded'
                 ):
        self.df = df
        self.file_column = file_column
        self.label_column = label_column
        self.transform = transform
        self.base_path = pathlib.Path(base_path)
        self.targets = df[label_column].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        image = PIL.Image.open(self.base_path / self.df.iloc[ix][self.file_column])
        label = self.df.iloc[ix][self.label_column]
        if self.transform is not None:
            image = self.transform(image=np.array(image))['image']
        return image, label


class BaseClassificationDataModule(pl.LightningDataModule):

    def __init__(self,
                 path_to_labels_df: str,
                 base_path: str,
                 batch_size: int,
                 base_model_data: str,
                 image_size=224,
                 train_workers=4,
                 augmentation_strategy='hard_1',

                 ):
        super().__init__()
        self.path_to_labels_df = path_to_labels_df
        self.base_path = base_path
        self.size = image_size
        self.base_model_data = base_model_data
        self.augmentation_strategy = augmentation_strategy
        self.batch_size = batch_size
        self.train_workers = train_workers

        self.df = self._load_df_meta(path_to_labels_df)

        self.post_transform = build_post_transform(self.base_model_data)
        self.train_transform = build_training_transform(self.size, self.base_model_data, self.augmentation_strategy)
        self.val_transform = build_eval_transform(self.base_model_data, self.size)

        self.df_train, self.df_val = None, None

        self.prepare()
        self.build_datasets()
        self.save_hyperparameters()

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          num_workers=self.train_workers)

    def val_dataloader(self):
        valload = DataLoader(self.val_ds,
                             batch_size=self.batch_size,
                             shuffle=False,
                             num_workers=self.train_workers)
        return valload

    def build_datasets(self):
        self.train_ds = ImageClassificationDataset(self.df_train, self.base_path, self.train_transform)
        self.val_ds = ImageClassificationDataset(self.df_val, self.base_path, self.val_transform)

    def _load_df_meta(self, path_or_df):
        if not isinstance(path_or_df, pd.DataFrame):
            path_or_df = pd.read_pickle(self.path_to_labels_df)
        return path_or_df

    @abstractmethod
    def prepare(self) -> [pd.DataFrame, pd.DataFrame]:
        """
        While subclassing override this method to change the behaviour of the dataset.
        For instance you can reframe the problem, reduce the number of classes or whatever else you might need to do.
        """
        pass


class BaseDatasetAllClasses(BaseClassificationDataModule):
    """
    Create subclasses to create a dataset with differnet behaviour, split, classes etc.
    """

    def __str__(self):
        return 'Base'

    def prepare(self) -> None:
        self.encoder = LabelEncoder()

        self.df['label_encoded'] = self.encoder.fit_transform(self.df.label)

        self.df_val = self.df[self.df.is_val].reset_index()
        self.df_train = self.df[~(self.df.is_val)].reset_index()

    @property
    def num_classes(self) -> int:
        return 2
