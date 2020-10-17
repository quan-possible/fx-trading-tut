import numpy as np
import pandas as pd
import datetime
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from sklearn.model_selection import train_test_split

class ArrayDataset(Dataset):
    def __init__(self, datasets):
        super(ArrayDataset, self).__init__()
        self._length = len(datasets[0])
        for i, data in enumerate(datasets):
            assert len(data) == self._length, \
                "All arrays must have the same length; \
                array[0] has length %d while array[%d] has length %d." \
                % (self._length, i+1, len(data))
        self.datasets = datasets

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return tuple(torch.from_numpy(data[idx]).float() \
                     for data in self.datasets)

class FXDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, length, source_len, target_len, step):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.length = length
        self.source_len = source_len
        self.target_len = target_len
        self.step = step

    def split_sequence(self, source, target, source_len, target_len, step, target_start_next):
        """ Split sequence with sliding window into
            sequences of context features and target.
            Args:
                source (np.array): Source sequence
                target (np.array): Target sequence
                source_len (int): Length of input sequence.
                target_len (int): Length of target sequence.
                target_start_next (bool): If True, target sequence
                        starts on the next time step of last step of source
                        sequence. If False, target sequence starts at the
                        same time step of source sequence.
            Return:
                X (np.array): sequence of features
                y (np.array): sequence of targets
        """
        assert len(source) == len(target), \
                'Source sequence and target sequence should have the same length.'

        X, y = list(), list()
        if not target_start_next:
            target = np.vstack((np.zeros(target.shape[1], dtype=target.dtype), target))
        for i in range(0, len(source), step):
            # Find the end of this pattern:
            src_end = i + source_len
            tgt_end = src_end + target_len
            # Check if beyond the length of sequence:
            if tgt_end > len(target):
                break
            # Split sequences:
            X.append(source[i:src_end, :])
            y.append(target[src_end:tgt_end, :])
        return np.array(X), np.array(y)

    def prepare_data(self):
        df = pd.read_csv(self.data_dir, parse_dates=['DATE_TIME'])
        self.data = df.iloc[:,1:].values
        self.src, self.tgt = self.split_sequence(
                self.data,
                self.data,
                self.source_len,
                self.target_len,
                self.step,
                True
        )

    def setup(self):
        # Split data into training set and test set :
        test_idx = int(len(self.src) * 0.7)
        src_train, src_test, tgt_train, tgt_test \
            = self.src[:test_idx], self.src[test_idx:], self.tgt[:test_idx], self.tgt[test_idx:]
        # Split training data into train set and validation set:
        src_train, src_val, tgt_train, tgt_val \
            = train_test_split(src_train, tgt_train, test_size=0.25, random_state=1)
        # Prepare datasets
        self.trainset = ArrayDataset([src_train, tgt_train])
        self.valset = ArrayDataset([src_val, tgt_val])
        self.testset = ArrayDataset([src_test, tgt_test])

    def train_dataloader(self):
        self.trainloader = DataLoader(
                self.trainset,
                batch_size=self.batch_size,
                shuffle=True
        )
        return self.trainloader

    def val_dataloader(self):
        self.valloader = DataLoader(
                self.valset,
                batch_size=self.batch_size,
                shuffle=False
        )
        return self.valloader

    def test_dataloader(self):
        self.testloader = DataLoader(
                self.testset,
                batch_size=self.batch_size,
                shuffle=False
        )
        return self.testloader


"""class FXModule(pl.LightningModule):
    def __init__()
    def forward()
    def configure_optimizers()
    def training_step()
    def validation_step()
    def return():
        return array()"""


if __name__ == "__main__":
    from argparse import ArgumentParser

    # Parse arguments:
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--cuda", default=False)

    """
    args = parser.parse_args()

    # Model & data module:
    detector = PropagandaDetector(hparams=args)
    prop_dm = PropagandaDataModule()

    # Train & valid:
    trainer = pl.Trainer.from_argparse_args(args, fast_dev_run=True)
    trainer.fit(detector, prop_dm)
    """

    fx_dm = FXDataModule(
            data_dir="forex15m/EURUSD-2000-2020-15m.csv",
            batch_size=256,
            length=100,
            source_len=192,
            target_len=4,
            step=4
    )
    
    fx_dm.prepare_data()
    fx_dm.setup()
    cac, lon = next(iter(fx_dm.train_dataloader()))
    print(cac.shape, lon.shape)





