# Python Import:
import os
import shutil
import joblib
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from argparse import ArgumentParser
# Pytorch Import:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
# Pytorch Lightning Import:
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

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
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Args:
                data_dir (str): Data directory.
                batch_size (int): Bacth size.
                source_len (int): Length of input sequence.
                target_len (int): Length of target sequence.
                step (int): Window size.
                test_size (float): Percentage of test dataset.
                val_size (float): Percentage of valid dataset (exclude test dataset).
                num_workers (int): Number of workers for data loading.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--source_len", type=int, default=192)
        parser.add_argument("--target_len", type=int, default=32)
        parser.add_argument("--step", type=int, default=1)
        parser.add_argument("--test_size", type=float, default=0.3)
        parser.add_argument("--val_size", type=float, default=0.25)
        parser.add_argument("--num_workers", type=int, default=8)
        return parser    
    
    def __init__(self, hparams):
        super(FXDataModule, self).__init__()
        self.hparams = hparams
        self.scaler = MinMaxScaler()

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
        df = pd.read_csv(self.hparams.data_dir)
        self.data = df.iloc[:,1:].values
        self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)
        self.src, self.tgt = self.split_sequence(
            self.data,
            self.data,
            self.hparams.source_len,
            self.hparams.target_len,
            self.hparams.step,
            True
        )

    def setup(self, stage=None):
        # Split data into training set and test set :
        test_idx = int(len(self.src) * self.hparams.test_size)
        src_train, src_test, tgt_train, tgt_test \
            = self.src[:test_idx], self.src[test_idx:], self.tgt[:test_idx], self.tgt[test_idx:]
        # Split training data into train set and validation set:
        src_train, src_val, tgt_train, tgt_val \
            = train_test_split(src_train, tgt_train, test_size=self.hparams.val_size)
        # Prepare datasets
        self.trainset = ArrayDataset([src_train, tgt_train])
        self.valset = ArrayDataset([src_val, tgt_val])
        self.testset = ArrayDataset([src_test, tgt_test])

    def train_dataloader(self):
        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_loader = DataLoader(
            self.valset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(
            self.testset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
        return self.test_loader
    
class FXModule(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Args:
                source_size (int): The expected number of features in the input.
                target_size (int): The expected number of sequence features.
                hidden_size (int): The number of features in the hidden state.
                num_layers (int): Number of recurrent layers.
                bidirectional (boolean): whether to use bidirectional model.
                dropout (float): dropout probability.
                lr (float): Learning rate.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--source_size", type=int, default=4)
        parser.add_argument("--target_size", type=int, default=4)
        parser.add_argument("--hidden_size", type=int, default=256)
        parser.add_argument("--num_layers", type=int, default=1)
        parser.add_argument("--bidirectional", type=bool, default=False)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser
        
    def __init__(self, hparams):
        super(FXModule, self).__init__()
        self.hparams = hparams
        # Encoder:
        num_directions = 2 if self.hparams.bidirectional else 1
        self.en_gru = nn.GRU(
            self.hparams.source_size,
            self.hparams.hidden_size,
            self.hparams.num_layers,
            dropout=self.hparams.dropout if self.hparams.num_layers else 0,
            bidirectional=self.hparams.bidirectional,
            batch_first=True
        )
        self.en_fc = nn.Linear(
            self.hparams.num_layers*num_directions, 
            self.hparams.num_layers
        )
        
        # Decoder:
        self.de_gru = nn.GRU(
            self.hparams.target_size,
            self.hparams.hidden_size,
            self.hparams.num_layers,
            dropout=self.hparams.dropout if self.hparams.num_layers else 0,
            batch_first=True
        )
        self.de_fc = nn.Linear(
            self.hparams.hidden_size, 
            self.hparams.target_size
        )
        
    def encode(self, input, hidden=None):
        """ Args:
                input (batch, seq_len, source_size): Input sequence.
                hidden (num_layers*num_directions, batch, hidden_size): Initial states.
                
            Returns:
                output (batch, seq_len, num_directions*hidden_size): Outputs at every step.
                hidden (num_layers, batch, hidden_size): Final state.
        """
        # Feed source sequences into GRU:
        outputs, hidden = self.en_gru(input, hidden)
        # Compress bidirection to one direction for decoder:
        hidden = hidden.permute(1, 2, 0)
        hidden = self.en_fc(hidden)
        hidden = hidden.permute(2, 0, 1)
        return outputs, hidden.contiguous()
    
    def forward(self, hidden, pred_len=32, target=None, teacher_forcing=0.0):
        """ Args:
                hidden (num_layers, batch, hidden_size): States of the GRU.
                pred_len (int): Length of predicted sequence.
                target (batch, seq_len, target_size): Target sequence. If None,
                    the output sequence is generated by feeding the output
                    of the previous timestep (teacher_forcing has to be False).
                teacher_forcing (float): Probability to apply teacher forcing.
                
            Returns:
                outputs (batch, seq_len, target_size): Tensor of log-probabilities
                    of words in the target language.
                hidden of shape (1, batch_size, hidden_size): New states of the GRU.
        """
        if target is None:
            assert not teacher_forcing, 'Cannot use teacher forcing without a target sequence.'

        # Determine constants:
        batch = hidden.shape[1]
        # Initial value to feed to the GRU:
        val = torch.zeros((batch, 1, self.hparams.target_size), device=hidden.device)
        if target is not None:
            target = torch.cat([val, target[:, :-1, :]], dim=1)
            pred_len = target.shape[1]
        # Sequence to record the predicted values:
        outputs = list()
        for i in range(pred_len):
            # Embed the value at ith time step:
            # If teacher_forcing then use the target value at current step
            # Else use the predicted value at previous step:
            val = target[:, i:i+1, :] if (np.random.rand() < teacher_forcing) else val
            # Feed the previous value and the hidden to the network:
            output, hidden = self.de_gru(val, hidden)
            # Predict new output:
            val = self.de_fc(output.relu()).sigmoid()
            # Record the predicted value:
            outputs.append(val)
        # Concatenate predicted values:
        outputs = torch.cat(outputs, dim=1)
        return outputs, hidden
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=1e-1, patience=2, verbose=True)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        _, h = self.encode(x)
        y_hat, _ = self(h, None, y, 1.0)
        loss = F.mse_loss(y_hat, y)
        return {"loss": loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_loss": avg_loss}
        return {"log": tensorboard_logs}
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, h = self.encode(x)
        y_hat, _ = self(h, None, y, 0.0)
        val_loss = F.mse_loss(y_hat, y)
        return {'val_loss': val_loss}
    
    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss": val_loss}
        return {"val_loss": val_loss, "log": tensorboard_logs}
    
if __name__ == "__main__":
    # Argument parser:
    parser = ArgumentParser()
    parser = FXDataModule.add_model_specific_args(parser)
    parser = FXModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    # Model & data module:
    fx_dm = FXDataModule(args)
    fx_model = FXModule(args)
    
    # Callbacks:
    checkpoint = ModelCheckpoint(
        filepath="./checkpoint/fx-{epoch:02d}-{val_loss:.2f}",
        monitor="avg_val_loss"
    )
    
    # Trainer:
    trainer = pl.Trainer.from_argparse_args(
        args, 
        checkpoint_callback=checkpoint
    )
    trainer.fit(fx_model, fx_dm)

