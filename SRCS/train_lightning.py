import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning import Trainer
import pytorch_lightning as pl

import random
import numpy as np
from argparse import ArgumentParser

from pytorchtools import EarlyStopping
from plot import plot_learningCurve
from utils import isExists

class LightningLSTM(pl.LightningModule):
    def __init__(self, args, input_size, scaler=None, output_size=1):
        super(LightningLSTM, self).__init__()
        self.args      = args
        self.input_size= input_size
        self.output_size=output_size
        ### HyperParameter
        self.hidden    = args.hidden_size
        self.nlayer    = args.num_layer
        self.lr        = args.lr
        self.epochs    = args.epochs
        self.WSize     = args.windowSize
        self.nearK     = args.nearestK
        ### Model Architecture
        self.lstm    = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden,
                               num_layers=self.nlayer, batch_first=True)
        self.fc      = nn.Linear(self.hidden, self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.criterion= nn.MSELoss()
        self.scaler   = scaler
        ### For Output
        self.pred    = []
        self.dlst    = []

    def custom_histogram_adder(self):
        # interating thorough all parameters
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def forward(self, x):
        # Forward propagate LSTM
        out, _ = self.lstm(x) # out : tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:,-1,:])
        out = self.sigmoid(out)
        return out

    def on_train_start(self):
        #for k,v in vars(self.args).items():
        self.logger.log_hyperparams(params=vars(self.args),metrics=dict(val_loss=0, train_loss=0))  # logging hyper-parameter

    def training_step(self, batch, batch_nb):  # Required
        x, y, d = batch
        y_hat = self.forward(x)
        if self.scaler is not None: 
            y_hat = self.scaler.rev_transform(y_hat)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.log('loss', loss)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        if self.current_epoch == 1:
            sample_x = torch.rand((1,self.WSize, self.nearK * 3))
            self.logger.experiment.add_graph(LightningLSTM(self.args, self.input_size, self.output_size), sample_x)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss)
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)

    def validation_step(self, batch, batch_nb): # Optional
        x, y, d = batch
        y_hat = self.forward(x)
        if self.scaler is not None: 
            y_hat = self.scaler.rev_transform(y_hat)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):         # Optional
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)

    def configure_optimizers(self):            # Required
        #ds_len = self.train_dataset.__len__()
        #total_bs = self.batch_size*self.gpus
        #self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr * 5,
        #                    steps_per_epoch=int((ds_len/total_bs)), epochs=self.args.epochs)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser],add_help=False)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--hidden_size', default=32, type=int)
        parser.add_argument('--num_layer', default=3, type=int)
        # training specific (for this model)
        parser.add_argument('--epochs', default=200, type=int)

        return parser

def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
