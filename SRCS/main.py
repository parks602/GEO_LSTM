import numpy as np
import deepspeed
import sys
import torch.backends.cudnn as cudnn
import torch
from utils import get_argument_parser, set_seed, create_folder
from data import MyDataModule, getTestDataset, MinMaxScaler
from train_lightning import LightningLSTM, weights_update

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, TestTubeLogger

def main():
    parser = get_argument_parser()
    parser = MyDataModule.add_model_specific_args(parser)
    parser = LightningLSTM.add_model_specific_args(parser)
    args   = parser.parse_args()
    assert args.split_frac >=0 and args.split_frac <=1, "Invalid training set fraction"
    create_folder(args.opath)
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(args.manualSeed)
    # Logger 
    logger = TensorBoardLogger(args.opath, name='tensorboard')
    #logger2 = TestTubeLogger(args.opath, name='test_tube')

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping( monitor='val_loss',
                                         min_delta=0.00,
                                         patience=args.patience,
                                         verbose=False,
                                         mode='min')
    checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=args.opath,
            filename="sample_{epoch:03d}_{vloss:.2f}",
            save_top_k=1,
            mode='min',
    )
    # Data
    scaler= MinMaxScaler(args.var)
    model = LightningLSTM(args, args.nearestK * 3, scaler)
    # preprocess & train
    if args.mode == "train":
        # Dataset
        dm = MyDataModule(args, scaler)
        dm.prepare_data()
        dm.setup()
        # Train
        trainer = Trainer(callbacks=[early_stop_callback, lr_monitor,checkpoint_callback],
                          logger=logger,
                          check_val_every_n_epoch=3,
                          log_every_n_steps=10.,
                          log_gpu_memory='min_max',
                          max_epochs=args.epochs,
                          gpus=1,
                          accumulate_grad_batches=8,
                          precision=16)
        trainer.fit(model, datamodule=dm)
    else:
        # Dataset
        x, d = getTestDataset(args, scaler)
        # Load Trained Model
        pretrained_model = weights_update(model, torch.load("%s" %(args.model)))
        pretrained_model.eval()
        # Predict
        test_x = torch.rand((1,8,30))
        y_pred = model(x)
        print(y_pred.shape)

if __name__ == "__main__":
    main()
