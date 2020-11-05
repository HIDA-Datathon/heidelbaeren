import os
from argparse import ArgumentParser
from dotenv import load_dotenv
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision.datasets.mnist import MNIST
from torchvision import transforms

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

import segmentation_models_pytorch as smp

from hida.utils import get_os 
from hida.HidaDataLoader import HidaDataLoader
from hida.HidaDataset import NeuronSegmentationDataset

file_dir = Path(__file__).parent
load_dotenv(file_dir.parent / 'user_settings.env')


class HidaImageSegmentation(pl.LightningModule):

    def __init__(self, encoder, encoder_weights, classes, activation, learning_rate=1e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # create segmentation model with pretrained encoder
        # self.model = smp.FPN(
        #     encoder_name=encoder, 
        #     # encoder_weights=encoder_weights, 
        #     classes=len(classes), 
        #     # activation=activation,
        # )

        self.model = smp.Unet('resnet34', classes=len(classes))

        # self.model = smp.FPN(
        #     encoder_name=encoder, 
        #     encoder_weights=encoder_weights, 
        #     classes=len(classes), 
        #     activation=activation,
        # )

        self.loss = smp.utils.losses.DiceLoss()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def calc_step(self, batch, batch_idx, mode='train'):
        #x = x.reshape(8,3,600,800)
        x, y = batch
        prediction = self.model(x)
        if mode != 'test':
            loss = self.loss(prediction, y)
            self.log(f'{mode}_losseeee', loss, on_step=True)

            if batch_idx == 0:
                self.logger.experiment.add_image(f"{mode}_image", x[0, :, :, :], self.global_step + 1)
                self.logger.experiment.add_image(f"{mode}_label", prediction[0, :, :], self.global_step + 1)
                self.logger.experiment.add_image(f"{mode}_prediction", x[0, :, :], self.global_step + 1)
            return {'loss': loss }
        else:
            pass

    def calc_epoch_end(self, outputs, mode):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log(f'{mode}_loss', avg_loss, on_epoch=True)

        pass

    def training_step(self, batch, batch_idx):
        return self.calc_step(batch, batch_idx, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.calc_step(batch, batch_idx, mode='val')

    def test_step(self, batch, batch_idx):        
        return self.calc_step(batch, batch_idx, mode='test')

    def training_epoch_end(self, outputs):
        self.calc_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.calc_epoch_end(outputs, 'val')
    
    def test_epoch_end(self, outputs):
        self.calc_epoch_end(outputs, 'test')

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optim = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return {
            'optimizer': optim,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=10, min_lr=1.e-7, eps=1e-08, verbose=False),
            'monitor': 'val_loss'
        }
        # return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--encoder', type=str, default='se_resnext50_32x4d')
        parser.add_argument('--encoder_weights', type=str, default='imagenet')
        parser.add_argument('--activation', type=str, default='sigmoid')
        return parser

def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=get_os('LOG_DIR', required=True),
                        help="Name of the experiment you are running.")
    # Experiment
    parser.add_argument("--name", type=str, default=get_os('NAME', required=True),
                        help="Name of the experiment you are running.")
    parser.add_argument("--version", type=str, default=get_os('VERSION', None),
                        help="Name of the experiment you are running.")
    # Training specific
    parser.add_argument("--cuda", type=int, default=int(get_os('CUDA', 1)),
                        help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--gpus", type=int, default=[0],
                        help="GPUS where your pgoram should run")
    parser.add_argument("--val_check_interval", type=int, default=float(get_os('VAL_CHECK_INTERVAL', 1)),
                        help="number of images after which the training loss is check_val_every_n_epoch, default is 500")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=int(get_os('CHECK_VAL_EVERY_N_EPOCHS', 1)),
                        help="number of images after which the training loss is logged, default is 500")
    parser.add_argument("--save_top_k", type=int, default=int(get_os('SAVE_TOP_K', 0)),
                        help="random seed for training")
    parser.add_argument("--max_epochs", type=int, default=int(get_os('MAX_EPOCHS', 3)),
                        help="random seed for training")

    parser = HidaDataLoader.add_data_module_specific_args(parser)
    parser = HidaImageSegmentation.add_model_specific_args(parser)
    args = parser.parse_args()

    print(args)
    logger = TensorBoardLogger(args.log_dir, name=args.name, version=args.version)
    lr_logger = LearningRateMonitor(logging_interval='step')

    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(logger.log_dir, 'checkpoints'),
        save_top_k=args.save_top_k,
        verbose=True,
        monitor=f'val_loss',
        mode='min',
        prefix=f'{args.name}_{args.version}'
    )

    transform_args = {"vflip_chance":0.5, "rotation_chance": 0.5, "rotation_angle": 30, "normalize_mean": [0.485, 0.456, 0.406], "normalize_std": [0.229, 0.224, 0.225]}


    model = HidaImageSegmentation(**vars(args), classes=NeuronSegmentationDataset.get_classes())

    # preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
    preprocessing_params = smp.encoders.get_preprocessing_params(args.encoder, args.encoder_weights)

    dm = HidaDataLoader(
        transform_args=transform_args,
        preprocessing=preprocessing_params,
        data_dir=args.data_dir,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
    ) 
    
    trainer = pl.Trainer(
        #fast_dev_run=True,
        callbacks=[lr_logger],
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        checkpoint_callback=checkpoint_callback,
        gpus=args.gpus,
        logger=logger,
        max_epochs=args.max_epochs
        )

    trainer.fit(model, dm)

    # result = trainer.test(datamodule=dm)
    # print(result)


if __name__ == '__main__':
    cli_main()