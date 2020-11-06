from argparse import ArgumentParser
from dotenv import load_dotenv
from pathlib import Path
from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
# from torchvision import transforms/
import albumentations as albu

from HidaDataset import NeuronSegmentationDataset 

from hida.utils import get_os 


class HidaDataLoader(pl.LightningDataModule):


    @staticmethod
    def get_preprocessing(preprocessing_fn):
        """Construct preprocessing transform
        
        Args:
            preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose
        
        """
        
        _transform = [
            albu.Lambda(image=preprocessing_fn),
        ]
        return albu.Compose(_transform)
        
    def __init__(self,
        transform_args,
        preprocessing,
        data_dir: str = './',
        train_batch_size=8,
        val_batch_size=8,
        test_batch_size=8,
        num_workers=16
    ):

        super().__init__()
        self.data_dir = data_dir

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers=num_workers

        self.transform_args = transform_args
        self.preprocessing = preprocessing #HidaDataLoader.get_preprocessing(preprocessing)
        
        self.dims = (3, 600, 800)

    def prepare_data(self, *args, **kwargs):
        pass
    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_data = NeuronSegmentationDataset(self.data_dir, folds=[1,2,3,4], preprocessing=self.preprocessing, transform_args=self.transform_args)
            self.val_data = NeuronSegmentationDataset(self.data_dir, folds=[5], preprocessing=self.preprocessing, transform_args=self.transform_args)
            
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_data = NeuronSegmentationDataset(self.data_dir, folds=None, preprocessing=self.preprocessing, transform_args=self.transform_args)

    def train_dataloader(self):
        return DataLoader(self.train_data, pin_memory=True, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, pin_memory=True, batch_size=self.val_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, pin_memory=True, batch_size=self.test_batch_size, num_workers=self.num_workers)
        
    @staticmethod
    def add_data_module_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default=get_os('DATA_DIR', required=True),
                            help="Name of the experiment you are running.")
        parser.add_argument("--train_folds", nargs='*', default=get_os('TRAIN_FOLDS'),
                            help="Set name to which operation is applied to.")
        parser.add_argument("--train_batch_size", type=int, default=int(get_os('TRAIN_BATCH_SIZE', 8)),
                            help="Train batch size.")
        parser.add_argument("--val_folds", nargs='*', default=get_os('VAL_FOLDS'),
                            help="Set name to which operation is applied to.")
        parser.add_argument("--val_batch_size", type=int, default=int(get_os('VAL_BATCH_SIZE', 8)),
                            help="Eval batch size.")
        parser.add_argument("--test", nargs='*', default=get_os('TEST'),
                            help="Set name to which operation is applied to.")
        parser.add_argument("--test_batch_size", type=int, default=int(get_os('TEST_BATCH_SIZE', 8)),
                            help="Eval batch size.")
        parser.add_argument("--num_workers", type=int, default=int(get_os('NUM_WORKERS', 8)),
                            help="Number of CPU cores used for data loading.")

        return parser

if __name__ == "__main__":

    import segmentation_models_pytorch as smp
    file_dir = Path(__file__).parent
    load_dotenv(file_dir.parent / 'user_settings.env')

    parser = ArgumentParser()
    parser = HidaDataLoader.add_data_module_specific_args(parser)
    args = parser.parse_args()

    transform_args = {"vflip_chance":0.5, "rotation_chance": 0.5, "rotation_angle": 30, "size": 256}


    #preprocessing_fn = smp.encoders.get_preprocessing_fn('se_resnext50_32x4d', pretrained='imagenet')
    preprocessing_params = smp.encoders.get_preprocessing_params('se_resnext50_32x4d', pretrained='imagenet')

    dm = HidaDataLoader(
        transform_args=transform_args,
        preprocessing=preprocessing_params,
        data_dir=args.data_dir,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
    ) 

    dm.prepare_data()

    # splits/transforms
    dm.setup('fit')
    for batch in dm.train_dataloader():
        print(batch)
    # print('training')
    # batch = list(dm.val_dataloader())
    # print(batch[0])
    # print('validating') 

    # dm.setup('test')
    # for batch in dm.test_dataloader():
    #     print(batch) 