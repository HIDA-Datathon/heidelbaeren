from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
import torchvision.transforms.functional as TF
import random
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

DEFAULT_TRANSFORM_ARGS = {"vflip_chance":0.5, "rotation_chance": 0.5, "rotation_angle": 30, "size_x": 256, "size_y": 256}
DATA_ROOT_PATH = "/home/ksquare/repositories/datathon/data/ufz_im_challenge/"


class NeuronSegmentationDataset(VisionDataset):

    @staticmethod
    def get_classes():
        features_legend = pd.read_csv(Path(DATA_ROOT_PATH) / 'features_legend.csv', skipinitialspace=True)
        return  list(features_legend["label"])
 
    def __init__(self, root, folds=[1, 2, 3, 4, 5], preprocessing=None, transform_args=DEFAULT_TRANSFORM_ARGS):
        super(NeuronSegmentationDataset, self).__init__(root)
        self.root = Path(root)
        # read in segmentation classes
        features_legend = pd.read_csv(self.root / 'features_legend.csv', skipinitialspace=True)
        self.classes = list(features_legend["label"])
        self.class_to_idx = {self.classes[i]: i + 1 for i in range(len(self.classes))}
        # read in split files
        split_names = []
        if folds is None:
            # this will be the testing set
            self.samples = list((self.root / "photos").iterdir())
            self.testing = True
        else:
            for fold in folds:
                with open(self.root / "dataset{}.txt".format(fold), "r") as file:
                    split_names.extend([name.strip() for name in file.readlines()])
            self.samples = [self.root / 'photos_annotated' / name for name in split_names]
            self.testing = False
        self.loader = default_loader
        assert transform_args.keys() == DEFAULT_TRANSFORM_ARGS.keys()
        self.transform_args = transform_args
        self.mean = preprocessing["mean"]
        self.std = preprocessing["std"]

    def my_transform(self, img, mask):

        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.int64)

        # rotate
        if random.random() < self.transform_args["rotation_chance"]:
            angle = random.randint(-1 * self.transform_args["rotation_angle"], self.transform_args["rotation_angle"])
            img = TF.rotate(img, angle)
            if mask is not None:
                mask = TF.rotate(mask, angle)

        # to Tensor
        img = TF.to_tensor(img)
        # normalize 
        img = TF.normalize(tensor=img, mean=self.mean, std=self.std)
        # flipping
        if random.random() < self.transform_args["vflip_chance"]:
            img = TF.vflip(img)
            if mask is not None:
                mask = TF.vflip(mask)
        
        # resize
        img = TF.resize(img, [self.transform_args["size_x"], self.transform_args["size_y"]] )
        if mask is not None:
            mask = TF.resize(mask, [self.transform_args["size_x"], self.transform_args["size_y"]] )
        return img, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = self.samples[index]
        img = self.loader(img_path)
        # shape (3 x 600 x 800)
        if self.testing is False:
            seg = self.loader(img_path.with_suffix(".png"))
            seg = np.asarray(seg)[:, :, 0]
            # seg(600, 800)
            masks = [(seg == v) for v in self.class_to_idx.values()]
            mask = np.stack(masks, axis=0)
            # mask = torch.as_tensor(mask, dtype=torch.int64)
            # shape ()0, 600, 800 
        else:
            mask=None
        
        # apply transforms
        img, mask = self.my_transform(img, mask)
        if mask is None:
            return img
        return img, mask
        

if __name__ == "__main__":
    dataset = NeuronSegmentationDataset(root=DATA_ROOT_PATH, preprocessing=None, transformer_args=DEFAULT_TRANSFORM_ARGS)
    img, seg = dataset[0]