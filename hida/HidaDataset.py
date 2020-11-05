from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
import torchvision.transforms.functional as TF
import random
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

DEFAULT_TRANSFORM_ARGS = {"vflip_chance":0.5, "rotation_chance": 0.5, "rotation_angle": 30, "normalize_mean": [0.485, 0.456, 0.406], "normalize_std": [0.229, 0.224, 0.225]}
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

    def my_transform(self, img, seg):
        # flipping
        if random.random() < self.transform_args["vflip_chance"]:
            img = TF.vflip(img)
            seg = TF.vflip(seg)
        # rotate
        if random.random() < self.transform_args["rotation_chance"]:
            angle = random.randint(-1 * self.transform_args["rotation_angle"], self.transform_args["rotation_angle"])
            img = TF.rotate(img, angle)
            seg = TF.rotate(seg, angle)

        img = TF.resize(img, [1024, 1024] )
        seg = TF.resize(img, [1024, 1024] )
        return img, seg

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = self.samples[index]
        img = self.loader(img_path)
        # shape (3 x 600 x 800)
        if self.testing is False:
            seg = self.loader(img_path.with_suffix(".png"))
            #seg(600,800,3)
        else:
            seg = torch.zeros_like(img)[0].unsqueeze(2)
            #shape(600.,800,1)

        # apply augmentations
        #img = torch.from_numpy(TF.to_tensor(img).numpy())
        img, seg = self.my_transform(img, seg)
        seg = np.asarray(seg)
        seg = seg[:,:,0]
        #seg(600,,800)

        masks = [(seg == v) for v in self.class_to_idx.values()]
        mask = np.stack(masks, axis=-1).astype('float')
        #seg(600,,800,20)

        
        # apply preprocessing
        if self.preprocessing:
            img = np.asarray(img)
            sample = self.preprocessing(image=img, mask=mask)
            image, mask = sample['image'], sample['mask']
            img = torch.as_tensor(img, dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.int64)
        else:
            # to Tensor
            img = torch.as_tensor(img, dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.int64)
            # normalize 
            img = TF.normalize(tensor=img, mean=self.transform_args["normalize_mean"], std=self.transform_args["normalize_std"])
        
        img = img.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)
        if self.testing:
            return img
        return img, mask
        

if __name__ == "__main__":
    dataset = NeuronSegmentationDataset(root=DATA_ROOT_PATH, preprocessing=None, transformer_args=DEFAULT_TRANSFORM_ARGS)
    img, seg = dataset[0]