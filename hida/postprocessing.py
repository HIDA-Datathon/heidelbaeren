import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from HidaDataset import DATA_ROOT_PATH

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


features_legend = pd.read_csv(Path(DATA_ROOT_PATH) / 'features_legend.csv', skipinitialspace=True)
classes = list(features_legend["label"])


def get_test_files(list_of_processed_folder_paths):
    # load test set file names
    file_names = [file.name for file in (Path(DATA_ROOT_PATH) / "photos").iterdir()]
    file_dic = {}
    for img_name in file_names:
        file_dic[img_name] = {}
    for output_folder in list_of_processed_folder_paths:
        output_folder = Path(output_folder)
        assert output_folder.exists()
        assert output_folder.is_dir()
        key = output_folder.name
        for img_name, resource_path in zip(file_names, sorted(output_folder.iterdir())):
            file_dic[img_name][key] = resource_path
    return file_dic


def ensemble(arr_list):
    if len(arr_list) == 1:
        return arr_list[0]
    else:
        assert all([arr.shape == arr_list[0].shape for arr in arr_list[1:]]
                   ), "array shapes do not match"
        raise NotImplementedError("ensembling not yet supported...")


def load_processed(path, mode):
    assert mode in ['img', 'prediction']
    return np.load(path)[mode]

def get_percentages(arr):
    # variables TODO TUNING
    confidence_threshold = 0.5
    min_percentage = 0.05
    orientation_hit_percentage = 0.7
    # utils
    percentages = {}
    channel_size = arr[0].size
    # iterate over segmentation classes 
    for ix, seg_class in enumerate(classes):
        # inspect prediction channel (corresponds to one segmentation class)
        channel = arr[ix]
        # calculate percentage within whole image
        perc = np.greater_equal(channel, confidence_threshold).sum() / channel_size
        if perc >= min_percentage:
            sides = np.split(channel, 2, axis=1)
            orients = [np.greater_equal(sides[split_ix], confidence_threshold).sum() / (channel_size * perc) >= orientation_hit_percentage for split_ix in range(2)]
            percentages[seg_class] = {'percentage': perc*100, 'left': orients[0], 'right': orients[1]}
    return percentages


def log_final_output(percentages, img_name, image, mask):
    fig = plt.figure()
    plt.imshow(image.transpose(1, 2, 0), interpolation='none') 

    t = ''
    key_matrix = np.zeros((256,256,len(percentages.items())))

    for mask_id, (key, value) in enumerate(percentages.items()):
        
        mask_idx = classes.index(key)
        plt.imshow(np.greater(mask[mask_idx], 0.5)*((mask_idx+1)*255/20), 'jet',alpha=0.3) 
        t = t + f'{key} ({int(value["percentage"])}%) '
        if value['left']:
            t = t + ': left; '
        elif value['right']:
            t = t + ': right; ' 
        

    plt.title(t)
    fig.subplots_adjust(top=0.8)
    plt.show()

    writer = SummaryWriter(f'results/datathon/results3')
    writer.add_figure(f'image: {img_name}', fig)
    # writer.close()

def do_postprocessing(fast_dev=True):
    # generating files overview
    processed_folders = ["/home/ksquare/repositories/datathon/experiments/logs/heidelbaer/final/images"]
    file_dic = get_test_files(processed_folders)
    if fast_dev:
        for key in list(file_dic.keys())[100:]:
            del file_dic[key]
    # iterate over image instances
    for img_key in file_dic.keys():
        all_preds = []
        # load preds
        for exp_key in file_dic[img_key].keys():
            all_preds.append(load_processed(file_dic[img_key][exp_key], mode='prediction'))
        # ensemble preds
        ensembled = ensemble(all_preds)
        # calculate percentages
        percentages = get_percentages(ensembled)
        # produce output
        image = load_processed(file_dic[img_key][exp_key], mode='img')
        log_final_output(percentages, img_key, image, mask=ensembled)
        print(percentages)

if __name__ == "__main__":
    do_postprocessing()
