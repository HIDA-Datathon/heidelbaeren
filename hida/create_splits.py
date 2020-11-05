import pickle
import os 
import glob
import numpy as np


def create_splits(output_dir, image_dir):
    data_list=[]

    for file in os.listdir(image_dir):
        if file.endswith(".jpg"):
          data_list.append(file)
    #print the length of the folder
    print(len(data_list))

    #split the data into 5 patches of roughly the same size
    split_size1 = int(np.floor(len(data_list)/5))
    split_size2 = split_size1
    split_size3 = split_size1
    split_size4 = split_size1
    split_size5 = split_size1 + int(np.mod(len(data_list),5))
    print(split_size1)
    print(split_size5)

    #do random splits
    image_list = data_list.copy()
    set1 = []
    set2 = []
    set3 = []
    set4 = []
    set5 = []
    for i in range(0, split_size1):
        data = np.random.choice(image_list)
        image_list.remove(data)
        set1.append(data)
    for i in range(0, split_size2):
        data = np.random.choice(image_list)
        image_list.remove(data)
        set2.append(data)
    for i in range(0, split_size3):
        data = np.random.choice(image_list)
        image_list.remove(data)
        set3.append(data)
    for i in range(0, split_size4):
        data = np.random.choice(image_list)
        image_list.remove(data)
        set4.append(data)
    for i in range(0, split_size5):
        data = np.random.choice(image_list)
        image_list.remove(data)
        set5.append(data)

    #save the random splits as .txt files
    with open(os.path.join(output_dir, 'dataset1.txt'), 'w') as f:
        for item in set1:
            f.write("%s\n" % item)
    with open(os.path.join(output_dir, 'dataset2.txt'), 'w') as f:
        for item in set2:
            f.write("%s\n" % item)
    with open(os.path.join(output_dir, 'dataset3.txt'), 'w') as f:
        for item in set3:
            f.write("%s\n" % item)
    with open(os.path.join(output_dir, 'dataset4.txt'), 'w') as f:
        for item in set4:
            f.write("%s\n" % item)
    with open(os.path.join(output_dir, 'dataset5.txt'), 'w') as f:
        for item in set5:
            f.write("%s\n" % item)    

create_splits('/home/ksquare/repositories/datathon/data/ufz_im_challenge/', '/home/ksquare/repositories/datathon/data/ufz_im_challenge/photos_annotated') 