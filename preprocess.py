#%%
import re
import numpy as np
import os
import xml
import pickle
from tqdm import tqdm
from utils.aihub_utils import *


def load_file_dirs(file_path):
    with open(file_path, 'rb') as lf:
        file_dir_list = pickle.load(lf)
    length = len(file_dir_list)
    print(length)

    return file_dir_list, length


def save_file_dirs(file_path, file_dirs):
    with open(file_path, 'wb') as lf:
        pickle.dump(file_dirs, lf)


def split_dataset(file_dir_list, length):
    np.random.seed(1234)
    train_num ,valid_num, test_num = round(length * 0.8), round(length * 0.1), round(length * 0.1)
    split_list = []
    for data_num in [train_num, valid_num, test_num]:
        split_file_list = list(np.random.choice(file_dir_list, data_num, replace=False))
        file_dir_list = [file_dir for file_dir in file_dir_list if file_dir not in split_file_list]
        split_list.append(split_file_list)

    return split_list


#%%
path = "/Volumes/LaCie/data/해상 객체 이미지"
various_path = f'{path}/dir_25pix_various_one.pickle'
ship_path = f'{path}/dir_25pix_ship_one.pickle'
various_list, various_length = load_file_dirs(various_path)
ship_list, ship_length = load_file_dirs(ship_path)


ship_train, ship_valid, ship_test = split_dataset(ship_list, ship_length)
various_train, various_valid, various_test = split_dataset(various_list, various_length)
train = ship_train + various_train
valid = ship_valid + various_valid
test = ship_test + various_test

for split, dataset in [("train", train), ("valid", valid), ("test", test)]:
    save_file_dirs(f"{path}/{split}.pickle", dataset)