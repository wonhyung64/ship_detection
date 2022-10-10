import numpy as np
import pickle
from typing import *


def load_file_dirs(file_path: str) -> Tuple[list, int]:
    """
    Load data file directories saved as pickle

    Args:
        file_path (str): pickle file directory

    Returns:
        Tuple[list, int]: list of direcroties and file num
    """
    with open(file_path, 'rb') as lf:
        file_dir_list = pickle.load(lf)
    length = len(file_dir_list)
    print(length)

    return file_dir_list, length


def save_file_dirs(file_path: str, file_dirs: list) -> None:
    """
    Save file directories as pickle

    Args:
        file_path (str): save directory
        file_dirs (list): file directories list
    """
    with open(file_path, 'wb') as lf:
        pickle.dump(file_dirs, lf)


def split_dataset(file_dir_list: list, length: int) -> Tuple[list, list, list]:
    """
    File directory list to split

    Args:
        file_dir_list (list): total file directories
        length (int): total file num

    Returns:
        Tuple[list, list, list]: list of train, validation, test file dirctories
    """
    np.random.seed(1234)
    train_num ,valid_num, test_num = round(length * 0.8), round(length * 0.1), round(length * 0.1)
    split_list = []
    for data_num in [train_num, valid_num, test_num]:
        split_file_list = list(np.random.choice(file_dir_list, data_num, replace=False))
        file_dir_list = [file_dir for file_dir in file_dir_list if file_dir not in split_file_list]
        split_list.append(split_file_list)

    return split_list
