import numpy as np
import pickle
from typing import *


def unicode2ascii(string: str) -> list:
    return [ord(x) for x in string]


def ascii2unicode(int_list) -> str:
    return "".join([chr(x) for x in int_list])


def load_pickle(file_path: str) -> Tuple[list, int]:
    """
    Load data saved as pickle

    Args:
        file_path (str): pickle file directory

    Returns:
        Tuple[list, int]: contents and length
    """
    with open(file_path, 'rb') as lf:
        contents = pickle.load(lf)
    length = len(contents)

    return contents, length


def save_pickle(file_path: str, contents: list) -> None:
    """
    Save file as pickle

    Args:
        file_path (str): save directory
        file_dirs (list): contents
    """
    with open(file_path, 'wb') as lf:
        pickle.dump(contents, lf)


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
