#%%
import re
import numpy as np
import os
import xml
import pickle
from tqdm import tqdm
from utils.aihub_utils import *
from typing import *
from module.datasets.utils import load_file_dirs




#%%
if __name__ == "__main__":
    path = "/Volumes/LaCie/data/해상 객체 이미지"
    train, length = load_file_dirs(f"{path}/train.pickle")
