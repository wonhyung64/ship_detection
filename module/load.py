import os
import tensorflow as tf
from .datasets.utils import load_pickle
from .datasets.preprocess import fetch_dataset, deserialize_example


def load_dataset(data_dir, split="all"):
    data_dir = f"{data_dir}/ship"
    datasets = []
    if split == "all":
        split_list = ["train", "valid", "test"]
    elif type(split) == list:
        pass
    else:
        split_list = [split]

    for split in split_list:
        if f"{split}.tfrecord" not in os.listdir(data_dir):
            fetch_dataset(data_dir, split)
        dataset = tf.data.TFRecordDataset(
            f"{data_dir}/{split}.tfrecord".encode("utf-8")
            ).map(deserialize_example)
        datasets.append(dataset)

    labels, _ = load_pickle(f"{data_dir}/labels.pickle")
    
    return datasets, labels
