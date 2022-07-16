import os
import numpy as np
import tensorflow as tf
from models.faster_rcnn.utils import (
    load_data_num,
    rand_flip_horiz,
)
from .tfrecord_utils import deserialize_example
from . import aihub_utils, gc_utils


def load_dataset(name, data_dir, img_size):
    save_dir = f"{data_dir}/{name}_{img_size[0]}_{img_size[1]}"
    if not (os.path.exists(save_dir)):
        fetch_dataset(name, img_size, save_dir, data_dir)
    train_set, valid_set, test_set, labels = load_fetched_dataset(save_dir)
    train_num, valid_num, test_num = load_data_num(name, data_dir, train_set, valid_set, test_set)
    labels = ["bg"] + labels

    return (train_set, valid_set, test_set), labels, train_num, valid_num, test_num


def fetch_dataset(name, img_size, save_dir, data_dir="D:/won/data"):
    os.makedirs(save_dir, exist_ok=True)

    if name == "gc":
        data_sub_dirs = gc_utils.extract_sub_dir(data_dir)
    elif name == "aihub":
        data_sub_dirs = aihub_utils.extract_sub_dir(data_dir)

    train_dir_idx, valid_dir_idx, test_dir_idx = get_split_idx(data_sub_dirs)

    if name == "gc":
        gc_utils.write_datasets(
            train_dir_idx,
            valid_dir_idx,
            test_dir_idx,
            save_dir,
            data_sub_dirs,
            img_size,
        )
    elif name == "aihub":
        aihub_utils.write_datasets(
            train_dir_idx,
            valid_dir_idx,
            test_dir_idx,
            save_dir,
            data_sub_dirs,
            img_size,
        )


def get_split_idx(data_sub_dirs):
    np.random.seed(1)
    train_dir_idx = np.random.choice(
        len(data_sub_dirs), round(len(data_sub_dirs) * 0.7), replace=False
    )
    rest_dir_idx = [x for x in range(len(data_sub_dirs)) if x not in train_dir_idx]
    valid_dir_idx = np.random.choice(
        len(rest_dir_idx), round(len(rest_dir_idx) * 0.6), replace=False
    )
    test_dir_idx = [x for x in range(len(rest_dir_idx)) if x not in valid_dir_idx]

    return train_dir_idx, valid_dir_idx, test_dir_idx


def load_fetched_dataset(save_dir):
    train = tf.data.TFRecordDataset(f"{save_dir}/train.tfrecord".encode("utf-8")).map(
        deserialize_example
    )
    validation = tf.data.TFRecordDataset(
        f"{save_dir}/validation.tfrecord".encode("utf-8")
    ).map(deserialize_example)
    test = tf.data.TFRecordDataset(f"{save_dir}/test.tfrecord".encode("utf-8")).map(
        deserialize_example
    )
    labels = read_labels(save_dir)
    labels = preprocess_labels(labels)

    return train, validation, test, labels


def read_labels(save_dir):
    with open(f"{save_dir}/labels.txt", "r") as f:
        labels = eval(f.readline())

    return labels


def preprocess_labels(labels):
    labels_dict = {
        "어망부표": "fishing net buoy",
        "선박": "ship",
        "기타부유물": "other floats",
        "해상풍력": "offshore wind power",
        "등대": "lighthouse",
        "부표": "buoy",
    }
    labels = [labels_dict[k[1]] for k in sorted(labels.items())]

    return labels


def normalize_image(image):
    norm_mean = (0.4738637621963933, 0.5181327285241354, 0.5290525313499966)
    norm_mean = tf.expand_dims(tf.expand_dims(tf.constant(norm_mean), axis=0), axis=0)
    norm_std = (0.243976435460058, 0.23966295898251888, 0.24247457088379498)
    norm_std = tf.expand_dims(tf.expand_dims(tf.constant(norm_std), axis=0), axis=0)
    norm_img = (image - norm_mean) / norm_std

    return norm_img


def build_dataset(datasets, batch_size):
    train_set, valid_set, test_set = datasets

    data_shapes = ([None, None, None], [None, None], [None])
    padding_values = (
        tf.constant(0, tf.float32),
        tf.constant(0, tf.float32),
        tf.constant(-1, tf.int32),
    )

    train_set = train_set.map(lambda x, y, z, w: preprocess(x, y, z, w, split="train"))
    valid_set = valid_set.map(
        lambda x, y, z, w: preprocess(x, y, z, w, split="validation")
    )
    test_set = test_set.map(lambda x, y, z, w: preprocess(x, y, z, w, split="test"))

    train_set = train_set.repeat().padded_batch(
        batch_size,
        padded_shapes=data_shapes,
        padding_values=padding_values,
        drop_remainder=True,
    )
    valid_set = valid_set.repeat().padded_batch(
        batch_size=1,
        padded_shapes=data_shapes,
        padding_values=padding_values,
        drop_remainder=True,
    )
    test_set = test_set.repeat().padded_batch(
        batch_size=1,
        padded_shapes=data_shapes,
        padding_values=padding_values,
        drop_remainder=True,
    )

    train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE)
    valid_set = valid_set.prefetch(tf.data.experimental.AUTOTUNE)
    test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE)

    train_set = iter(train_set)
    valid_set = iter(valid_set)
    test_set = iter(test_set)

    return train_set, valid_set, test_set


def preprocess(image, gt_boxes, gt_labels, file_name, split):
    if split == "train":
        image, gt_boxes = rand_flip_horiz(image, gt_boxes)

    return image, gt_boxes, gt_labels
