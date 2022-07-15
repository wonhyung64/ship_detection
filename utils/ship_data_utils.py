import json
import tensorflow as tf
from models.faster_rcnn.utils import (
    load_data_num,
    rand_flip_horiz,
)
from .aihub_utils import fetch_dataset

def load_dataset(name, data_dir, img_size):
    train_set, valid_set, test_set, labels = fetch_dataset(img_size, data_dir)
    train_num, valid_num ,test_num = load_data_num(name, train_set, valid_set, test_set)

    return (train_set, valid_set, test_set), labels, train_num, valid_num, test_num


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
    valid_set = valid_set.map(lambda x, y, z, w: preprocess(x, y, z, w, split="validation"))
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

