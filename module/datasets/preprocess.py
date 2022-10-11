import os
import re
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as elemTree
from PIL import Image
from tqdm import tqdm
from typing import *
from .utils import load_pickle, save_pickle, unicode2ascii


def extract_img(img_dir: str, reduce_ratio: float=1/4) -> Tuple[tf.Tensor, tuple]:
    """
    Extract image file from directory to tf.Tensor array

    Args:
        img_dir (str): image file directory without extension name
        reduce_ratio (float, optional): image resize ratio. Defaults to 1/4.

    Returns:
        Tuple[tf.Tensor, tuple]: tuple of image tensor and original image size. image size is formated as (w, h)
    """
    img =  Image.open(f"{img_dir}.jpg")
    img_size = img.size

    img = tf.keras.utils.img_to_array(img, dtype=np.float32)
    img = tf.image.resize(
        img,
        [int(x * reduce_ratio) for x in reversed(img_size)],
        method="bicubic"
        )
    
    return img, img_size


def extract_ant(ant_dir, label_dict, img_size):
    tree = elemTree.parse(f"{ant_dir}.xml")
    root = tree.getroot()
    bboxes, labels = [], []

    for x in root:
        if x.tag == "object":
            for y in x:
                if y.tag == "bndbox":
                    bbox = [int(z.text) for z in y]
                    bbox_ = [
                        bbox[1] / img_size[1],
                        bbox[0] / img_size[0],
                        bbox[3] / img_size[1],
                        bbox[2] / img_size[0],
                    ]
                    bboxes.append(bbox_)

                if y.tag == "category_id":
                    label = int(y.text) - 1
                    labels.append(label)

                if y.tag == "name":
                    label_dict[label] = y.text

    gt_boxes = tf.cast(bboxes, dtype=tf.float32)
    gt_labels = tf.cast(labels, dtype=tf.int32)

    return gt_boxes, gt_labels, label_dict


def fetch_dataset(path: str, split: str) -> None:
    file_dirs, _ = load_pickle(f"{path}/{split}.pickle")
    label_dict = {}
    writer = tf.io.TFRecordWriter(
        f"{path}/{split}.tfrecord".encode("utf-8")
    )

    progress = tqdm(file_dirs)
    progress.set_description(f"Generate {split} dataset")
    for file_dir in progress:
        
        img, img_size = extract_img(re.sub("label", "image", file_dir))
        gt_boxes, gt_labels, label_dict = extract_ant(file_dir, label_dict, img_size)
        filename = "/".join(file_dir.split("/")[6:])

        dic = {
            "image": img,
            "image_shape": tf.shape(img),
            "bbox": gt_boxes,
            "bbox_shape": tf.shape(gt_boxes),
            "label": gt_labels,
            "filename": tf.cast(unicode2ascii(filename), dtype=tf.int32)
                }

        writer.write(serialize_example(dic))
    
    if "labels.pickle" not in os.listdir(path):
        save_pickle(f"{path}/labels.pickle", label_dict)


def serialize_example(dic):
    image = np.array(dic["image"]).tobytes()
    image_shape = np.array(dic["image_shape"]).tobytes()
    bbox = np.array(dic["bbox"]).tobytes()
    bbox_shape = np.array(dic["bbox_shape"]).tobytes()
    label = np.array(dic["label"]).tobytes()
    filename = np.array(dic["filename"]).tobytes()

    feature_dict = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        "image_shape": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_shape])
        ),
        "bbox": tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox])),
        "bbox_shape": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[bbox_shape])
        ),
        "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        "filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example.SerializeToString()


def deserialize_example(serialized_string):
    image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_shape": tf.io.FixedLenFeature([], tf.string),
        "bbox": tf.io.FixedLenFeature([], tf.string),
        "bbox_shape": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
        "filename": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(serialized_string, image_feature_description)

    image = tf.io.decode_raw(example["image"], tf.float32)
    image_shape = tf.io.decode_raw(example["image_shape"], tf.int32)
    bbox = tf.io.decode_raw(example["bbox"], tf.float32)
    bbox_shape = tf.io.decode_raw(example["bbox_shape"], tf.int32)
    label = tf.io.decode_raw(example["label"], tf.int32)
    filename = tf.io.decode_raw(example["filename"], tf.int32)

    image = tf.reshape(image, image_shape)
    bbox = tf.reshape(bbox, bbox_shape)

    return image, bbox, label, filename
