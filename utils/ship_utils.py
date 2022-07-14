import re
import os
import json
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as elemTree
from PIL import Image
from tqdm import tqdm
from models.faster_rcnn.utils import (
    load_data_num,
    rand_flip_horiz,
)

def build_sample_set(name, file_dir, img_size):
    save_dir = f"{file_dir}/{name}_{img_size[0]}_{img_size[1]}"

    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir, exist_ok=True)
        file_sub_dir = f"{file_dir}/test_image"
        writer = tf.io.TFRecordWriter(
            f"{save_dir}/sample.tfrecord".encode("utf-8")
        )
        for content in os.listdir(file_sub_dir):
            image_dir = f"{file_sub_dir}/{content}"
            image = Image.open(image_dir)
            image = tf.convert_to_tensor(np.array(image, dtype=np.int32))
            image = tf.image.resize(image, img_size) / 255
            image = np.array(image)
            dic = {
                "image": image,
            }
            image = dic["image"].tobytes()
            feature_dict = {
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

            writer.write(example.SerializeToString())

    sample = tf.data.TFRecordDataset(f"{save_dir}/sample.tfrecord".encode("utf-8")).map(
        deserialize_sample
    )
    sample

    sample = sample.repeat().batch(1)

    sample = sample.prefetch(tf.data.experimental.AUTOTUNE)
    
    sample = iter(sample)

    return sample


def deserialize_sample(serialized_string):
    image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(serialized_string, image_feature_description)

    image = tf.io.decode_raw(example["image"], tf.float32)

    image = tf.reshape(image, (500, 500, 3))

    return image


def load_ship_dataset(name, data_dir, img_size):
    train_set, valid_set, test_set, labels = fetch_dataset(img_size, data_dir)
    train_num ,test_num = load_data_num(name, train_set, test_set)
    labels = preprocess_labels(labels)

    return (train_set, valid_set, test_set), labels, train_num, test_num


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


def fetch_dataset(img_size, file_dir="D:/won/data"):
    save_dir = f"{file_dir}/ship_{img_size[0]}_{img_size[1]}"

    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir, exist_ok=True)

        file_sub_dirs = extract_sub_dir(file_dir)
        train_dir_idx, valid_dir_idx, test_dir_idx = get_split_idx(file_sub_dirs)
        label_dict = write_datasets(
            train_dir_idx,
            valid_dir_idx,
            test_dir_idx,
            save_dir,
            file_sub_dirs,
            img_size,
        )
        write_labels(save_dir, label_dict)

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

    return train, validation, test, labels


def extract_sub_dir(file_dir):
    file_main_dir = f"{file_dir}/ship_detection/train/남해_여수항1구역_BOX"
    file_mid_dirs = [f"{file_main_dir}/{cont}" for cont in os.listdir(file_main_dir)]
    file_sub_dirs = []
    for file_mid_dir in file_mid_dirs:
        file_sub_dirs += [f"{file_mid_dir}/{cont}" for cont in os.listdir(file_mid_dir)]

    return file_sub_dirs


def get_split_idx(file_sub_dirs):
    np.random.seed(1)
    train_dir_idx = np.random.choice(len(file_sub_dirs), 600, replace=False)
    rest_dir_idx = [x for x in range(len(file_sub_dirs)) if x not in train_dir_idx]
    valid_dir_idx = np.random.choice(len(rest_dir_idx), 100, replace=False)
    test_dir_idx = [x for x in range(len(rest_dir_idx)) if x not in valid_dir_idx]

    return train_dir_idx, valid_dir_idx, test_dir_idx


def write_datasets(
    train_dir_idx, valid_dir_idx, test_dir_idx, save_dir, file_sub_dirs, img_size
):
    label_dict = {}
    for split_idx, split_name in (
        (train_dir_idx, "train"),
        (valid_dir_idx, "validation"),
        (test_dir_idx, "test"),
    ):
        writer = tf.io.TFRecordWriter(
            f"{save_dir}/{split_name}.tfrecord".encode("utf-8")
        )
        split_progress = tqdm(range(len(split_idx)))
        split_progress.set_description(f"Fetch {split_name} set")
        for i in split_progress:
            folder_dir = file_sub_dirs[split_idx[i]]
            folder_conts = os.listdir(folder_dir)
            filename_lst = sorted(
                list(set([folder_conts[l][:25] for l in range(len(folder_conts))]))
            )
            for j in range(len(filename_lst)):
                if j % 3 == 0:
                    sample_name = filename_lst[j]
                    sample_name_ = re.sub(r"[^0-9]", "", sample_name)
                    sample = f"{folder_dir}/{sample_name}"

                    image = extract_image(sample, img_size)
                    bboxes, labels, label_dict = extract_annot(sample, label_dict)
                    # to_dictionary
                    dic = {
                        "image": image,
                        "image_shape": image.shape,
                        "bbox": bboxes,
                        "bbox_shape": bboxes.shape,
                        "label": labels,
                        "filename": np.array(
                            [int(element) for element in list(sample_name_)]
                        ),
                    }

                    writer.write(serialize_example(dic))

    return label_dict


def extract_image(sample, img_size):
    # jpg
    image = Image.open(f"{sample}.jpg")
    image = tf.convert_to_tensor(np.array(image, dtype=np.int32))
    image = tf.image.resize(image, img_size) / 255
    # image = normalize_image(image)
    image = np.array(image)

    return image


def normalize_image(image):
    norm_mean = (0.4738637621963933, 0.5181327285241354, 0.5290525313499966)
    norm_mean = tf.expand_dims(tf.expand_dims(tf.constant(norm_mean), axis=0), axis=0)
    norm_std = (0.243976435460058, 0.23966295898251888, 0.24247457088379498)
    norm_std = tf.expand_dims(tf.expand_dims(tf.constant(norm_std), axis=0), axis=0)
    norm_img = (image - norm_mean) / norm_std

    return norm_img


def extract_annot(sample, label_dict):
    # xml
    tree = elemTree.parse(f"{sample}.xml")
    root = tree.getroot()
    bboxes_ = []
    labels_ = []
    for x in root:
        if x.tag == "object":
            for y in x:
                if y.tag == "bndbox":
                    bbox_ = [int(z.text) for z in y]
                    bbox = [
                        bbox_[1] / 2160,
                        bbox_[0] / 3840,
                        bbox_[3] / 2160,
                        bbox_[2] / 3840,
                    ]
                    bboxes_.append(bbox)
                if y.tag == "category_id":
                    label = int(y.text)
                    labels_.append(label)
                if y.tag == "name":
                    label_dict[label] = y.text
    bboxes = np.array(bboxes_, dtype=np.float32)
    labels = np.array(labels_, dtype=np.int32)

    return bboxes, labels, label_dict


def serialize_example(dic):
    image = dic["image"].tobytes()
    image_shape = np.array(dic["image_shape"]).tobytes()
    bbox = dic["bbox"].tobytes()
    bbox_shape = np.array(dic["bbox_shape"]).tobytes()
    label = dic["label"].tobytes()
    filename = dic["filename"].tobytes()

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


def write_labels(save_dir, label_dict):
    with open(f"{save_dir}/labels.txt", "w") as f:
        f.write(json.dumps(label_dict, ensure_ascii=False))


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


def read_labels(save_dir):
    with open(f"{save_dir}/labels.txt", "r") as f:
        labels = eval(f.readline())

    return labels


def build_dataset(datasets, batch_size, img_size):
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
    gt_labels = tf.cast(gt_labels, dtype=tf.int32)

    return image, gt_boxes, gt_labels
