import re
import os
import json
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as elemTree
from PIL import Image
from tqdm import tqdm
from .tfrecord_utils import serialize_example


def extract_sub_dir(data_dir):
    data_main_dir = f"{data_dir}/ship_detection/train/남해_여수항1구역_BOX"
    data_mid_dirs = [f"{data_main_dir}/{cont}" for cont in os.listdir(data_main_dir)]
    data_sub_dirs = []
    for data_mid_dir in data_mid_dirs:
        data_sub_dirs += [f"{data_mid_dir}/{cont}" for cont in os.listdir(data_mid_dir)]

    return data_sub_dirs


def write_datasets(
    train_dir_idx, valid_dir_idx, test_dir_idx, save_dir, data_sub_dirs, img_size
):
    error_num = 0
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
            folder_dir = data_sub_dirs[split_idx[i]]
            folder_conts = os.listdir(folder_dir)
            filename_lst = sorted(
                list(set([folder_conts[l][:25] for l in range(len(folder_conts))]))
            )
            try:
                for j in range(len(filename_lst)):
                    if j % 3 == 0:
                        sample_name = filename_lst[j]
                        sample_name_ = re.sub(r"[^0-9]", "", sample_name)
                        sample = f"{folder_dir}/{sample_name}"

                        image, org_img_size = extract_image(sample, img_size)
                        bboxes, labels, label_dict = extract_annot(
                            sample, label_dict, org_img_size
                        )

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
            except:
                error_num += 1
                print(f"\n {error_num} Error Occured\n")
                continue

    write_labels(save_dir, label_dict)


def extract_image(sample, img_size):
    image = Image.open(f"{sample}.jpg")
    image = tf.convert_to_tensor(np.array(image, dtype=np.int32))
    org_img_size = tf.shape(image).numpy()
    image = tf.image.resize(image, img_size)
    if tf.reduce_max(image).numpy() >= 1:
        image /= 255
    image = np.array(image)

    return image, org_img_size


def extract_annot(sample, label_dict, org_img_size):
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
                        bbox_[1] / org_img_size[0],
                        bbox_[0] / org_img_size[1],
                        bbox_[3] / org_img_size[0],
                        bbox_[2] / org_img_size[1],
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


def write_labels(save_dir, label_dict):
    with open(f"{save_dir}/labels.txt", "w") as f:
        f.write(json.dumps(label_dict, ensure_ascii=False))
