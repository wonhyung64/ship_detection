#%%
import os 
import json
import numpy as np
import xml.etree.ElementTree as elemTree
import re
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

#%%
def serialize_example(dic):
    image = dic["image"].tobytes()
    image_shape = np.array(dic["image_shape"]).tobytes()
    bbox = dic["bbox"].tobytes()
    bbox_shape = np.array(dic["bbox_shape"]).tobytes()
    label = dic["label"].tobytes()
    filename = dic["filename"].tobytes()

    feature_dict={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'image_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_shape])),
        'bbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox])),
        'bbox_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox_shape])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict)) 
    return example.SerializeToString()

#%%
def deserialize_example(serialized_string):
    image_feature_description = { 
        'image': tf.io.FixedLenFeature([], tf.string), 
        'image_shape': tf.io.FixedLenFeature([], tf.string), 
        'bbox': tf.io.FixedLenFeature([], tf.string), 
        'bbox_shape': tf.io.FixedLenFeature([], tf.string), 
        'label': tf.io.FixedLenFeature([], tf.string), 
        'filename': tf.io.FixedLenFeature([], tf.string),
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

#%%
def save_dict_to_file(dic,dict_dir):
    f = open(dict_dir + '/label_dict.txt', 'w')
    f.write(str(dic))
    f.close()

#%%
def read_labels(label_dir):
    f = open(f"{label_dir}/labels.txt", "r")
    labels = f.read().split(",")
    del labels[-1]
    return labels


# %%
def fetch_dataset(name, split, img_size, file_dir="D:/won/data"):
    name = "ship"

    save_dir = f"{file_dir}/{name}_{img_size[0]}_{img_size[1]}"

    if os.path.exists(save_dir) == False:
        os.mkdirs(save_dir, exist_ok=True)
        label_dict = {}

        file_main_dir = f"{file_dir}/ship_detection/train/남해_여수항1구역_BOX"
        file_mid_dirs = [f"{file_main_dir}/{cont}" for cont in os.listdir(file_main_dir)]
        file_sub_dirs = []
        for file_mid_dir in file_mid_dirs:
            file_sub_dirs += [f"{file_mid_dir}/{cont}" for cont in os.listdir(file_mid_dir)]

        np.random.seed(1)
        train_dir_idx = np.random.choice(len(file_sub_dirs), 600, replace=False)
        rest_dir_idx = [x for x in range(len(file_sub_dirs)) if x not in train_dir_idx]
        valid_dir_idx = np.random.choice(len(rest_dir_idx), 100, replace=False)
        test_dir_idx = [x for x in range(len(rest_dir_idx)) if x not in valid_dir_idx]

        for split_idx, split_name in ((train_dir_idx, "train"), (valid_dir_idx), (test_dir_idx, "test")):
            writer = tf.io.TFRecordWriter(f'{save_dir}/{split_name}.tfrecord'.encode("utf-8"))
            split_progress = tqdm(range(len(split_idx)))
            split_progress.set_description(
                f"Fetch {split_name} set"
            )
            for i in split_progress:
                folder_dir = file_sub_dirs[split_idx[i]]
                folder_conts = os.listdir(folder_dir)
                filename_lst = sorted(list(set([folder_conts[l][:25] for l in range(len(folder_conts))])))
                for j in range(len(filename_lst)):
                    if j % 3 == 0:
                        sample_name = filename_lst[j]
                        sample_name_ = re.sub(r'[^0-9]', '', sample_name)
                        sample = f"{folder_dir}/{sample_name}"

                        #jpg
                        img_ = Image.open(sample + ".jpg")
                        img_ = tf.convert_to_tensor(np.array(img_, dtype=np.int32))
                        img_ = tf.image.resize(img_, img_size) / 255
                        norm_mean = (0.4738637621963933, 0.5181327285241354, 0.5290525313499966)
                        norm_mean = tf.expand_dims(tf.expand_dims(tf.constant(norm_mean), axis=0), axis=0)
                        norm_std = (0.243976435460058, 0.23966295898251888, 0.24247457088379498)
                        norm_std = tf.expand_dims(tf.expand_dims(tf.constant(norm_std), axis=0), axis=0)
                        norm_img = (img_ - norm_mean) / norm_std
                        img = np.array(norm_img)

                        #xml
                        tree = elemTree.parse(sample + ".xml")
                        root = tree.getroot()
                        bboxes_ = []
                        labels_ = []
                        for x in root:
                            if x.tag == "object":
                                for y in x:
                                    # print("--", y.tag)
                                    if y.tag == "bndbox":
                                        bbox_ = [int(z.text) for z in y] 
                                        bbox = [bbox_[1] / 2160 , bbox_[0] / 3840, bbox_[3] / 2160, bbox_[2] / 3840]
                                        # print("----", bbox)
                                        bboxes_.append(bbox)
                                    if y.tag == "category_id":
                                        # print("----", y.text)
                                        label = int(y.text)
                                        labels_.append(label)
                                    if y.tag == "name": 
                                        label_dict[str(label)] = y.text
                        bboxes = np.array(bboxes_, dtype=np.float32)
                        labels = np.array(labels_, dtype=np.int32)
                        bboxes = bboxes[labels == 2]
                        labels = labels[labels == 2] - 2

                        #json
                        with open(sample + "_meta.json", "r", encoding="UTF8") as st_json:
                            st_python = json.load(st_json)
                        st_python["Date"]
                        time = st_python["Date"][11:-1]
                        weather = st_python["Weather"]
                        season = st_python["Season"]

                        #to_dictionary
                        dic = {
                            "image":img,
                            "image_shape":img.shape,
                            "bbox":bboxes,
                            "bbox_shape":bboxes.shape,
                            "label":labels,
                            "filename":np.array([int(element) for element in list(sample_name_)])
                        }

                        writer.write(serialize_example(dic))

    dataset = tf.data.TFRecordDataset(f"{save_dir}/{split}.tfrecord".encode("utf-8")).map(deserialize_example)
    labels = read_labels(save_dir)

    return dataset, labels

eval("[1,3,4,5,6]")
