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

#%%
def fetch_dataset(dataset, split, img_size, file_dir="D:/won/data", save_dir="D:/won/data"):
    save_dir = f"{save_dir}/{dataset}_tfrecord_{img_size[0]}_{img_size[1]}"

    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)
        try_num = 0
        label_dict = {}
        filename_lst = []
        file_dir1 = file_dir

        writer1 = tf.io.TFRecordWriter(f'{save_dir}/train.tfrecord'.encode("utf-8"))
        writer2 = tf.io.TFRecordWriter(f'{save_dir}/test.tfrecord'.encode("utf-8"))

        file_dir2 = file_dir1 + "/ship_detection/train/남해_여수항1구역_BOX"
        file_dir2_conts = os.listdir(file_dir2)
        if ".DS_Store" in file_dir2_conts: file_dir2_conts.remove(".DS_Store")

        for i in range(len(file_dir2_conts)):
            file_dir3 = file_dir2 + "/" + file_dir2_conts[i]
            file_dir3_conts = os.listdir(file_dir3)
            if ".DS_Store" in file_dir3_conts: file_dir3_conts.remove(".DS_Store")

            for j in range(len(file_dir3_conts)):
                file_dir4 = file_dir3 + "/" + file_dir3_conts[j]
                file_dir4_conts = os.listdir(file_dir4)
                if ".DS_Store" in file_dir4_conts: file_dir4_conts.remove(".DS_Store")
                filename_lst = list(set([file_dir4_conts[l][:25] for l in range(len(file_dir4_conts))]))

                for k in range(len(filename_lst)):
                    try_num += 1
                    if try_num % 3 == 0:
                        print(try_num)
                        file_dir5 = file_dir4 + "/" + filename_lst[k]
                        filename = re.sub(r'[^0-9]', '', filename_lst[k])
                        filename_ = list(filename)

                        #jpg
                        img_ = Image.open(file_dir5 + ".jpg")
                        img_ = tf.convert_to_tensor(np.array(img_, dtype=np.int32)) / 255 # image
                        img_ = tf.image.resize(img_, img_size)
                        img = np.array(img_)

                        #xml
                        tree = elemTree.parse(file_dir5 + ".xml")
                        root = tree.getroot()
                        bboxes_ = []
                        labels_ = []
                        for x in root:
                            # print(x.tag)
                            if x.tag == "object":
                                for y in x:
                                    # print("--", y.tag)
                                    if y.tag == "bndbox":
                                        bbox_ = [int(z.text) for z in y] 
                                        bbox = [bbox_[1] / 2160, bbox_[0] / 3840, bbox_[3] / 2160, bbox_[2] / 3840]
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
                        with open(file_dir5 + "_meta.json", "r", encoding="UTF8") as st_json:
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
                            "filename":np.array(filename_)
                        }

                        info_ = {
                            "time":time,
                            "weather":weather,
                            "season":season,
                        }

                        info = np.array([info_])
                        if try_num % 50 == 0 : writer2.write(serialize_example(dic))
                        else:writer1.write(serialize_example(dic))

                        if os.path.isdir(save_dir + "/meta") == False: os.mkdir(save_dir + "/meta")

                        info_dir = save_dir + "/meta/" + filename
                        np.save(info_dir + ".npy", info, allow_pickle=True)

        save_dict_to_file(label_dict, save_dir)

    dataset = tf.data.TFRecordDataset(f"{save_dir}/{split}.tfrecord".encode("utf-8")).map(deserialize_example)
    labels = read_labels(save_dir)

    return dataset, labels

# %%
def fetch_dataset_v2(dataset, split, img_size, file_dir="D:/won/data", save_dir="D:/won/data"):

    save_dir = f"{save_dir}/{dataset}_tfrecord_{img_size[0]}_{img_size[1]}"

    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)
        label_dict = {}
        folder_dir_lst = []

        file_dir1 = file_dir
        file_dir2 = f"{file_dir1}/ship_detection/train/남해_여수항1구역_BOX"
        file_dir2_conts = os.listdir(file_dir2)
        for i in range(len(file_dir2_conts)):
            file_dir3 = f"{file_dir2}/{file_dir2_conts[i]}"
            file_dir3_conts = os.listdir(file_dir3)
            for j in range(len(file_dir3_conts)):
                file_dir4 = f"{file_dir3}/{file_dir3_conts[j]}"
                folder_dir_lst.append(file_dir4)

        np.random.seed(1)
        train_dir_idx = np.random.choice(len(folder_dir_lst), 600, replace=False)
        test_dir_idx = [x for x in range(len(folder_dir_lst)) if x not in train_dir_idx]

        for split_idx, split_name in ((train_dir_idx, "train"), (test_dir_idx, "test")):
            writer = tf.io.TFRecordWriter(f'{save_dir}/{split_name}.tfrecord'.encode("utf-8"))
            for i in tqdm(range(len(split_idx))):
                folder_dir = folder_dir_lst[train_dir_idx[i]]
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
                            # print(x.tag)
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

                        info_ = {
                            "time":time,
                            "weather":weather,
                            "season":season,
                        }

                        info = np.array([info_])

                        writer.write(serialize_example(dic))
                        if os.path.isdir(save_dir + "/meta") == False: os.mkdir(save_dir + "/meta")

                        info_dir = f"{save_dir}/meta/{sample_name_}"
                        np.save(info_dir + ".npy", info, allow_pickle=True)

    dataset = tf.data.TFRecordDataset(f"{save_dir}/{split}.tfrecord".encode("utf-8")).map(deserialize_example)
    labels = read_labels(save_dir)

    return dataset, labels

#%%
def serialize_feature(dic):
    filename = np.array(dic["filename"]).tobytes()
    feature_map = np.array(dic["feature_map"]).tobytes()
    dtn_reg_output = np.array(dic["dtn_reg_output"]).tobytes()
    dtn_cls_output = np.array(dic["dtn_cls_output"]).tobytes()
    best_threshold = np.array(dic["best_threshold"]).tobytes()

    feature_dict={
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'feature_map': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature_map])),
        'dtn_reg_output': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dtn_reg_output])),
        'dtn_cls_output': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dtn_cls_output])),
        'best_threshold': tf.train.Feature(bytes_list=tf.train.BytesList(value=[best_threshold])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict)) 
    return example.SerializeToString()

#%%
def deserialize_feature(serialized_string):
    image_feature_description = { 
        'filename': tf.io.FixedLenFeature([], tf.string), 
        'feature_map': tf.io.FixedLenFeature([], tf.string), 
        'dtn_reg_output': tf.io.FixedLenFeature([], tf.string), 
        'dtn_cls_output': tf.io.FixedLenFeature([], tf.string),
        'best_threshold': tf.io.FixedLenFeature([], tf.string),
    } 
    example = tf.io.parse_single_example(serialized_string, image_feature_description) 

    filename = tf.io.decode_raw(example["filename"], tf.int32)
    feature_map = tf.io.decode_raw(example["feature_map"], tf.float32)
    dtn_reg_output = tf.io.decode_raw(example["dtn_reg_output"], tf.float32) 
    dtn_cls_output = tf.io.decode_raw(example["dtn_cls_output"], tf.float32)
    best_threshold = tf.io.decode_raw(example["best_threshold"], tf.float32)

    feature_map = tf.reshape(feature_map, (31, 31, 512))
    dtn_reg_output = tf.reshape(dtn_reg_output, (1500, 16))
    dtn_cls_output = tf.reshape(dtn_cls_output, (1500, 4))

    return filename, feature_map, dtn_reg_output, dtn_cls_output, best_threshold
