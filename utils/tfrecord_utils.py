import numpy as np
import tensorflow as tf

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
