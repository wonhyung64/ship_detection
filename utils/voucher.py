import tensorflow as tf
from tensorflow.keras.layers import Lambda    
from typing import Tuple


def build_dataset(args):
    train_set, valid_set, test_set = load_fetched_dataset(args.data_dir)
    train_set = train_set.map(lambda x: preprocess(x, split="train", img_size=args.img_size))
    valid_set = valid_set.map(lambda x: preprocess(x, split="valid", img_size=args.img_size))
    test_set = test_set.map(lambda x: preprocess(x, split="test", img_size=args.img_size))

    # data_shapes = ([None, None, None], [None, None], [None], [])
    # padding_values=(0., 0., -1, "")
    data_shapes = ([None, None, None], [None, None], [None])
    padding_values=(0., 0., -1)
    train_set = train_set.repeat().padded_batch(
        args.batch_size,
        padded_shapes=data_shapes,
        padding_values=padding_values,
        drop_remainder=True
    )
    valid_set = valid_set.repeat(100).batch(1)
    test_set = test_set.repeat().batch(1)

    train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE)
    valid_set = valid_set.prefetch(tf.data.experimental.AUTOTUNE)
    test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE)

    train_set = iter(train_set)
    valid_set = iter(valid_set)
    test_set = iter(test_set)
    
    return train_set, valid_set, test_set

    
def parse_func(example):
    image = tf.io.decode_raw(example["image"], tf.float32)
    bbox = tf.io.decode_raw(example["bbox"], tf.float32)
    bbox_shape = tf.io.decode_raw(example["bbox_shape"], tf.int32)
    image=tf.reshape(image,[512, 512, 3])
    bbox=tf.reshape(bbox,bbox_shape)
        
    filename=example["filename"]

    return {'image':image,'bbox':bbox,'filename':filename}


def decode_filename(strtensor):
    strtensor=strtensor.numpy()
    filename=[str(i,'utf-8') for i in strtensor]
    return filename


def deserialize_example(serialized_string):
    image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_shape": tf.io.FixedLenFeature([], tf.string),
        "bbox": tf.io.FixedLenFeature([],tf.string),
        "bbox_shape": tf.io.FixedLenFeature([],tf.string),
        "filename": tf.io.FixedLenFeature([],tf.string)
    }
    example = tf.io.parse_example(serialized_string, image_feature_description)
    dataset=parse_func(example)
            
    return dataset


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

    return train, validation, test

def preprocess(dataset, split, img_size):
    image, gt_boxes, filename = export_data(dataset)
    image = resize(image, img_size)
    if split == "train":
        image, gt_boxes = rand_flip_horiz(image, gt_boxes)
    gt_labels = tf.zeros_like(gt_boxes[...,0], dtype=tf.int32)

    return image, gt_boxes, gt_labels


def export_data(sample):
    image = Lambda(lambda x: x["image"])(sample)
    gt_boxes = Lambda(lambda x: x["bbox"])(sample)
    filename = Lambda(lambda x: x["filename"])(sample)

    return image, gt_boxes, filename


def resize(image, img_size):
    transform = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Resizing(
                img_size[0], img_size[1]
            ),
        ]
    )
    image = transform(image)

    return image


def rand_flip_horiz(image: tf.Tensor, gt_boxes: tf.Tensor) -> Tuple:
    if tf.random.uniform([1]) > tf.constant([0.5]):
        image = tf.image.flip_left_right(image)
        gt_boxes = tf.stack(
            [
                Lambda(lambda x: x[..., 0])(gt_boxes),
                Lambda(lambda x: 1.0 - x[..., 3])(gt_boxes),
                Lambda(lambda x: x[..., 2])(gt_boxes),
                Lambda(lambda x: 1.0 - x[..., 1])(gt_boxes),
            ],
            -1,
        )

    return image, gt_boxes
