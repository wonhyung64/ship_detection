#%%
import numpy as np
import cv2
import tensorflow as tf
from module.load import load_dataset

def fusion_edge(img, fusion_scale):
    bilateral = cv2.bilateralFilter(np.array(img), d=-1, sigmaColor=10, sigmaSpace=5)
    for _ in range(14):
        bilateral = cv2.bilateralFilter(bilateral, d=-1, sigmaColor=10, sigmaSpace=5)
    gaussian = cv2.GaussianBlur(bilateral, (5,5), 0)
    gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    threshold1, _ = cv2.threshold(gray.astype(np.uint8), -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    canny = cv2.Canny(
        gaussian.astype(np.uint8),
        threshold1=threshold1,
        threshold2=255,
        apertureSize=3,
        L2gradient=False
        )
    canny = tf.cast(tf.expand_dims(canny / 255, -1), dtype=tf.float32)
    augmented_img = tf.clip_by_value(canny*fusion_scale + img, tf.reduce_min(img), tf.reduce_max(img))

    return augmented_img, canny


def pixel_scaling(img):
    minimum = tf.math.minimum(tf.reduce_min(img), 0.)
    img = img - minimum
    maximum = tf.math.maximum(tf.reduce_max(img), 255.)
    img = img / maximum * 255.
    
    return img


def retina_preprocess(img, gt_boxes, gt_labels, filename, fusion_scale, with_filename):
    img = pixel_scaling(img)
    if fusion_scale == None:
        output_img = img
    else:
        output_img, _ = fusion_edge(img, fusion_scale)

    if with_filename:
        return output_img, gt_boxes, gt_labels, filename
    return output_img, gt_boxes, gt_labels



#%%
[train_set, valid_set, test_set], labels = load_dataset()

valid_set = valid_set.map(
    lambda x, y, w, z:
        tf.py_function(
            retina_preprocess,
            [x, y, w, z, 50., False],
            [tf.float32, tf.float32, tf.int32, tf.float32]
            )
        )
import tensorflow_datasets as tfds

dataset = tfds.load(name="voc/2007", data_dir="/Volumes/Lacie/data/tfds/")
voc = iter(dataset["train"])
voc_img = next(voc)["image"]
voc_box = next(voc)["objects"]["bbox"]
voc_label = next(voc)["objects"]["label"]
tf.reduce_max(voc_img)
tf.reduce_min(voc_img)


tmp = iter(valid_set)
img, gt_boxes, gt_labels, aug_img = next(tmp)
tf.keras.utils.array_to_img(img)
tf.keras.utils.array_to_img(aug_img)

img, gt_boxes, gt_labels, filename = next(test_set)


img, gt_boxes, gt_labels, filename = next(test_set)
augmented_img, _ = fusion_edge(img)
tf.keras.utils.array_to_img(_)

tf.keras.utils.array_to_img(img)
tf.keras.utils.array_to_img(augmented_img)

