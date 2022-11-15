#%%
import tensorflow as tf
from tqdm import tqdm
from module.load import load_dataset
from PIL import ImageDraw

def draw_gt(
    image,
    gt_boxes,
    gt_labels,
    labels,
    colors,
    img_size=[540, 960]
):
    image = tf.keras.preprocessing.image.array_to_img(image)
    draw = ImageDraw.Draw(image)
    gt_boxes *= tf.cast(tf.tile(img_size, [2]), dtype=tf.float32)

    for index, bbox in enumerate(gt_boxes):
        y1, x1, y2, x2 = tf.split(bbox, 4, axis=-1)
        label_index = int(gt_labels[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0}".format(labels[label_index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=1)

    return image
#%%
datasets, labels = load_dataset(data_dir="/Volumes/LaCie/data")
train_set, valid_set, test_set = datasets
train_set = iter(train_set)

for _ in tqdm(range(8000)):
    img, gt_boxes, gt_labels, filename = next(train_set)
    if tf.reduce_any(gt_labels == 0).numpy(): break
tf.keras.utils.array_to_img(img)

img
gt_boxes
gt_labels
colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)

labels


#%%
draw_gt(img, gt_boxes, gt_labels, labels, colors)
img

import cv2
backSub = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=10, detectShadows=False)
fgMask = backSub.apply(img.numpy())
tf.reduce_min(fgMask)

tf.keras.utils.array_to_img(tf.expand_dims(fgMask, -1))
tf.keras.utils.array_to_img(img)

from tensorflow.keras.applications.vgg16 import VGG16
model = VGG16(input_shape=[None, None, 3], include_top=False)
feature_map = model(tf.expand_dims(img, 0))
feature_map_ = tf.reduce_sum(feature_map, [-1, 0])
import matplotlib.pyplot as plt
plt.imshow(feature_map_)
img
buoy_img
import cv2
img = cv2.imread('image.png')
sharp_img = cv2.createBackgroundSubtractorMOG2().apply(img.numpy())
cv2.imshow(sharp_img)
