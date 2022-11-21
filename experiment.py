#%%
import tensorflow as tf
from tqdm import tqdm
from module.load import load_dataset
from PIL import ImageDraw
from module.datasets.augment import fusion_edge

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
    if tf.reduce_any(gt_labels == 4).numpy(): break
tf.keras.utils.array_to_img(img)
draw_gt(img, gt_boxes, gt_labels, labels, colors)
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
from tensorflow.keras.applications.resnet50 import ResNet50
backbone = ResNet50(input_shape=[None, None, 3], include_top=False)
model.get
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

#%%

    
feature_model = tf.keras.models.Model(input )
backbone.summary()


import matplotlib.pyplot as plt

def show_feature_map():
    backbone = ResNet50(input_shape=[None, None, 3], include_top=False)
    c2_output, c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]

    feature_model = tf.keras.Model(
        inputs=[backbone.inputs], outputs=[c2_output, c3_output, c4_output, c5_output]
    )
    feature_model.build([None, None, 3])

    fig, axes = plt.subplots(nrows=1, ncols=4, dpi=1000)
    c2, c3, c4, c5 = feature_model(tf.expand_dims(img, 0))
    c2_, c3_, c4_, c5_ = feature_model(tf.expand_dims(edge_img, 0))
    

    plt.imshow(tf.reduce_sum(c2, [-1, 0]))
    plt.imshow(tf.reduce_sum(c3, [-1, 0]))
    plt.imshow(tf.reduce_sum(c4, [-1, 0]))
    plt.imshow(tf.reduce_sum(c5, [-1, 0]))

    plt.imshow(tf.reduce_sum(c2-c2_, [-1, 0]))
    plt.imshow(tf.reduce_sum(c3+c3_, [-1, 0]))
    plt.imshow(tf.reduce_sum(c4+c4_, [-1, 0]))
    plt.imshow(tf.reduce_sum(c5+c5_, [-1, 0]))
    
    plt.imshow(tf.reduce_sum(c2_, [-1, 0]))
    plt.imshow(tf.reduce_sum(c3_, [-1, 0]))
    plt.imshow(tf.reduce_sum(c4_, [-1, 0]))
    plt.imshow(tf.reduce_sum(c5_, [-1, 0]))

img_, edge = fusion_edge(img, fusion_scale=50)
tf.keras.utils.array_to_img(edge)
tf.keras.utils.array_to_img(img)
edge_img = tf.tile(edge, [1,1,3])

#%%
all_gt_boxes = []
all_gt_labels = []
for _ in tqdm(range(8000)):
    continue
    img, gt_boxes, gt_labels, filename = next(train_set)
    all_gt_boxes.append(gt_boxes)
    all_gt_labels.append(gt_labels)

all_gt_boxes = tf.concat(all_gt_boxes, axis=0)
all_gt_labels = tf.concat(all_gt_labels, axis=0)
import seaborn as sns

sns.histplot(all_gt_labels.numpy())
import pandas as pd

label_df = pd.Series(all_gt_labels.numpy()).value_counts().reset_index()
label_df.columns = ["Label", "Count"]

sns.barplot(data=label_df, x="Label", y="Count", color="#598DBC", edge_color="black")
sns.histplot(all_gt_labels.numpy())
from matplotlib import rc
rc('font', family='AppleGothic') 
fig, ax = plt.subplots(dpi=600)
plt.sca(ax)
bar = plt.bar(label_df["Label"], height=label_df["Count"], color= "#598DBC", edgecolor="black")
plt.xticks(label_df["Label"], labels_kor) 
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%i' % height, ha='center', va='bottom', size=8)

fig.savefig("./label_bar_600.eps")

labels[3] = "offshore windpower"
labels[2] = "-windpower"
labels_kor = ["선박", "기타부유물", "어망부표", "해상풍력", "등대", "부표"]

not_ship = all_gt_labels != 1
ship = all_gt_labels == 1
labels[0]
labels = [labels[i] for i in range(6)]
all_gt_boxes
all_gt_hws = tf.stack([
    (all_gt_boxes[...,2] - all_gt_boxes[...,0]) * 540,
    (all_gt_boxes[...,3] - all_gt_boxes[...,1]) * 960
    ], -1)
not_ship_hws = all_gt_hws[ship]
all_gt_hws
k_means

from module.datasets.eda import k_means, draw_hws
all_kmeans = k_means(not_ship_hws, 9)
fig_ship = draw_hws(all_kmeans)
fig_all
fig_not
fig_ship
from PIL import Image, ImageDraw
fig_all.save("./all_gt.pdf")
fig_all.save("./all_gt.eps")
fig_not.save("./not_ship_gt.pdf")
fig_not.save("./not_ship_gt.eps")
fig_ship.save("./ship_gt.pdf")
fig_ship.save("./ship_gt.eps")