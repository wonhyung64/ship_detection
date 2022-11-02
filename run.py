#%%
import numpy as np
import cv2
import tensorflow as tf
from module.load import load_dataset
from sklearn.cluster import KMeans

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
    if fusion_scale == -1.:
        output_img = img
    else:
        output_img, _ = fusion_edge(img, fusion_scale)

    if with_filename:
        return output_img, gt_boxes, gt_labels, filename
    return output_img, gt_boxes, gt_labels

from models.retinanet.module.bbox import swap_xy, convert_to_xywh
from models.retinanet.module.preprocess import resize_and_pad_image
from models.retinanet.module.target import LabelEncoder

def preprocess_train(image, gt_boxes, gt_labels):
    bbox = swap_xy(gt_boxes)
    class_id = gt_labels

    image_shape = [512, 512]
    image_shape = tf.cast(image_shape, dtype=tf.float32)
    # image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)

    return image, bbox, class_id


def preprocess_test(image, gt_boxes, gt_labels):

    input_image, _, ratio = resize_and_pad_image(image, jitter=None)
    input_image = tf.keras.applications.resnet.preprocess_input(input_image)
    input_image = tf.expand_dims(input_image, axis=0)

    return image, gt_boxes, gt_labels, input_image, ratio



#%%
datasets, labels = load_dataset()
# def build_dataset(datasets, batch_size):
    autotune = tf.data.AUTOTUNE
    label_encoder = LabelEncoder()
    (train_set, valid_set, test_set) = datasets

    train_set = train_set.map(
        lambda x, y, w, z:
            tf.py_function(
                retina_preprocess,
                [x, y, w, z, -1., False],
                [tf.float32, tf.float32, tf.int32]
                )
            )

    valid_set = valid_set.map(
        lambda x, y, w, z:
            tf.py_function(
                retina_preprocess,
                [x, y, w, z, -1., False],
                [tf.float32, tf.float32, tf.int32]
                )
            )

    test_set = test_set.map(
        lambda x, y, w, z:
            tf.py_function(
                retina_preprocess,
                [x, y, w, z, -1., False],
                [tf.float32, tf.float32, tf.int32]
                )
            )

    train_set = train_set.map(preprocess_train, num_parallel_calls=autotune)
    train_set = train_set.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    train_set = train_set.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    ).repeat()
    train_set = train_set.apply(tf.data.experimental.ignore_errors())
    train_set = train_set.prefetch(autotune)

    valid_set = valid_set.map(preprocess_test, num_parallel_calls=autotune).repeat()
    valid_set = valid_set.apply(tf.data.experimental.ignore_errors())
    valid_set = valid_set.prefetch(autotune)

    test_set = test_set.map(preprocess_test, num_parallel_calls=autotune).repeat()
    test_set = test_set.apply(tf.data.experimental.ignore_errors())
    test_set = test_set.prefetch(autotune)

    train_set = iter(train_set)
    valid_set = iter(valid_set)
    test_set = iter(test_set)


valid_set = valid_set.map(
    lambda x, y, w, z:
        tf.py_function(
            retina_preprocess,
            [x, y, w, z, -1., False],
            [tf.float32, tf.float32, tf.int32]
            )
        )
from tqdm import tqdm
gt_list = []
tmp = iter(train_set)
for _ in tqdm(range(6033)):
    img, gt_boxes, gt_labels, filename = next(tmp)
    # aug_img, edge = fusion_edge(img, 50)
    gt = tf.concat([gt_boxes, tf.expand_dims(tf.cast(gt_labels, dtype=tf.float32), axis=-1)], axis=-1)
    gt_list.append(gt)
total_gt = tf.concat(gt_list, axis=0)

select_gt = tf.gather(total_gt, tf.where(total_gt[..., -1] != 9.))
select_hw = tf.stack([select_gt[...,2] - select_gt[...,0], select_gt[...,3] - select_gt[...,1]], axis=-1)
select_hw = tf.squeeze(select_hw, axis=1)


#%%


img = tf.keras.utils.array_to_img(img)
img.save("./img2.pdf")
edge = tf.keras.utils.array_to_img(_)
edge.save("./edge2.pdf")

tf.keras.utils.array_to_img(img)
tf.keras.utils.array_to_img(augmented_img)

import pickle
with open("/Volumes/LaCie/data/해상 객체 이미지/train.pickle", "rb") as f:
    contents = pickle.load(f)
len(contents)

fusionimg

def build_box_prior(dataset, img_size, data_num, k_per_grid):
    gt_hws = collect_boxes(dataset, data_num, img_size)
    hw_area = gt_hws[..., 0] * gt_hws[..., 1]
    hw1 = gt_hws[hw_area <= np.quantile(hw_area, 0.333333)]
    hw2 = gt_hws[
        np.logical_and(
            hw_area > np.quantile(hw_area, 0.333333),
            hw_area <= np.quantile(hw_area, 0.666666),
        )
    ]
    hw3 = gt_hws[hw_area > np.quantile(hw_area, 0.666666)]

    box_prior = []
    progress = tqdm(range(3))
    progress.set_description("Clustering boxes")
    for i in progress:
        cluster_sample = (hw1, hw2, hw3)[i]
        box_prior.append(k_means(cluster_sample, k_per_grid))
    box_prior = np.concatenate(box_prior)

    prior_df = pd.DataFrame(box_prior, columns=["height", "width"])
    prior_df.insert(2, "area", box_prior[..., 0] * box_prior[..., 1])
    prior_df = prior_df.sort_values(by=["area"], axis=0)
    prior_df = prior_df[["height", "width"]]

    return prior_df

def k_means(cluster_sample, k_per_grid):
    model = KMeans(n_clusters=k_per_grid, random_state=1)
    model.fit(cluster_sample)
    box_prior = model.cluster_centers_

    return box_prior


model = KMeans(n_clusters=9, random_state=1)
model.fit(select_hw)
box_prior = model.cluster_centers_

ship_box = box_prior * 512
other_box = box_prior * 512
all_box = box_prior * 512

labels = total_gt[...,-1]

import matplotlib.pyplot as plt
from PIL import ImageDraw, Image


def draw_box(boxes):
    box_xy = tf.stack([
        270 - boxes[..., 0] / 2 ,
        480 - boxes[..., 1] / 2 ,
        270 + boxes[..., 0] / 2 ,
        480 + boxes[..., 1] / 2 ,
        ], axis=-1) 

    canvas = Image.new("RGB", (960, 540), color="#fff")
    draw = ImageDraw.Draw(canvas)
    for box in box_xy:
        y1, x1, y2, x2 = tf.split(box, 4, axis=-1)
        draw.rectangle((x1, y1, x2, y2), outline=(1,0,0,1), width=3)
    
    return canvas

canvas_ship = draw_box(ship_box)
canvas_other = draw_box(other_box)
canvas_all = draw_box(all_box)
canvas_ship.save("./ship_boxes.pdf")
canvas_other.save("./other_boxes.pdf")
canvas_all.save("./all_boxes.pdf")

    image = tf.squeeze(image, axis=0)
    image = tf.keras.preprocessing.image.array_to_img(image)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    y1 = final_bboxes[0][..., 0] * height
    x1 = final_bboxes[0][..., 1] * width
    y2 = final_bboxes[0][..., 2] * height
    x2 = final_bboxes[0][..., 3] * width

    denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

    colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)

    for index, bbox in enumerate(denormalized_box):
        y1, x1, y2, x2 = tf.split(bbox, 4, axis=-1)
        width = x2 - x1
        height = y2 - y1

        final_labels_ = tf.reshape(final_labels[0], shape=(200,))
        final_scores_ = tf.reshape(final_scores[0], shape=(200,))
        label_index = int(final_labels_[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0} {1:0.3f}".format(labels[label_index], final_scores_[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    return image

import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 10))
sns.histplot(labels, stat="density", ax=ax)
fig.savefig("./label_dist.pdf")

tf.keras.utils.array_to_img(img)
tf.keras.utils.array_to_img(tf.image.adjust_contrast(img, 10))
a = tf.image.rgb_to_grayscale(img)
b = tf.image.adjust_contrast(a, 10)
tf.keras.utils.array_to_img(b)

from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
model = VGG16(include_top=False, input_shape=[None, None, 3])

while True:
    img, gt_boxes, gt_labels, filename = next(tmp)
    if not all((gt_labels == 1).numpy()):break
augmented_img, _ = fusion_edge(img, 50)
edge  = tf.concat([_, _, _], axis= -1)
tf.keras.utils.array_to_img(b)
output = model.predict(tf.expand_dims(edge * 255, axis=0))
output_img = model.predict(tf.expand_dims(img, axis=0))
output_cont = model.predict(tf.expand_dims(tf.concat([b,b,b], axis=-1), 0))
output_aug = model.predict(tf.expand_dims(img+edge, 0))

tf.keras.utils.array_to_img(img)
tf.keras.utils.array_to_img(edge)
plt.imshow(tf.reduce_sum(output[0], axis=-1))
plt.imshow(tf.reduce_sum(output_cont[0], axis=-1))
plt.imshow(tf.reduce_sum(output_img[0], axis=-1))
plt.imshow(tf.reduce_sum(output[0] + output_img[0], axis=-1))
plt.imshow(tf.reduce_sum(output[0] + output_cont[0] + output_img[0], axis=-1))
plt.imshow(tf.reduce_sum(output_aug[0], axis=-1))

from tensorflow.keras.applications.resnet50 import ResNet50
model = ResNet50(include_top=False, input_shape=[None, None, 3])
