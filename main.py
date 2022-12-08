#%%
import os, time, sys, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import neptune.new as neptune
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import ImageDraw, Image
from datetime import datetime
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from retina_utils import build_dataset
from module.load import load_dataset
from module.datasets.eda import k_means, draw_hws
from models.retinanet.module.model import RetinaNet, DecodePredictions, get_backbone
from models.retinanet.module.variable import NEPTUNE_API_KEY, NEPTUNE_PROJECT
from models.retinanet.module.ap import calculate_ap_const
from models.retinanet.module.draw import draw_output
from models.retinanet.module.dataset import load_data_num
from models.retinanet.module.model import build_model


def feat2gray(feature_map):
    feature = tf.squeeze(feature_map, 0)
    gray = tf.expand_dims(tf.reduce_sum(feature, axis=-1), -1)

    return gray


def crop_img(image, gt_box):
    crop_box = tf.cast(gt_box * tf.tile([540., 960.], [2]), dtype=tf.int32)
    y1, x1, y2, x2 = tf.split(crop_box, 4)
    cropped_img = tf.image.crop_to_bounding_box(image, int(y1), int(x1), int(y2-y1), int(x2-x1))
    
    return cropped_img

parser = argparse.ArgumentParser()
parser.add_argument("--file-dir", type=str, default="/Volumes/LaCie/data/ship")

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

# file_dir = args.file_dir
file_dir = "/media/optim1/Data/won/ship"

#%%
'''
run = neptune.init(
project=NEPTUNE_PROJECT,
api_token=NEPTUNE_API_KEY,
mode="async",
run="MOD2-158"
)
run["model"].download("./model_weights/retinanet/MOD2-158.h5")
run.stop()
colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)
model = build_model(len(labels))
model.load_weights("./model_weights/retinanet/MOD2-158.h5")
decoder = DecodePredictions(confidence_threshold=0.5)

datasets, labels = load_dataset(data_dir="/Volumes/LaCie/data")
train_num, valid_num, test_num = load_data_num(
    "ship", "/Volumes/LaCie/data", datasets[0], datasets[1], datasets[2]
    )
train_set, valid_set, test_set = build_dataset(datasets, 2, -1.)
# while True:
#     image, gt_boxes, gt_labels, input_image, ratio = next(test_set)
#     if any(gt_labels != 1): break

#%%
image, gt_boxes, gt_labels, input_image, ratio = next(test_set)
predictions = model(input_image, training=False)
scaled_bboxes, final_bboxes, final_scores, final_labels = decoder(input_image, predictions, ratio, tf.shape(image)[:2])
ap = calculate_ap_const(scaled_bboxes, final_labels, gt_boxes, gt_labels, len(labels))
fig_pred = draw_output(image, final_bboxes, final_labels, final_scores, labels, colors)
fig_true = draw_gt(image, gt_boxes, gt_labels, labels, colors)

fig_pred.save(f"{save_dir}/pred{i}.pdf", quality=100)
fig_true.save(f"{save_dir}/true{i}.pdf", quality=100)

feat_extractor = get_backbone() 
c3, c4, c5 = feat_extractor(tf.expand_dims(image, 0))

#%%
fig, axes = plt.subplots(figsize = (30, 10), nrows=1, ncols=3)
axes[0].imshow(feat2gray(c3), cmap="gray")
axes[0].set_title("C3 Feature Map", fontsize=30)
axes[0].set_axis_off()
axes[1].imshow(feat2gray(c4), cmap="gray")
axes[1].set_title("C4 Feature Map", fontsize=30)
axes[1].set_axis_off()
axes[2].imshow(feat2gray(c5), cmap="gray")
axes[2].set_title("C5 Feature Map", fontsize=30)
axes[2].set_axis_off()
fig.tight_layout()

fig.savefig("./result/feature_maps.png")


fig, axes = plt.subplots(figsize = (20, 10), nrows=1, ncols=2)
axes[0].imshow(fig_pred)
axes[0].set_title("RetinaNet Prediction", fontsize=30)
axes[0].set_axis_off()
axes[1].imshow(fig_true)
axes[1].set_title("Ground-Truth", fontsize=30)
axes[1].set_axis_off()
fig.tight_layout()

fig.savefig("./result/result.png")

#%%
for split, dataset_num, dataset in [("valid", valid_num, valid_set), ("test", test_num, test_set)]:
    os.makedirs(f"{file_dir}/{split}", exist_ok=True)
    for num in tqdm(range(dataset_num)):
        image, gt_boxes, gt_labels, input_image, ratio = next(dataset)
        idxs = gt_labels != 1
        for idx, boolean in enumerate(idxs):
            if boolean.numpy():
                gt_box = gt_boxes[idx]
                cropped_img = crop_img(image, gt_box)
                gt_label = gt_labels[idx]

                filename = f'{labels[gt_label.numpy()].replace(" ", "_")}_{datetime.now().strftime("%H%M%S%f")}'
                np.save(f"{file_dir}/{split}/{filename}.npy", [cropped_img, gt_label], allow_pickle=True)
'''
#%% other label classification w resnet50
class CustomResNet50(Model):
    def __init__(self, **kwargs):
        super(CustomResNet50, self).__init__(**kwargs)
        self.backbone = ResNet50(include_top=False, input_shape=[None, None, 3])
        self.backbone.trainable = False
        self.pooling = GlobalAveragePooling2D()
        self.dense = Dense(5, activation="softmax")

    @tf.function 
    def call(self, image):
        feature_map = self.backbone(image)
        x = self.pooling(feature_map)
        x = self.dense(x)

        return x


classifier = CustomResNet50()
classifier.build(input_shape = [None, None, None, 3])
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)

train_dir = f"{file_dir}/valid"
test_dir = f"{file_dir}/test"
image, label = list(np.load(f"{train_dir}/{os.listdir(train_dir)[367]}", allow_pickle=True))

#%%
for epoch in range(1):
    train_progress = tqdm(os.listdir(train_dir))
    for file in train_progress:
        if not file.__contains__(".npy"):
            continue
        sample = f"{train_dir}/{file}"
        image, label = list(np.load(sample, allow_pickle=True))
        input_image = tf.expand_dims(image, 0)
        true = tf.expand_dims(tf.one_hot(label, 5), 0)
    
        with tf.GradientTape(persistent=True) as tape:
            pred = classifier(input_image)
            loss = tf.keras.losses.CategoricalCrossentropy()(true, pred)
        grads = tape.gradient(loss, classifier.trainable_weights)
        optimizer.apply_gradients(zip(grads, classifier.trainable_weights))

        train_progress.set_description(f"{epoch}/50 Epoch: loss - {loss.numpy()}")
    
    metrics = []
    for file in os.listdir(test_dir):
        if not file.__contains__(".npy"):
            continue
        sample = f"{test_dir}/{file}"
        image, label = list(np.load(sample, allow_pickle=True))
        input_image = tf.expand_dims(image, 0)
        with tf.device('/device:GPU:0'):
            pred = classifier.predict(input_image)
        onehot_pred = tf.one_hot(tf.argmax(pred, axis=-1), 5)
        metric = tf.keras.metrics.Accuracy()(true, onehot_pred)
        metrics.append(metric)

    accuracy = tf.reduce_mean([metrics])
    print(f"{epoch}: {accuracy}")
    classifier.save_weights(f"{file_dir}/classifier.h5")

classifier.load_weights(f"{file_dir}/classifier.h5")
test_set = iter(os.listdir(test_dir))

error_pred = []
error_true = []
error_image = []
correct_pred = []
correct_true = []
correct_image = []

while True: 
    image, label = list(np.load(f"{test_dir}/{next(test_set)}", allow_pickle=True))
    pred = tf.argmax(classifier(tf.expand_dims(image, 0))[0], -1).numpy()
    true = label.numpy()
    if pred != true:
        error_pred.append(pred)
        error_true.append(true)
        error_image.append(image)
    else:
        correct_pred.append(pred)
        correct_true.append(true)
        correct_image.append(image)
        
pd.Series(error_true).value_counts()
pd.Series(correct_true).value_counts()
classifier.get_layer("resnet50").output
imgs = []
for img in error_image:
    imgs.append(tf.shape(img)[:2].numpy())

hws = k_means(imgs, 9)
fig2 = draw_hws(hws)
fig1 = draw_hws(hws)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
axes[0].imshow(fig1)
axes[0].set_title("Correct Prediction Boxes", fontsize=20)
# axes[0].set_axis_off()
axes[1].imshow(fig2)
axes[1].set_title("Wrong Prediction Boxes", fontsize=20)
# axes[1].set_axis_off()
fig.tight_layout()

#%% 부표 feature_map 확인
image, label = list(np.load(f"{test_dir}/{next(test_set)}", allow_pickle=True))
samples = iter([sample for sample in  os.listdir(test_dir) if sample.__contains__("buoy") and not sample.__contains__("fishing")])
image, label = list(np.load(f"{test_dir}/{next(samples)}", allow_pickle=True))
feat_extractor = get_backbone() 
c3, c4, c5 = feat_extractor(tf.expand_dims(image, 0))
plt.imshow(feat2gray(c3))
tf.reduce_sum(c3)