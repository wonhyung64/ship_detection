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

file_dir = args.file_dir
# file_dir = "/media/optim1/Data/won/ship"

#%%
run = neptune.init(
project=NEPTUNE_PROJECT,
api_token=NEPTUNE_API_KEY,
mode="async",
run="MOD2-158"
)
run["model"].download("./model_weights/retinanet/MOD2-158.h5")
run.stop()

datasets, labels = load_dataset(data_dir="/Volumes/LaCie/data")
train_num, valid_num, test_num = load_data_num(
    "ship", "/Volumes/LaCie/data", datasets[0], datasets[1], datasets[2]
    )
train_set, valid_set, test_set = build_dataset(datasets, 1, -1.)

colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)
model = build_model(len(labels))
model.load_weights("./model_weights/retinanet/MOD2-158.h5")
decoder = DecodePredictions(confidence_threshold=0.00)

# while True:
#     image, gt_boxes, gt_labels, input_image, ratio = next(test_set)
#     if any(gt_labels != 1): break

#%%
from models.retinanet.module.ap import calculate_pr, calculate_ap_per_class
from retina_utils import retina_eval

train_set = datasets[0]
autotune = tf.data.AUTOTUNE
train_set = train_set.map(
    lambda x, y, z, w:
        tf.py_function(
            retina_eval,
            [x, y, z, w, -1.],
            [tf.float32, tf.float32, tf.int32, tf.float32, tf.float32]
            )
        ).repeat()
train_set = train_set.apply(tf.data.experimental.ignore_errors())
train_set = train_set.prefetch(autotune)
train_set = iter(train_set)

total_labels = len(labels)
mAP_threshold = 0.5
AP = {
    "verytiny": {},
    "tiny": {},
    "small": {},
    "medium": {},
    "large": {},
    }
from models.retinanet.module.anchor import AnchorBox
# 1801
i = 0
for _ in tqdm(range(train_num)):
    image, gt_boxes, gt_labels, input_image, ratio = next(train_set)
    # if not all(gt_labels == 1):
    #     break
   
    predictions = model(input_image, training=False)
    scaled_bboxes, final_bboxes, final_scores, final_labels = decoder(input_image, predictions, ratio, tf.shape(image)[:2])
    fig1 = draw_gt(image, gt_boxes, gt_labels, labels, colors)
    fig2 = draw_output(image, final_bboxes[:10], final_labels[:10], final_scores[:10], labels, colors)
    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), dpi=300)
    axes[0].imshow(fig1)
    axes[0].set_axis_off()
    axes[1].imshow(fig2)
    axes[1].set_axis_off()
    f.tight_layout()
    
    f.savefig(f"./result/low_conf_pred{i}.png")
    i+=1

    pred_boxes = scaled_bboxes * tf.tile([749., 1333.], [2]) 
    true_boxes = gt_boxes * tf.tile([749., 1333.], [2])

    pred_verytiny = tf.reduce_prod(pred_boxes[...,2:] - pred_boxes[...,:2], -1) < 8.**2
    true_verytiny = tf.reduce_prod(true_boxes[...,2:] - true_boxes[...,:2], -1) < 8.**2

    pred_tiny = tf.logical_and(tf.reduce_prod(pred_boxes[...,2:] - pred_boxes[...,:2], -1) >= 8.**2, tf.reduce_prod(pred_boxes[...,2:] - pred_boxes[...,:2], -1) < 16.**2)
    true_tiny = tf.logical_and(tf.reduce_prod(true_boxes[...,2:] - true_boxes[...,:2], -1) >= 8.**2, tf.reduce_prod(true_boxes[...,2:] - true_boxes[...,:2], -1) < 16.**2)

    pred_small = tf.logical_and(tf.reduce_prod(pred_boxes[...,2:] - pred_boxes[...,:2], -1) >= 16.**2, tf.reduce_prod(pred_boxes[...,2:] - pred_boxes[...,:2], -1) < 32.**2)
    true_small = tf.logical_and(tf.reduce_prod(true_boxes[...,2:] - true_boxes[...,:2], -1) >= 16.**2, tf.reduce_prod(true_boxes[...,2:] - true_boxes[...,:2], -1) < 32.**2)

    pred_medium = tf.logical_and(tf.reduce_prod(pred_boxes[...,2:] - pred_boxes[...,:2], -1) >= 32.**2, tf.reduce_prod(pred_boxes[...,2:] - pred_boxes[...,:2], -1) < 96.**2)
    true_medium = tf.logical_and(tf.reduce_prod(true_boxes[...,2:] - true_boxes[...,:2], -1) >= 32.**2, tf.reduce_prod(true_boxes[...,2:] - true_boxes[...,:2], -1) < 96.**2)

    pred_large = tf.reduce_prod(pred_boxes[...,2:] - pred_boxes[...,:2], -1) >= 96.**2
    true_large = tf.reduce_prod(true_boxes[...,2:] - true_boxes[...,:2], -1) >= 96.**2

    for pred_idx, true_idx, key in [
        (pred_verytiny, true_verytiny, "verytiny"),
        (pred_tiny, true_tiny, "tiny"),
        (pred_small, true_small, "small"),
        (pred_medium, true_medium, "medium"),
        (pred_large, true_large, "large"),
        ]:

        selected_gt_boxes = gt_boxes[true_idx]
        selected_gt_labels = gt_labels[true_idx]
        selected_pred_boxes = scaled_bboxes[pred_idx]
        selected_pred_labels = final_labels[pred_idx]

        for c in range(total_labels):
            if tf.math.reduce_any(selected_pred_labels == c) or tf.math.reduce_any(selected_gt_labels == c):
                final_bbox = tf.expand_dims(selected_pred_boxes[selected_pred_labels == c], axis=0)
                gt_box = tf.expand_dims(selected_gt_boxes[selected_gt_labels == c], axis=0)

                if final_bbox.shape[1] == 0 or gt_box.shape[1] == 0:
                    ap = tf.constant(0.0)
                else:
                    precision, recall = calculate_pr(final_bbox, gt_box, mAP_threshold)
                    ap = calculate_ap_per_class(recall, precision)

                try:
                    AP[key][c].append(ap.numpy())
                except:
                    AP[key][c] = [ap.numpy()]

for k, v in AP["large"].items():
    print(f"{k}: {tf.reduce_mean(v)}")

rows = []
for size, dicts in AP.items():
    for label, aps in dicts.items():
        for ap in aps:
            rows.append([size, label, ap])

train_df = pd.DataFrame(
    data=rows,
    columns=["object_size", "label", "ap_50"],
)
train_df = train_df.sort_values(["object_size", "label"]).reset_index(drop=True)

train_df.to_csv("./result/train_pred_result.csv", index=False)


#%%


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
dataset = iter(datasets[0])
for split, dataset_num, dataset in [("valid", valid_num, valid_set), ("test", test_num, test_set)]:
    os.makedirs(f"{file_dir}/{split}", exist_ok=True)
    for num in tqdm(range(dataset_num)):
        image, gt_boxes, gt_labels, filename = next(dataset)
        idxs = gt_labels == 5
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
'''
#%%
indices = [0, 1, 2, 3, 4, 5]
columns = ["verytiny", "tiny", "small", "medium", "large"]

train_count = pd.DataFrame(index=indices, columns=columns)
train_sum = pd.DataFrame(index=indices, columns=columns)
train_mean = pd.DataFrame(index=indices, columns=columns)
train_std = pd.DataFrame(index=indices, columns=columns)

for i in train_df["label"].unique().tolist():
    for c in train_df["object_size"].unique().tolist():
        train_count.loc[i,c] = train_df[(train_df["label"] == i) & (train_df["object_size"] == c)]["ap_50"].count()
        train_sum.loc[i,c] = train_df[(train_df["label"] == i) & (train_df["object_size"] == c)]["ap_50"].sum()
        train_mean.loc[i,c] = train_df[(train_df["label"] == i) & (train_df["object_size"] == c)]["ap_50"].mean()
        train_std.loc[i,c] = train_df[(train_df["label"] == i) & (train_df["object_size"] == c)]["ap_50"].std()
        
test_count = pd.DataFrame(index=indices, columns=columns)
test_sum = pd.DataFrame(index=indices, columns=columns)
test_mean = pd.DataFrame(index=indices, columns=columns)
test_std = pd.DataFrame(index=indices, columns=columns)

for i in test_df["label"].unique().tolist():
    for c in test_df["object_size"].unique().tolist():
        test_count.loc[i,c] = test_df[(test_df["label"] == i) & (test_df["object_size"] == c)]["ap_50"].count()
        test_sum.loc[i,c] = test_df[(test_df["label"] == i) & (test_df["object_size"] == c)]["ap_50"].sum()
        test_mean.loc[i,c] = test_df[(test_df["label"] == i) & (test_df["object_size"] == c)]["ap_50"].mean()
        test_std.loc[i,c] = test_df[(test_df["label"] == i) & (test_df["object_size"] == c)]["ap_50"].std()
        
train_count
train_mean

ship_ap = AP["verytiny"][1] + AP["tiny"][1] + AP["small"][1] + AP["medium"][1] + AP["large"][1]
ship_df = pd.DataFrame(ship_ap)
ship_df_not_0_1 = ship_df[(ship_df[0] != 0.0) & (ship_df[0] != 1.0)]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
sns.histplot(ship_ap, ax=axes[0])
sns.histplot(ship_df_not_0_1[0].tolist(), ax=axes[1])
# plt.sca(axes[1])
# axes[1].legend("")
fig.tight_layout()
fig.savefig("./result/ship_ap_hist.png")

help(draw_hws)
fig = draw_hws(tf.constant([
    [8, 8],
    [16, 16],
    [32, 32],
    [64, 64],
    [96, 96],
    [539, 959]
], dtype=tf.float32))
fig.save("./result/box_criterion.png")