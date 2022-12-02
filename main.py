#%%
import os, time, sys
import numpy as np
import tensorflow as tf
import neptune.new as neptune
from tqdm import tqdm
from models.retinanet.module.model import RetinaNet, DecodePredictions, get_backbone
from models.retinanet.module.variable import NEPTUNE_API_KEY, NEPTUNE_PROJECT
from module.load import load_dataset
from models.retinanet.module.ap import calculate_ap_const
from models.retinanet.module.draw import draw_output

from models.retinanet.module.dataset import load_data_num
from models.retinanet.module.model import build_model
from retina_utils import build_dataset
from PIL import ImageDraw
import matplotlib.pyplot as plt
from datetime import datetime

def feat2gray(feature_map):
    feature = tf.squeeze(feature_map, 0)
    gray = tf.expand_dims(tf.reduce_sum(feature, axis=-1), -1)

    return gray


def crop_img(image, gt_box):
    crop_box = tf.cast(gt_box * tf.tile([540., 960.], [2]), dtype=tf.int32)
    y1, x1, y2, x2 = tf.split(crop_box, 4)
    cropped_img = tf.image.crop_to_bounding_box(image, int(y1), int(x1), int(y2-y1), int(x2-x1))
    
    return cropped_img

#%%
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
file_dir = "/Volumes/LaCie/data/ship"

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

#%% other label classification w resnet50
