#%% 
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

import ship, etc_utils, model_utils, preprocessing_utils, postprocessing_utils, anchor_utils, test_utils

#%% 
hyper_params = etc_utils.get_hyper_params()
hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])

hyper_params["batch_size"] = batch_size = 1
img_size = (hyper_params["img_size"], hyper_params["img_size"])
dataset_name = hyper_params["dataset_name"]

dataset, labels = ship.fetch_dataset_v2(dataset_name, "train", img_size)
dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing_ship(x, y, z, w))
dataset = dataset.batch(1)
dataset = iter(dataset)

labels = ["bg"] + labels
hyper_params["total_labels"] = len(labels)

anchors = anchor_utils.generate_anchors(hyper_params)

#%%
weights_dir = os.getcwd() + "/frcnn_atmp"
weights_dir = weights_dir + "/" + os.listdir(weights_dir)[-1]

rpn_model = model_utils.RPN(hyper_params)
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)
rpn_model.load_weights(weights_dir + '/rpn_weights/weights')

dtn_model = model_utils.DTN(hyper_params)
input_shape = (None, hyper_params['train_nms_topn'], 7, 7, 512)
dtn_model.build(input_shape)
dtn_model.load_weights(weights_dir + '/dtn_weights/weights')

#%%
# save_dir = os.getcwd()
# save_dir = utils.generate_save_dir(save_dir, hyper_params)

total_time = []
mAP = []
optimal_threshold = []
threshold_lst = np.arange(0.5, 1.0, 0.05)
progress_bar = tqdm(range(10431))
writer = tf.io.TFRecordWriter(f'C:/won/data/optimal_threshold/train.tfrecord'.encode("utf-8"))

for _ in progress_bar:
    img, gt_boxes, gt_labels, filename = next(dataset)
    rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
    tf.squeeze(feature_map, axis=0)
    roi_bboxes, _ = postprocessing_utils.RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
    pooled_roi = postprocessing_utils.RoIAlign(roi_bboxes, feature_map, hyper_params)
    dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)

    best_threshold = 0.
    best_AP = 0.
    for threshold in threshold_lst:
        final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(dtn_reg_output, dtn_cls_output, roi_bboxes, hyper_params, iou_threshold=threshold)
        AP = test_utils.calculate_AP(final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params)
        if AP >= best_AP: 
            best_threshold = threshold
            best_AP = AP
    mAP.append(best_AP)
    optimal_threshold.append(best_threshold)

    feature_dic = {
        "filename":tf.squeeze(filename, axis=0),
        "feature_map":tf.squeeze(feature_map, axis=0),
        "dtn_reg_output":tf.squeeze(dtn_reg_output, axis=0),
        "dtn_cls_output": tf.squeeze(dtn_cls_output, axis=0),
        "best_threshold": tf.constant(best_threshold, tf.float32),
    }

    writer.write(ship.serialize_feature(feature_dic))

print("mAP: %.2f" % (tf.reduce_mean(mAP)))

