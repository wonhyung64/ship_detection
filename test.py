#%%
import json
import time
from datetime import datetime
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import os
from models.faster_rcnn.utils import (
    build_models,
    build_anchors,
    build_args,
    NEPTUNE_API_KEY,
    NEPTUNE_PROJECT,
    RoIBBox,
    RoIAlign,
    Decode,
    draw_dtn_output,
    calculate_ap_const,
)
from tqdm import tqdm
from utils.voucher import build_dataset, decode_filename
from model import ShipDetector, Evaluate
import pandas as pd
import xml.etree.ElementTree as elemTree
import numpy as np


def annot2dict(current_time, annots):
    annotation = {}
    annotation["time"] = current_time
    for num, annot in enumerate(annots):
        annot_dict = {
        "bnd_xmin": str(annot[1].numpy()),
        "bnd_ymin": str(annot[0].numpy()),
        "bnd_xmax": str(annot[3].numpy()),
        "bnd_ymax": str(annot[2].numpy()),
        }
        annotation[f"name{num+1}"] = annot_dict

    return annotation


def dict2csv(annotation):
    annotation_csv = pd.DataFrame()
    for i in annotation.keys():
        if i == "time": continue
        df = pd.DataFrame.from_dict([annotation[i]])
        df[i[:4]] = i[4:]
        annotation_csv = pd.concat([annotation_csv, df], axis=0)
    annotation_csv["time"] = annotation["time"]
    annotation_csv = annotation_csv[["time", "name", "bnd_xmin", "bnd_ymin", "bnd_xmax", "bnd_ymax"]]

    return annotation_csv


def extract_annot(data_dir, sample, org_img_size):
    tree = elemTree.parse(f"{data_dir}/raw/{sample}.xml")
    root = tree.getroot()
    bboxes_ = []
    labels_ = []
    for x in root:
        if x.tag == "object":
            for y in x:
                if y.tag == "bndbox":
                    bbox_ = [int(z.text) for z in y]
                    bbox = [
                        bbox_[1] / org_img_size[0],
                        bbox_[0] / org_img_size[1],
                        bbox_[3] / org_img_size[0],
                        bbox_[2] / org_img_size[1],
                    ]
                    bboxes_.append(bbox)
                if y.tag == "category_id":
                    label = int(y.text)
                    labels_.append(label)
    bboxes = np.array(bboxes_, dtype=np.float32)
    labels = np.array(labels_, dtype=np.int32)
    gt_boxes = tf.expand_dims(tf.convert_to_tensor(bboxes), axis=0)
    gt_labels = tf.expand_dims(tf.convert_to_tensor(labels), axis=0)

    return gt_boxes, gt_labels

#%%
if __name__ == "__main__":
    args = build_args()
    args.data_dir = "/Users/wonhyung64/data/aihub_512_512"
    args.name = "aihub"

    train_set, valid_set, test_set = build_dataset(args)

    weights_dir = "/Users/wonhyung64/data/voucher/model"
    detector = ShipDetector(weights_dir)
    evaluate = Evaluate()
    save_dir = "/Users/wonhyung64/data/voucher"

#%%
    org_file = {}
    for num, dataset in [(950, valid_set), (950, test_set), (8000, train_set)]:
        progress = tqdm(range(num))
        for step in progress:
            image, gt_boxes, gt_labels, orgname = next(dataset)
            orgname = decode_filename(orgname)[0]
            idx = ((gt_boxes[...,2] - gt_boxes[...,0]) * 500) * ((gt_boxes[...,3] - gt_boxes[...,1]) * 500)  > 2500
            gt_boxes = tf.expand_dims(gt_boxes[idx], axis=0)
            gt_labels = tf.expand_dims(gt_labels[idx], axis=0)

            final_bboxes, final_labels, final_scores = detector.predict(image)
            ap_50 = evaluate.cal_ap(final_bboxes, final_labels, gt_boxes, gt_labels + 1)
            progress.set_description(f"ap50: {float(ap_50)}")
            res = evaluate.visualize(image, final_bboxes, final_labels, final_scores)
            if float(ap_50) > 0 and float(ap_50) < 1:
                current_time = datetime.now().strftime("%y%m%d%H%M%S")
                org_file[str(current_time)] = orgname
                filename = f"{current_time}_1"

                img = tf.keras.utils.array_to_img(tf.squeeze(image, 0))

                ants = tf.cast(tf.squeeze(gt_boxes, 0) * 500, dtype=tf.int32)
                annotation = annot2dict(current_time, ants)
                annotation_csv = dict2csv(annotation)

                preds = tf.cast(final_bboxes[final_labels != 0.] * 500, dtype=tf.int32)
                prediction = annot2dict(current_time, preds)
                prediction_csv = dict2csv(prediction)

                img.save(f"{save_dir}/train/{filename}_trn_img.jpg")
                with open(f"{save_dir}/train/{filename}_trn_ant.json", "w") as f:
                    json.dump(annotation, f)
                annotation_csv.to_csv(f"{save_dir}/train/{filename}_trn_ant.csv", index=False)
                with open(f"{save_dir}/train/{filename}_trn_pred.json", "w") as f:
                    json.dump(prediction, f)
                prediction_csv.to_csv(f"{save_dir}/train/{filename}_trn_pred.csv", index=False)
                res.save(f"{save_dir}/train/{filename}_trn_res.jpg")

                img.save(f"{save_dir}/test/{filename}_test_img.jpg")
                with open(f"{save_dir}/test/{filename}_test_ant.json", "w") as f:
                    json.dump(annotation, f)
                annotation_csv.to_csv(f"{save_dir}/test/{filename}_test_ant.csv", index=False)
                with open(f"{save_dir}/test/{filename}_test_pred.json", "w") as f:
                    json.dump(prediction, f)
                prediction_csv.to_csv(f"{save_dir}/test/{filename}_test_pred.csv", index=False)
                res.save(f"{save_dir}/test/{filename}_test_res.jpg")

    with open(f"{save_dir}/org_file.json", "w", encoding="utf-8") as f:
        json.dump(org_file, f, ensure_ascii=False)
