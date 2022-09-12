#%%
import tensorflow as tf
import os
import neptune.new as neptune
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
from utils.voucher import build_dataset

def extract(image, gt_boxes, gt_labels, filename): 
    return image, gt_boxes, gt_labels

#%%
if __name__ == "__main__":
    args = build_args()
    run = neptune.init(
        project=NEPTUNE_PROJECT,
        api_token=NEPTUNE_API_KEY,
        run="MOD-174"
    )
    args.data_dir = "/Users/wonhyung64/data/aihub_512_512"
    args.batch_size = 1

    train_set, valid_set, test_set = build_dataset(args)
    train_num , valid_num, test_num = 8000, 100, 1000
    labels = ["bg", "ship"]

    experiment_name = run.get_run_url().split("/")[-1].replace("-", "_")
    model_name = NEPTUNE_PROJECT.split("-")[1]
    experiment_dir = f"./model_weights/{model_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    weights_dir = f"{experiment_dir}/{experiment_name}"

    run["rpn_model"].download(f"{weights_dir}_rpn.h5")
    run["dtn_model"].download(f"{weights_dir}_dtn.h5")

    rpn_model, dtn_model = build_models(args, len(labels))
    rpn_model.load_weights(f"{weights_dir}_rpn.h5")
    dtn_model.load_weights(f"{weights_dir}_dtn.h5")
    anchors = build_anchors(args)

    test_progress = tqdm(range(30))
    colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)
    for step in test_progress:
        for _ in range(30):
            next(valid_set)
        image, gt_boxes, gt_labels = next(valid_set)
        rpn_reg_output, rpn_cls_output, feature_map = rpn_model(image)
        roi_bboxes, roi_scores = RoIBBox(rpn_reg_output, rpn_cls_output, anchors, args)
        pooled_roi = RoIAlign(roi_bboxes, feature_map, args)
        dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
        final_bboxes, final_labels, final_scores = Decode(
            dtn_reg_output, dtn_cls_output, roi_bboxes, args, len(labels)
        )

        result = draw_dtn_output(
            image, final_bboxes, labels, final_labels, final_scores, colors
        )
        ap50 = calculate_ap_const(final_bboxes, final_labels, gt_boxes, gt_labels, len(labels))
        print(ap50)
        result
        result.save(f"res_{step}.jpg")
# %%
