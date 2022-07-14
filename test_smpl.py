#%%
import os
import time
import tensorflow as tf
import neptune.new as neptune
from tqdm import tqdm
from models.faster_rcnn.utils import (
    build_args,
    plugin_neptune,
    NEPTUNE_API_KEY,
    NEPTUNE_PROJECT,
    build_anchors,
    build_models,
    record_result,
    RoIAlign,
    RoIBBox,
    Decode,
    draw_dtn_output,
)
from utils import (
    build_sample_set,
)

def main():
    args = build_args()
    os.makedirs("./data_chkr", exist_ok=True)
    run = plugin_neptune(NEPTUNE_API_KEY, NEPTUNE_PROJECT, args)

    weights_dir = "/home1/wonhyung64/data/ship_weights"

    test_set, labels = build_sample_set(args.name, args.data_dir, args.img_size)
    anchors = build_anchors(args)

    rpn_model, dtn_model = build_models(args, len(labels))

    rpn_model.load_weights(f"{weights_dir}_MOD_90_rpn.h5")
    dtn_model.load_weights(f"{weights_dir}_MOD_90_dtn.h5")

    mean_test_time = test(run, 360, test_set, rpn_model, dtn_model, labels, anchors, args)


def test(run, test_num, test_set, rpn_model, dtn_model, labels, anchors, args):
    test_times = []
    test_progress = tqdm(range(test_num))
    for step in test_progress:
        image = next(test_set)
        start_time = time.time()
        rpn_reg_output, rpn_cls_output, feature_map = rpn_model(image)
        roi_bboxes, roi_scores = RoIBBox(rpn_reg_output, rpn_cls_output, anchors, args)
        pooled_roi = RoIAlign(roi_bboxes, feature_map, args)
        dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
        final_bboxes, final_labels, final_scores = Decode(
            dtn_reg_output, dtn_cls_output, roi_bboxes, args, len(labels)
        )
        test_time = time.time() - start_time

        test_times.append(test_time)

        run["outputs/dtn"].log(
            neptune.types.File.as_image(
                draw_dtn_output(image, final_bboxes, labels, final_labels, final_scores)
            )
        )
    mean_test_time = tf.reduce_mean(test_times)

    return mean_test_time


if __name__ == "__main__":
    main()
