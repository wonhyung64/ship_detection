#%%
from utils import (
    load_dataset,
)
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
)
import cv2
from tqdm import tqdm

def extract(image, gt_boxes, gt_labels, filename): 
    return image, gt_boxes, gt_labels

#%%
if __name__ == "__main__":
    args = build_args()
    run = neptune.init(
        project=NEPTUNE_PROJECT,
        api_token=NEPTUNE_API_KEY,
        run="MOD-135",
    )
    datasets, labels, train_num, valid_num, test_num = load_dataset(
        name=args.name, data_dir=args.data_dir, img_size=args.img_size
    )
    train_set = datasets[0]
    train_set = iter(train_set.map(lambda x, y, z, w: extract(x, y, z, w)).batch(1))

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
    for step in test_progress:
        image, gt_boxes, gt_labels = next(train_set)
        rpn_reg_output, rpn_cls_output, feature_map = rpn_model(image)
        roi_bboxes, roi_scores = RoIBBox(rpn_reg_output, rpn_cls_output, anchors, args)
        pooled_roi = RoIAlign(roi_bboxes, feature_map, args)
        dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
        final_bboxes, final_labels, final_scores = Decode(
            dtn_reg_output, dtn_cls_output, roi_bboxes, args, len(labels)
        )


        result = draw_dtn_output(
            image, final_bboxes, labels, final_labels, final_scores
        )
        result.save(f"C:/Users/USER/Documents/GitHub/ship_detection/result/res_{step}.jpg")