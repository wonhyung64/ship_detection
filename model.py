import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, TimeDistributed, Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from typing import Dict, List
from PIL import ImageDraw
import argparse


class ShipDetector():
    def __init__(self, weights_dir):
        self.args = build_args()
        self.labels = ["bg", "buoy", "ship", "other floats"]
        self.total_labels = len(self.labels)
        self.rpn_model, self.dtn_model = build_models(self.args, self.total_labels)
        self.rpn_model.load_weights(f"{weights_dir}_rpn.h5")
        self.dtn_model.load_weights(f"{weights_dir}_dtn.h5")
        self.anchors = build_anchors(self.args)

    def predict(self, image):
        rpn_reg_output, rpn_cls_output, feature_map = self.rpn_model(image)
        roi_bboxes, roi_scores = RoIBBox(rpn_reg_output, rpn_cls_output, self.anchors, self.args)
        pooled_roi = RoIAlign(roi_bboxes, feature_map, self.args)
        dtn_reg_output, dtn_cls_output = self.dtn_model(pooled_roi)
        final_bboxes, final_labels, final_scores = Decode(
            dtn_reg_output, dtn_cls_output, roi_bboxes, self.args, self.total_labels
        )

        return final_bboxes, final_labels, final_scores


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="vgg16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", nargs="+", type=int, default=[500, 500])
    parser.add_argument("--feature-map-shape", nargs="+", type=int, default=[31, 31])
    parser.add_argument(
        "--anchor-ratios", nargs="+", type=float, default=[1.0, 2.0, 1.0 / 2.0]
    )
    parser.add_argument(
        "--anchor-scales", nargs="+", type=float, default=[64, 128, 256]
    )
    parser.add_argument("--pre-nms-topn", type=int, default=6000)
    parser.add_argument("--train-nms-topn", type=int, default=1500)
    parser.add_argument("--test-nms-topn", type=int, default=300)
    parser.add_argument("--total-pos-bboxes", type=int, default=128)
    parser.add_argument("--total-neg-bboxes", type=int, default=128)
    parser.add_argument("--pooling-size", nargs="+", type=int, default=[7, 7])
    parser.add_argument(
        "--variances", nargs="+", type=float, default=[0.1, 0.1, 0.2, 0.2]
    )
    parser.add_argument("--pos-threshold", type=float, default=0.65)
    parser.add_argument("--neg-threshold", type=float, default=0.25)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    return args


def build_models(args, total_labels):
    rpn_model = RPN(args)
    input_shape = [None] + args.img_size + [3]
    rpn_model.build(input_shape)

    dtn_model = DTN(args, total_labels)
    input_shape = [None, args.train_nms_topn, 7, 7, 512]
    dtn_model.build(input_shape)

    return rpn_model, dtn_model


class RPN(Model):
    def __init__(self, args) -> None:
        """
        parameters

        Args:
            hyper_params (Dict): hyper parameters
        """
        super(RPN, self).__init__()
        self.args = args
        self.shape = args.img_size + [3]
        self.anchor_counts = len(self.args.anchor_ratios) * len(self.args.anchor_scales)
        if args.base_model == "vgg16":
            self.base_model = VGG16(
                include_top=False,
                input_shape=self.shape,
            )
        elif args.base_model == "vgg19":
            self.base_model = VGG19(
                include_top=False,
                input_shape=self.shape,
            )
        self.layer = self.base_model.get_layer("block5_conv3").output

        self.feature_extractor = Model(inputs=self.base_model.input, outputs=self.layer)
        self.feature_extractor.trainable = False

        self.conv = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            name="rpn_conv",
        )

        self.rpn_cls_output = Conv2D(
            filters=self.anchor_counts,
            kernel_size=(1, 1),
            activation="sigmoid",
            name="rpn_cls",
        )

        self.rpn_reg_output = Conv2D(
            filters=self.anchor_counts * 4,
            kernel_size=(1, 1),
            activation="linear",
            name="rpn_reg",
        )

    @tf.function
    def call(self, inputs: tf.Tensor) -> List:
        """
        batch of images pass RPN

        Args:
            inputs (tf.Tensor): batch of images

        Returns:
            List: list of RPN reg, cls, and feature map
        """
        feature_map = self.feature_extractor(inputs)
        x = self.conv(feature_map)
        rpn_reg_output = self.rpn_reg_output(x)
        rpn_cls_output = self.rpn_cls_output(x)

        return [rpn_reg_output, rpn_cls_output, feature_map]


class DTN(Model):
    def __init__(self, args, total_labels) -> None:
        """
        parameters

        Args:
            hyper_params (Dict): hyper parameters
        """
        super(DTN, self).__init__()
        self.args = args
        self.total_labels = total_labels
        #
        self.FC1 = TimeDistributed(Flatten(), name="frcnn_flatten")
        self.FC2 = TimeDistributed(Dense(4096, activation="relu"), name="frcnn_fc1")
        self.FC3 = TimeDistributed(Dropout(0.5), name="frcnn_dropout1")
        self.FC4 = TimeDistributed(Dense(4096, activation="relu"), name="frcnn_fc2")
        self.FC5 = TimeDistributed(Dropout(0.5), name="frcnn_dropout2")
        #
        self.cls = TimeDistributed(
            Dense(self.total_labels, activation="softmax"),
            name="frcnn_cls",
        )
        self.reg = TimeDistributed(
            Dense(self.total_labels * 4, activation="linear"),
            name="frcnn_reg",
        )

    @tf.function
    def call(self, inputs: tf.Tensor) -> List:
        """
        pass detection network

        Args:
            inputs (tf.Tensor): pooled RoI

        Returns:
            List: list of detection reg, cls outputs
        """
        fc1 = self.FC1(inputs)
        fc2 = self.FC2(fc1)
        fc3 = self.FC3(fc2)
        fc4 = self.FC4(fc3)
        fc5 = self.FC5(fc4)
        dtn_reg_output = self.reg(fc5)
        dtn_cls_output = self.cls(fc5)

        return [dtn_reg_output, dtn_cls_output]




def build_anchors(args) -> tf.Tensor:
    """
    generate reference anchors on grid

    Args:
        hyper_params (Dict): hyper parameters

    Returns:
        tf.Tensor: anchors
    """
    grid_map = build_grid(args.feature_map_shape[0])

    base_anchors = []
    for scale in args.anchor_scales:
        scale /= args.img_size[0]
        for ratio in args.anchor_ratios:
            w = tf.sqrt(scale**2 / ratio)
            h = w * ratio
            base_anchors.append([-h / 2, -w / 2, h / 2, w / 2])

    base_anchors = tf.cast(base_anchors, dtype=tf.float32)

    anchors = tf.reshape(base_anchors, (1, -1, 4)) + tf.reshape(grid_map, (-1, 1, 4))

    anchors = tf.reshape(anchors, (-1, 4))
    anchors = tf.clip_by_value(t=anchors, clip_value_min=0, clip_value_max=1)

    return anchors


def build_grid(feature_map_shape):

    stride = 1 / feature_map_shape

    grid_coords_ctr = tf.cast(
        tf.range(0, feature_map_shape) / feature_map_shape + stride / 2,
        dtype=tf.float32,
    )

    grid_x_ctr, grid_y_ctr = tf.meshgrid(grid_coords_ctr, grid_coords_ctr)

    flat_grid_x_ctr, flat_grid_y_ctr = tf.reshape(grid_x_ctr, (-1,)), tf.reshape(
        grid_y_ctr, (-1,)
    )

    grid_map = tf.stack(
        [flat_grid_y_ctr, flat_grid_x_ctr, flat_grid_y_ctr, flat_grid_x_ctr], axis=-1
    )

    return grid_map


def RoIBBox(
    rpn_reg_output,
    rpn_cls_output,
    anchors,
    args,
    nms_iou_threshold=0.7,
    test=False,
):
    pre_nms_topn = args.pre_nms_topn
    post_nms_topn = args.train_nms_topn
    if test == True:
        post_nms_topn = args.test_nms_topn
    variances = args.variances
    total_anchors = (
        args.feature_map_shape[0]
        * args.feature_map_shape[1]
        * len(args.anchor_ratios)
        * len(args.anchor_scales)
    )
    batch_size = tf.shape(rpn_reg_output)[0]

    rpn_reg_output = tf.reshape(rpn_reg_output, (batch_size, total_anchors, 4))
    rpn_cls_output = tf.reshape(rpn_cls_output, (batch_size, total_anchors))

    rpn_reg_output *= variances

    rpn_bboxes = delta_to_bbox(anchors, rpn_reg_output)

    _, pre_indices = tf.nn.top_k(rpn_cls_output, pre_nms_topn)

    pre_roi_bboxes = tf.gather(rpn_bboxes, pre_indices, batch_dims=1)
    pre_roi_probs = tf.gather(rpn_cls_output, pre_indices, batch_dims=1)

    pre_roi_bboxes = tf.reshape(pre_roi_bboxes, (batch_size, pre_nms_topn, 1, 4))
    pre_roi_probs = tf.reshape(pre_roi_probs, (batch_size, pre_nms_topn, 1))

    roi_bboxes, roi_scores, _, _ = tf.image.combined_non_max_suppression(
        pre_roi_bboxes,
        pre_roi_probs,
        max_output_size_per_class=post_nms_topn,
        max_total_size=post_nms_topn,
        iou_threshold=nms_iou_threshold,
    )

    return roi_bboxes, roi_scores


def delta_to_bbox(anchors: tf.Tensor, bbox_deltas: tf.Tensor) -> tf.Tensor:
    """
    transform bbox offset to coordinates

    Args:
        anchors (tf.Tensor): reference anchors
        bbox_deltas (tf.Tensor): bbox offset

    Returns
        tf.Tensor: bbox coordinates
    """
    all_anc_width = anchors[..., 3] - anchors[..., 1]
    all_anc_height = anchors[..., 2] - anchors[..., 0]
    all_anc_ctr_x = anchors[..., 1] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[..., 0] + 0.5 * all_anc_height

    all_bbox_width = tf.exp(bbox_deltas[..., 3]) * all_anc_width
    all_bbox_height = tf.exp(bbox_deltas[..., 2]) * all_anc_height
    all_bbox_ctr_x = (bbox_deltas[..., 1] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (bbox_deltas[..., 0] * all_anc_height) + all_anc_ctr_y

    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1

    return tf.stack([y1, x1, y2, x2], axis=-1)


def RoIAlign(roi_bboxes, feature_map, args):
    pooling_size = args.pooling_size
    batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]

    row_size = batch_size * total_bboxes

    pooling_bbox_indices = tf.tile(
        tf.expand_dims(tf.range(batch_size), axis=1), (1, total_bboxes)
    )
    pooling_bbox_indices = tf.reshape(pooling_bbox_indices, (-1,))
    pooling_bboxes = tf.reshape(roi_bboxes, (row_size, 4))

    pooled_roi = tf.image.crop_and_resize(
        feature_map, pooling_bboxes, pooling_bbox_indices, pooling_size
    )

    pooled_roi = tf.reshape(
        pooled_roi,
        (
            batch_size,
            total_bboxes,
            pooled_roi.shape[1],
            pooled_roi.shape[2],
            pooled_roi.shape[3],
        ),
    )

    return pooled_roi


def Decode(
    dtn_reg_output,
    dtn_cls_output,
    roi_bboxes,
    args,
    total_labels,
    max_total_size=200,
    score_threshold=0.7,
    iou_threshold=0.5,
):
    variances = args.variances

    dtn_reg_output = tf.reshape(dtn_reg_output, (1, -1, total_labels, 4))
    dtn_reg_output *= variances

    expanded_roi_bboxes = tf.tile(
        tf.expand_dims(roi_bboxes, -2), (1, 1, total_labels, 1)
    )

    pred_bboxes = delta_to_bbox(expanded_roi_bboxes, dtn_reg_output)

    pred_labels_map = tf.expand_dims(tf.argmax(dtn_cls_output, -1), -1)
    pred_labels = tf.where(
        tf.not_equal(pred_labels_map, 0), dtn_cls_output, tf.zeros_like(dtn_cls_output)
    )

    final_bboxes, final_scores, final_labels, _ = tf.image.combined_non_max_suppression(
        pred_bboxes,
        pred_labels,
        max_output_size_per_class=max_total_size,
        max_total_size=max_total_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )

    return final_bboxes, final_labels, final_scores


class Evaluate():
    def __init__(self):
        self.labels = ["bg", "buoy", "ship", "other floats"]
        self.total_labels = len(self.labels)
        self.colors = tf.constant([
            [255, 255, 255, 0],
            [0, 0, 255, 255],
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            ])
    
    def cal_ap(self, final_bboxes, final_labels, gt_boxes, gt_labels):
        ap50 = calculate_ap_const(final_bboxes, final_labels, gt_boxes, gt_labels, self.total_labels)

        return ap50

    def visualize(self, image, final_bboxes, final_labels, final_scores):
        fig = draw_dtn_output(
            image, final_bboxes, self.labels, final_labels, final_scores, self.colors
        )

        return fig


def calculate_ap_const(
    final_bboxes, final_labels, gt_boxes, gt_labels, total_labels, mAP_threshold=0.5
):
    AP = []
    for c in range(1, total_labels):
        if tf.math.reduce_any(final_labels == c) or tf.math.reduce_any(gt_labels == c):
            final_bbox = tf.expand_dims(final_bboxes[final_labels == c], axis=0)
            gt_box = tf.expand_dims(gt_boxes[gt_labels == c], axis=0)

            if final_bbox.shape[1] == 0 or gt_box.shape[1] == 0:
                ap = tf.constant(0.0)
            else:
                precision, recall = calculate_pr(final_bbox, gt_box, mAP_threshold)
                ap = calculate_ap_per_class(recall, precision)
            AP.append(ap)
    if AP == []:
        AP = 1.0
    else:
        AP = tf.reduce_mean(AP)
    return AP


def calculate_pr(final_bbox, gt_box, mAP_threshold):
    bbox_num = final_bbox.shape[1]
    gt_num = gt_box.shape[1]

    true_pos = tf.Variable(tf.zeros(bbox_num))
    for i in range(bbox_num):
        bbox = tf.split(final_bbox, bbox_num, axis=1)[i]

        iou = generate_iou(bbox, gt_box)

        best_iou = tf.reduce_max(iou, axis=1)
        pos_num = tf.cast(tf.greater(best_iou, mAP_threshold), dtype=tf.float32)
        if tf.reduce_sum(pos_num) >= 1:
            gt_box = gt_box * tf.expand_dims(
                tf.cast(1 - pos_num, dtype=tf.float32), axis=-1
            )
            true_pos = tf.tensor_scatter_nd_update(true_pos, [[i]], [1])
    false_pos = 1.0 - true_pos
    true_pos = tf.math.cumsum(true_pos)
    false_pos = tf.math.cumsum(false_pos)

    recall = true_pos / gt_num
    precision = tf.math.divide(true_pos, true_pos + false_pos)

    return precision, recall


def generate_iou(anchors: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    """
    calculate Intersection over Union

    Args:
        anchors (tf.Tensor): reference anchors
        gt_boxes (tf.Tensor): bbox to calculate IoU

    Returns:
        tf.Tensor: Intersection over Union
    """
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(anchors, 4, axis=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)

    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=-1)

    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1]))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))

    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(
        y_bottom - y_top, 0
    )

    union_area = (
        tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area
    )

    return intersection_area / union_area


def calculate_ap_per_class(recall, precision):
    interp = tf.constant([i / 10 for i in range(0, 11)])
    AP = tf.reduce_max(
        [tf.where(interp <= recall[i], precision[i], 0.0) for i in range(len(recall))],
        axis=0,
    )
    AP = tf.reduce_sum(AP) / 11
    return AP


def draw_dtn_output(
    image,
    final_bboxes,
    labels,
    final_labels,
    final_scores,
    colors,
):
    image = tf.squeeze(image, axis=0)
    final_bboxes = tf.squeeze(final_bboxes, axis=0)
    final_labels = tf.squeeze(final_labels, axis=0)
    final_scores = tf.squeeze(final_scores, axis=0)

    fg_bool = final_labels != 0.
    final_bboxes = final_bboxes[fg_bool]
    final_labels = final_labels[fg_bool]
    final_scores = final_scores[fg_bool]

    image = tf.keras.preprocessing.image.array_to_img(image)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    y1 = final_bboxes[..., 0] * height
    x1 = final_bboxes[..., 1] * width
    y2 = final_bboxes[..., 2] * height
    x2 = final_bboxes[..., 3] * width
    denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

    for index, bbox in enumerate(denormalized_box):
        y1, x1, y2, x2 = tf.split(bbox, 4, axis=-1)
        width = x2 - x1
        height = y2 - y1

        label_index = int(final_labels[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0} {1:0.3f}".format(labels[label_index], final_scores[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    return image
