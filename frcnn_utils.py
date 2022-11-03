import tensorflow as tf
from module.datasets.augment import pixel_scaling, fusion_edge
from models.faster_rcnn.utils.data_utils import resize_and_rescale, rand_flip_horiz


def frcnn_preprocess(image, gt_boxes, gt_labels, filename, fusion_scale, split):
    image = pixel_scaling(image)
    if fusion_scale == -1.:
        output_img = image
    else:
        output_img, _ = fusion_edge(image, fusion_scale)

    image = resize_and_rescale(output_img, [500, 500])
    if split == "train":
        image, gt_boxes = rand_flip_horiz(image, gt_boxes)
    gt_labels = tf.cast(gt_labels + 1, dtype=tf.int32)

    return image, gt_boxes, gt_labels


def build_dataset(datasets, batch_size, fusion_scale=-1.):
    train_set, valid_set, test_set = datasets
    data_shapes = ([None, None, None], [None, None], [None])
    padding_values = (
        tf.constant(0, tf.float32),
        tf.constant(0, tf.float32),
        tf.constant(-1, tf.int32),
    )

    train_set = train_set.map(
        lambda x, y, z, w:
            tf.py_function(
                frcnn_preprocess,
                [x, y, z, w, fusion_scale, "train"],
                [tf.float32, tf.float32, tf.int32]
                )
            )

    valid_set = valid_set.map(
        lambda x, y, z, w:
            tf.py_function(
                frcnn_preprocess,
                [x, y, z, w, fusion_scale, "valid"],
                [tf.float32, tf.float32, tf.int32]
                )
            )

    test_set = test_set.map(
        lambda x, y, z, w:
            tf.py_function(
                frcnn_preprocess,
                [x, y, z, w, fusion_scale, "test"],
                [tf.float32, tf.float32, tf.int32]
                )
            )

    train_set = train_set.padded_batch(
        batch_size,
        padded_shapes=data_shapes,
        padding_values=padding_values,
        drop_remainder=True,
    ).shuffle(6033).repeat()
    valid_set = valid_set.batch(1).repeat()
    test_set = test_set.batch(1).repeat()

    train_set = train_set.apply(tf.data.experimental.ignore_errors())
    valid_set = valid_set.apply(tf.data.experimental.ignore_errors())
    test_set = test_set.apply(tf.data.experimental.ignore_errors())

    train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE)
    valid_set = valid_set.prefetch(tf.data.experimental.AUTOTUNE)
    test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE)

    train_set = iter(train_set)
    valid_set = iter(valid_set)
    test_set = iter(test_set)

    return train_set, valid_set, test_set
