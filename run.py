#%%
import cv2
import numpy as np
import tensorflow as tf
from module.load import load_dataset


def fit_retina():
    from retina_utils import build_dataset
    from models.retinanet.module.model import build_model, DecodePredictions
    from models.retinanet.module.neptune import record_result
    from models.retinanet.module.optimize import build_optimizer
    from models.retinanet.module.loss import RetinaNetBoxLoss, RetinaNetClassificationLoss
    from models.retinanet.module.utils import initialize_process, train, evaluate
    from models.retinanet.module.variable import NEPTUNE_API_KEY, NEPTUNE_PROJECT
    from models.retinanet.module.dataset import load_data_num

    args, run, weights_dir = initialize_process(NEPTUNE_API_KEY, NEPTUNE_PROJECT)
    datasets, labels = load_dataset(data_dir=args.data_dir)
    train_num, valid_num, test_num = load_data_num(
        args.name, args.data_dir, datasets[0], datasets[1], datasets[2]
        )
    train_set, valid_set, test_set = build_dataset(datasets, args.batch_size)
    colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)

    model = build_model(len(labels))
    decoder = DecodePredictions(confidence_threshold=0.5)
    box_loss_fn = RetinaNetBoxLoss(args.delta)
    clf_loss_fn = RetinaNetClassificationLoss(args.alpha, args.gamma)

    optimizer = build_optimizer(args.batch_size, train_num, args.momentum)

    train_time = train(run, args.epochs, args.batch_size,
        train_num, valid_num, train_set, valid_set, labels,
        model, decoder, box_loss_fn, clf_loss_fn, optimizer, weights_dir)

    model.load_weights(f"{weights_dir}.h5")
    mean_ap, mean_evaltime = evaluate(run, test_set, test_num, model, decoder, labels, "test", colors)
    record_result(run, weights_dir, train_time, mean_ap, mean_evaltime)

#%%
if __name__ == "__main__":
    fit_retina()

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


def build_dataset(datasets, batch_size, img_size):
    train_set, valid_set, test_set = datasets
    data_shapes = ([None, None, None], [None, None], [None])
    padding_values = (
        tf.constant(0, tf.float32),
        tf.constant(0, tf.float32),
        tf.constant(-1, tf.int32),
    )

    train_set = train_set.map(lambda x: frcnn_preprocess(x, split="train", img_size=img_size))
    test_set = test_set.map(lambda x: frcnn_preprocess(x, split="test", img_size=img_size))
    valid_set = valid_set.map(
        lambda x: preprocess(x, split="validation", img_size=img_size)
    )

    train_set = train_set.repeat().padded_batch(
        batch_size,
        padded_shapes=data_shapes,
        padding_values=padding_values,
        drop_remainder=True,
    )
    valid_set = valid_set.repeat().batch(1)
    test_set = test_set.repeat().batch(1)

    train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE)
    valid_set = valid_set.prefetch(tf.data.experimental.AUTOTUNE)
    test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE)

    train_set = iter(train_set)
    valid_set = iter(valid_set)
    test_set = iter(test_set)

    return train_set, valid_set, test_set

    args, run, weights_dir = initialize_process(
        NEPTUNE_API_KEY, NEPTUNE_PROJECT
    )
    datasets, labels = load_dataset(data_dir=args.data_dir)
    train_num, valid_num, test_num = load_data_num(
        args.name, args.data_dir, datasets[0], datasets[1], datasets[2]
        )
    


def build_dataset(datasets, batch_size, img_size):
    autotune = tf.data.AUTOTUNE
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
                retina_train,
                [x, y, z, w, -1.],
                [tf.float32, tf.float32, tf.int32]
                )
            )

    valid_set = valid_set.map(
        lambda x, y, z, w:
            tf.py_function(
                retina_eval,
                [x, y, z, w, -1.],
                [tf.float32, tf.float32, tf.int32, tf.float32, tf.float32]
                )
            )

    test_set = test_set.map(
        lambda x, y, z, w:
            tf.py_function(
                retina_eval,
                [x, y, z, w, -1.],
                [tf.float32, tf.float32, tf.int32, tf.float32, tf.float32]
                )
            )

    train_set = train_set.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), padded_shapes=([None, None, 3], [None, None], [None]), drop_remainder=True
    )
    train_set = train_set.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    ).shuffle(6050)

    train_set = train_set.repeat()
    valid_set = valid_set.repeat()
    test_set = test_set.repeat()

    train_set = train_set.apply(tf.data.experimental.ignore_errors())
    valid_set = valid_set.apply(tf.data.experimental.ignore_errors())
    test_set = test_set.apply(tf.data.experimental.ignore_errors())

    train_set = train_set.prefetch(autotune)
    valid_set = valid_set.prefetch(autotune)
    test_set = test_set.prefetch(autotune)

    train_set = iter(train_set)
    valid_set = iter(valid_set)
    test_set = iter(test_set)

    return train_set, valid_set, test_set