import tensorflow as tf
from models.retinanet.module.target import LabelEncoder
from models.retinanet.module.bbox import convert_to_xywh, swap_xy
from models.retinanet.module.preprocess import resize_and_pad_image
from module.datasets.augment import pixel_scaling, fusion_edge


def retina_train(img, gt_boxes, gt_labels, filename, fusion_scale):
    
    img = pixel_scaling(img)
    if fusion_scale == -1.:
        output_img = img
    else:
        output_img, _ = fusion_edge(img, fusion_scale)
    
    image_shape = [512, 512]
    image_shape = tf.cast(image_shape, dtype=tf.float32)
    image = tf.image.resize(output_img, tf.cast(image_shape, dtype=tf.int32))

    gt_boxes = swap_xy(gt_boxes)
    gt_boxes = tf.stack(
        [
            gt_boxes[:, 0] * image_shape[1],
            gt_boxes[:, 1] * image_shape[0],
            gt_boxes[:, 2] * image_shape[1],
            gt_boxes[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    gt_boxes = convert_to_xywh(gt_boxes)

    return image, gt_boxes, gt_labels
    
    
def retina_eval(img, gt_boxes, gt_labels, filename, fusion_scale):
    img = pixel_scaling(img)
    if fusion_scale == -1.:
        output_img = img
    else:
        output_img, _ = fusion_edge(img, fusion_scale)
    
    input_image, _, ratio = resize_and_pad_image(output_img, jitter=None)
    input_image = tf.keras.applications.resnet.preprocess_input(input_image)
    input_image = tf.expand_dims(input_image, axis=0)

    return img, gt_boxes, gt_labels, input_image, ratio


def build_dataset(datasets, batch_size):
    autotune = tf.data.AUTOTUNE
    label_encoder = LabelEncoder()
    (train_set, valid_set, test_set) = datasets

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


def random_flip_horizontal(image, boxes):
    
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )

    return image, boxes
