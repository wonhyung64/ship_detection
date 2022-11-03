import cv2
import numpy as np
import tensorflow as tf

def fusion_edge(img, fusion_scale):
    bilateral = cv2.bilateralFilter(np.array(img), d=-1, sigmaColor=10, sigmaSpace=5)
    for _ in range(14):
        bilateral = cv2.bilateralFilter(bilateral, d=-1, sigmaColor=10, sigmaSpace=5)
    gaussian = cv2.GaussianBlur(bilateral, (5,5), 0)
    gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    threshold1, _ = cv2.threshold(gray.astype(np.uint8), -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    canny = cv2.Canny(
        gaussian.astype(np.uint8),
        threshold1=threshold1,
        threshold2=255,
        apertureSize=3,
        L2gradient=False
        )
    canny = tf.cast(tf.expand_dims(canny / 255, -1), dtype=tf.float32)
    augmented_img = tf.clip_by_value(canny*fusion_scale + img, tf.reduce_min(img), tf.reduce_max(img))

    return augmented_img, canny


def pixel_scaling(img):
    minimum = tf.math.minimum(tf.reduce_min(img), 0.)
    img = img - minimum
    maximum = tf.math.maximum(tf.reduce_max(img), 255.)
    img = img / maximum * 255.
    
    return img
