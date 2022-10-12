#%%
import tensorflow as tf
from module.load import load_dataset

#%%
[train_set, valid_set, test_set], labels = load_dataset()
test_set = iter(test_set)
img, gt_boxes, gt_labels, filename = next(test_set)
tf.image.per_image_standardization(img / 255)
img = tf.image.per_image_standardization(img)
tf.keras.utils.array_to_img(img)
