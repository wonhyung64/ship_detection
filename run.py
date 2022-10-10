#%%
import tensorflow as tf
from PIL import Image

#%%
path = "/Users/wonhyung64/Downloads/sample1.jpeg"
image = Image.open(path)
img_size = [size * 3 for size in image.size]
image = tf.keras.utils.img_to_array(image)
image = tf.image.resize(image, [img_size[1], img_size[0]], method="bicubic")
image = tf.keras.utils.array_to_img(image)
image.save("/Users/wonhyung64/Downloads/result1.jpeg")