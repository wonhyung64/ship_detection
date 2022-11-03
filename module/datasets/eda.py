import tensorflow as tf
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans


def k_means(cluster_sample, k_per_grid):
    model = KMeans(n_clusters=k_per_grid, random_state=1)
    model.fit(cluster_sample)
    box_prior = model.cluster_centers_

    return box_prior


def draw_hws(boxes):
    box_xy = tf.stack([
        270 - boxes[..., 0] / 2 ,
        480 - boxes[..., 1] / 2 ,
        270 + boxes[..., 0] / 2 ,
        480 + boxes[..., 1] / 2 ,
        ], axis=-1) 

    canvas = Image.new("RGB", (960, 540), color="#fff")
    draw = ImageDraw.Draw(canvas)
    for box in box_xy:
        y1, x1, y2, x2 = tf.split(box, 4, axis=-1)
        draw.rectangle((x1, y1, x2, y2), outline=(1,0,0,1), width=3)
    
    return canvas
