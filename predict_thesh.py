#%%
import tensorflow as tf
import ship

#%%
data_dir = "C:\won\data\optimal_threshold\train.tfrecord"
dataset = tf.data.TFRecordDataset(f"{data_dir}".encode("unicode_escape")).map(ship.deserialize_feature)

dataset = iter(dataset)
print(dataset)
filename, feature_map, dtn_reg_output, dtn_cls_output, best_threshold = next(dataset)