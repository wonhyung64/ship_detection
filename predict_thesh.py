#%%
import tensorflow as tf
import ship, model_utils

#%%
data_dir = "C:\won\data\optimal_threshold\train.tfrecord"
dataset = tf.data.TFRecordDataset(f"{data_dir}".encode("unicode_escape")).map(ship.deserialize_feature)
dataset = iter(dataset)
filename, feature_map, dtn_reg_output, dtn_cls_output, best_threshold = next(dataset)

#%% sample data
filename = tf.random.uniform(shape=(16,), minval=0, maxval=50, dtype=tf.int32)
feature_map = tf.random.uniform(shape=(31, 31, 512), minval=0., maxval=1., dtype=tf.float32)
dtn_reg_output = tf.random.uniform(shape=(1500, 16), minval=0., maxval=1., dtype=tf.float32)
dtn_cls_output = tf.random.uniform(shape=(1500, 4), minval=0., maxval=1., dtype=tf.float32)
best_threshold = tf.random.uniform(shape=(1, ), minval=0., maxval=1., dtype=tf.float32)

filename = tf.expand_dims(filename, axis=0)
feature_map = tf.expand_dims(feature_map, axis=0)
dtn_reg_output = tf.expand_dims(dtn_reg_output, axis=0)
dtn_cls_output = tf.expand_dims(dtn_cls_output, axis=0)
best_threshold = tf.expand_dims(best_threshold, axis=0)

inputs = [feature_map, dtn_reg_output, dtn_cls_output]

#%%
model_utils.VectorizeFeatures()(inputs)

# %%
