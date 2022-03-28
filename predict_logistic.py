#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ship, model_utils
from numpy import mean, std
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

#%%
data_dir = "C:\won\data\optimal_threshold\train.tfrecord"
dataset = tf.data.TFRecordDataset(f"{data_dir}".encode("unicode_escape")).map(ship.deserialize_feature)
dataset = dataset.batch(1)
dataset = iter(dataset)

#%% generate data 
X = []
y = []
while True:
    try:
        _, feature_map, dtn_reg_output, dtn_cls_output, best_threshold = next(dataset)
        features = model_utils.VectorizeFeatures()([feature_map, dtn_reg_output, dtn_cls_output])
        best_threshold = tf.cast(best_threshold * 20 - 10 , dtype=tf.int32)
        X.append(features)
        y.append(best_threshold)
    except: break


X = tf.concat(X, axis=0).numpy()
y = tf.concat(y, axis=0).numpy().squeeze(axis=-1)
_ = plt.hist(y, bins="auto")

print(X.shape, y.shape)
print(Counter(y))

#%%
model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)

print("Mean Accuracy: %.3f (%.3f)" % (mean(n_scores), std(n_scores)))