#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import ship, model_utils
import joblib
from numpy import mean, std
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

#%%
data_dir = "C:\won\data\optimal_threshold\train.tfrecord"
dataset = tf.data.TFRecordDataset(f"{data_dir}".encode("unicode_escape")).map(ship.deserialize_feature)
dataset = dataset.batch(1)
dataset = iter(dataset)

test_dir = "C:\won\data\optimal_threshold\test.tfrecord"
dataset_test = tf.data.TFRecordDataset(f"{test_dir}".encode("unicode_escape")).map(ship.deserialize_feature)
dataset_test = dataset_test.batch(1)
dataset_test = iter(dataset_test)
#%%
data_dir = "C:\won\data\optimal_threshold\train_pooled_roi.tfrecord"
dataset = tf.data.TFRecordDataset(f"{data_dir}".encode("unicode_escape")).map(ship.deserialize_feature_v2)
dataset = dataset.batch(1)
dataset = iter(dataset)

test_dir = "C:\won\data\optimal_threshold\test_pooled_roi.tfrecord"
dataset_test = tf.data.TFRecordDataset(f"{test_dir}".encode("unicode_escape")).map(ship.deserialize_feature_v2)
dataset_test = dataset_test.batch(1)
dataset_test = iter(dataset_test)
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
test_X = []
test_y = []
while True:
    try:
        _, feature_map, dtn_reg_output, dtn_cls_output, best_threshold = next(dataset_test)
        features = model_utils.VectorizeFeatures()([feature_map, dtn_reg_output, dtn_cls_output])
        best_threshold = tf.cast(best_threshold * 20 - 10 , dtype=tf.int32)
        test_X.append(features)
        test_y.append(best_threshold)
    except: break


test_X = tf.concat(test_X, axis=0).numpy()
test_y = tf.concat(test_y, axis=0).numpy().squeeze(axis=-1)
test_y.shape

_ = plt.hist(test_y, bins="auto")

print(test_X.shape, test_y.shape)
print(Counter(test_y))

#%%
X = []
y = []
while True:
    try:
        _, feature_map, dtn_reg_output, dtn_cls_output, best_threshold = next(dataset)
        features = tf.reduce_sum(dtn_reg_output, axis=-1)
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
test_X = []
test_y = []
while True:
    try:
        _, feature_map, dtn_reg_output, dtn_cls_output, best_threshold = next(dataset_test)
        features = tf.reduce_sum(dtn_reg_output, axis=-1)
        best_threshold = tf.cast(best_threshold * 20 - 10 , dtype=tf.int32)
        test_X.append(features)
        test_y.append(best_threshold)
    except: break


test_X = tf.concat(test_X, axis=0).numpy()
test_y = tf.concat(test_y, axis=0).numpy().squeeze(axis=-1)
test_y.shape

_ = plt.hist(test_y, bins="auto")

print(test_X.shape, test_y.shape)
print(Counter(test_y))

#%%
X = []
y = []
while True:
    try:
        pooled_roi, best_threshold = next(dataset)
        features = pooled_roi
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
test_X = []
test_y = []
while True:
    try:
        pooled_roi, best_threshold = next(dataset_test)
        features = pooled_roi
        best_threshold = tf.cast(best_threshold * 20 - 10 , dtype=tf.int32)
        test_X.append(features)
        test_y.append(best_threshold)
    except: break


test_X = tf.concat(test_X, axis=0).numpy()
test_y = tf.concat(test_y, axis=0).numpy().squeeze(axis=-1)
test_y.shape

_ = plt.hist(test_y, bins="auto")

print(test_X.shape, test_y.shape)
print(Counter(test_y))


#%%
model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)

print("Mean Accuracy: %.3f (%.3f)" % (mean(n_scores), std(n_scores)))

model.fit(X, y)

filename = "logistic_model.sav"
joblib.dump(model, filename)

#%%
y_hat = model.predict(X)
accuracy_score(y, y_hat)

test_y_hat = model.predict(test_X)
accuracy_score(test_y, test_y_hat)
