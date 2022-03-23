#%% MODULE IMPORT
import tensorflow as tf
import etc_utils, ship

#%% 
hyper_params = etc_utils.get_hyper_params()
img_size = (hyper_params["img_size"], hyper_params["img_size"])

dataset, labels = ship.fetch_dataset("ship", "train", img_size)
dataset = dataset.prefetch(1)
dataset = iter(dataset)

i = 0
while True:
    i += 1
    print(i)
    # try: next(dataset)
    # except: continue
    next(dataset)
# %%
