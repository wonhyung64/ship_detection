#%%
import re
import os
import xml
import pickle
from tqdm import tqdm
from utils.aihub_utils import *

file_dir_list = []
path = "/Volumes/LaCie/data/해상 객체 이미지/label"
region_list = [string for string in os.listdir(path) if not string.__contains__("._")]

for region in tqdm(region_list):
    date_list = os.listdir(f"{path}/{region}")

    for date in date_list: 
        folder_list = os.listdir(f"{path}/{region}/{date}")

        for folder in folder_list:
            file_list = [string for string in os.listdir(f"{path}/{region}/{date}/{folder}") if string.__contains__(".xml")]
            
            for file in file_list:
                file = re.sub(".xml", "", file)
                file_dir_list.append(f"{path}/{region}/{date}/{folder}/{file}")

#%%

with open(filePath, 'wb') as lf:
    pickle.dump(new_file_list, lf)


filePath = f'{path}/dir_25pix_ship_one.pickle'
with open(filePath, 'rb') as lf:
    total_file_list = pickle.load(lf)
    len(total_file_list)


label_dict = {}
new_file_list = []
folder_dirs = ["_".join(file_dir.split("_")[:-1]) for file_dir in file_dirs]
folder_dirs = set(folder_dirs)
dir_num = 0
progress = tqdm(folder_dirs)
for folder_dir in progress:
    progress.set_description(f"{dir_num}")
    leaf_dirs = [file_dir for file_dir in file_dirs if folder_dir in file_dir] 
    for leaf_dir in leaf_dirs:
        _, _, label_dict = extract_annot2(leaf_dir, label_dict, [3840, 2160])
        if all(_ == 2) == False: 
            new_file_list.append(leaf_dir)
            dir_num += 1
            break
len(var_file_list)
print(total_file_list)


    for file_dir in 
    folder_dir = "_".join(file_dir.split("_")[:-1])
    new_file_list.append(file_dir)

len(total_file_list)
len(var_file_list)
total_folder_list = ["_".join(total_file.split("_")[:-1]) for total_file in total_file_list]
var_folder_list = ["_".join(var_file.split("_")[:-1]) for var_file in var_file_list]
var_folder_list[0] in 
len(var_folder_list)
unq_total_folder_list = set(total_folder_list)
len(unq_total_folder_list)
remain_folder_list = [unq_total_folder for unq_total_folder in unq_total_folder_list if unq_total_folder not in var_folder_list]
len(remain_folder_list)
5289 + 2252
tmp = [total_file for total_file in  total_file_list if "_".join(total_file.split("_")[:-1]) in remain_folder_list]
len(tmp)

folder = ""
new_file_list = []
for t in tqdm(tmp):
    if folder == "_".join(t.split("_")[:-1]): continue
    folder = "_".join(t.split("_")[:-1]) 
    new_file_list.append(t)



new_file_list = []
label_dict = {}
for file_dir in tqdm(file_dirs):
    _, _, label_dict = extract_annot2(file_dir, label_dict, [3840, 2160])
    new_file_list.append(file_dir)
len(new_file_list)

    

#%%
def extract_annot2(sample, label_dict, org_img_size):
    tree = elemTree.parse(f"{sample}.xml")
    root = tree.getroot()
    bboxes_ = []
    labels_ = []
    for x in root:
        if x.tag == "object":
            for y in x:
                if y.tag == "bndbox":
                    bbox_ = [int(z.text) for z in y]
                    bbox = [
                        bbox_[1] / org_img_size[0],
                        bbox_[0] / org_img_size[1],
                        bbox_[3] / org_img_size[0],
                        bbox_[2] / org_img_size[1],
                    ]
                    bboxes_.append(bbox)
                if y.tag == "category_id":
                    label = int(y.text)
                    labels_.append(label)
                if y.tag == "name":
                    try:
                        label_dict[label] += 1
                    except:
                        label_dict[label] = 0
    bboxes = np.array(bboxes_, dtype=np.float32)
    labels = np.array(labels_, dtype=np.int32)

    return bboxes, labels, label_dict