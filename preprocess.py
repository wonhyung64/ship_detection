#%%
import re
import os
import xml
import pickle
from tqdm import tqdm

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

filePath = f'{path}/files.pickle'
with open(filePath, 'wb') as lf:
    pickle.dump(file_dir_list, lf)


with open(filePath, 'rb') as lf:
    readList = pickle.load(lf)
