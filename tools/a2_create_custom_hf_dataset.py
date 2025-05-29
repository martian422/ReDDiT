from datasets import Dataset, DatasetDict
import numpy as np
import os
from tqdm import tqdm
import json
import random

def load_npy_files(text_dir):
    data=[]

    subfolders = sorted(
            [d for d in os.listdir(text_dir) if os.path.isdir(os.path.join(text_dir, d)) and d.isdigit()]
        )
    selected_folders = subfolders

    for folder in tqdm(selected_folders):
        folder_path = os.path.join(text_dir, folder)  # Path to the subfolder
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith('.npy') : ### add filter here
                    npy_file = os.path.join(root, file_name)
                    if os.path.exists(npy_file):
                        caption=root.split('/')[-1]
                        token_file = os.path.join(folder,file_name)
                        data.append({
                            'text': [caption],
                            'image_tokens': token_file
                        })
                    else:
                        print(f"File not found, skipping: {npy_file}") 

    ran_data = random.sample(data, len(data))
    print(len(ran_data))
    breakpoint()
    return ran_data

text_dir = '/nfs/mtr/datasets/imagenet_adm_code_llamagenf8'

recap_data = load_npy_files(text_dir = text_dir)

my_dict = {
    'text': [x['text'] for x in recap_data],# 1,L, 1280
    'image_tokens': [x['image_tokens'] for x in recap_data] # 256
}


train_dataset = Dataset.from_dict(my_dict)

data_dir = "/nfs/mtr/datasets/dataset_files/imagenet-adm-code"

train_dataset.save_to_disk(data_dir)

from datasets import load_from_disk

loaded_dataset = load_from_disk(data_dir)
