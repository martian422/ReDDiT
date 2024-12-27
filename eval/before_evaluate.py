# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py


from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse

def create_npz_from_sample_folder(sample_dir_o, num=48000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    sample_dir = os.path.join(sample_dir_o, 'images')
    samples = []
    for file in tqdm(os.listdir(sample_dir)):
        if file.endswith(('.png')) :
            sample_pil = Image.open(f"{sample_dir}/{file}")
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    sample_name = sample_dir_o.split('/')[-1] + '.npz'
    npz_path = os.path.join(sample_dir_o, sample_name)
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

create_npz_from_sample_folder('/home/node237/Code/ddit-c2i/outputs/to_evaluate/ddit-e107-s100-const25')