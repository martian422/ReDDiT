# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
import argparse
import os
import json

from tools.distributed import init_distributed_mode
from tokenizer.llamagen_vq import VQ_models

import tqdm

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, root_folder, transform):
        self.img_path_list = []
        self.transform = transform
        Image.MAX_IMAGE_PIXELS = None
        # Get sorted list of numeric subfolders
        subfolders = sorted(
            [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d)) ]
        )
        
        # Filter subfolders based on the specified range
        selected_folders = subfolders

        for folder in selected_folders:
            folder_path = os.path.join(root_folder, folder)
            for file in os.listdir(folder_path):
                if file.endswith(('.jpg')) :  # Add other formats if needed
                    img_path = os.path.join(folder_path, file)
                    self.img_path_list.append(img_path)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        try:
            # Try opening the image
            img = Image.open(img_path).convert("RGB")

            if min(img.size) > 4096:
                # Calculate new size while maintaining the aspect ratio
                if img.size[0] < img.size[1]:  # Width < Height
                    new_size = (512, int(512 * img.size[1] / img.size[0]))
                else:  # Height <= Width
                    new_size = (int(512 * img.size[0] / img.size[1]), 512)
                
                # Resize the image
                img = img.resize(new_size, Image.LANCZOS)
                print('Image too big, resize before processing')

        except (IOError, OSError) as e:
            # Handle the exception (e.g., log the error or skip the image)
            print(f"Error loading image at {img_path}: {e}")
            
            # Optionally, return a placeholder image
            img = Image.new("RGB", (256, 256), color="white")
        if self.transform is not None:
            img = self.transform(img)

        return img, img_path
        
#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    # dist.init_process_group("nccl")
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.code_path, exist_ok=True)


    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint


    # Setup data:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    print(f"Dataset is preparing...")
    dataset = CustomDataset(root_folder=args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=1, # important!
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Dataset contains {len(dataset):,} images")

    # total = 0
    # os.makedirs(os.path.join(args.code_path, 'code256'), exist_ok=True)
    for img, img_dir in tqdm.tqdm(loader):
        img = img.to(device)
        with torch.no_grad():
            _, _, [_, _, indices] = vq_model.encode(img)
        codes = indices.reshape(img.shape[0], -1)
        x = codes.detach().cpu().numpy()    # (1, args.image_size//16 * args.image_size//16)

        output_dataset_path = args.code_path
        current_folder = os.path.join(output_dataset_path, img_dir[0].split('/')[-2].split('-')[-1])
        os.makedirs(current_folder, exist_ok=True)
        save_path = os.path.join(current_folder, '{}.npy'.format(img_dir[0].split('/')[-1].split('.')[0]))
        np.save(save_path, x)

        # total += dist.get_world_size()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='/nfs/mtr/datasets/imagenet_10crops_x256')
    parser.add_argument("--code-path", type=str, default='/nfs/mtr/datasets/imagenet_10crop_code_llamagenf8')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-8")
    parser.add_argument("--vq-ckpt", type=str, default='/nfs/mtr/pretrained/vq_ds8_c2i.pt', help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=64)
    args = parser.parse_args()
    main(args)
