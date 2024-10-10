import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import argparse
import os
import json
import tqdm
from PIL import Image

import hydra
import lightning as L
import dataloader_t2i as dataloader
import diffusion
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, root_folder, folder_range):
        self.img_path_list = []
        start, end = folder_range

        # Get sorted list of numeric subfolders
        subfolders = sorted(
            [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d)) and d.isdigit()]
        )
        
        # Filter subfolders based on the specified range
        selected_folders = [folder for folder in subfolders if start <= int(folder) <= end]

        for folder in selected_folders:
            folder_path = os.path.join(root_folder, folder)
            for file in os.listdir(folder_path):
                if file.endswith(('.jpg', '.png')):  # Add other formats if needed
                    img_path = os.path.join(folder_path, file)
                    self.img_path_list.append(img_path)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        # image = Image.open(img_path).convert("RGB")  # Load the image
        
        # Corresponding .txt file
        txt_file_path = os.path.splitext(img_path)[0] + '.txt'
        
        # Read the caption from the .txt file
        with open(txt_file_path, 'r') as txt_file:
            # caption = txt_file.read().strip()  # Read and strip whitespace
            caption = txt_file.read()  # Read 
        return caption, img_path

        

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    # # Setup DDP:
    # # dist.init_process_group("nccl")
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # device = rank % torch.cuda.device_count()
    # seed = args.global_seed * dist.get_world_size() + rank
    # torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    # print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # # Setup a feature folder:
    # if rank == 0:
    #     os.makedirs(args.output_path, exist_ok=True)

    # # Setup data:
    data_path='/home/MaTianren/Workspace/llamaGen/Recap-1B/recap_datacomp_1b'
    print(f"Dataset is preparing...")
    num_workers=24

    dataset = CustomDataset(root_folder=data_path, folder_range=[0,10])

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=0
    )

    loader = DataLoader(
        dataset,
        batch_size=1, # important!
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Dataset contains {len(dataset):,} images")


    tokenizer = dataloader.get_tokenizer(config)

    mdlm = diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        strict=False
        )
    mdlm.to(device)
    PAD_MAX = mdlm.lm.cls_token_num
    mdlm.eval()

    for caption, image_id in tqdm.tqdm(loader):
        caption = ['A white dog is lying on the ground next to a blue bicycle parked on the side of a cobblestone street. The street is lined with buildings and has a few pedestrians walking in the distance.']
        caption_embs, emb_masks = tokenizer.get_text_embeddings(caption)
        text_embeds = [caption_embs[t][:emb_masks[t].sum(),:] for t in range(caption_embs.shape[0])]
        text_embeds = torch.stack([F.pad(t[:PAD_MAX,:], (0, 0, max(PAD_MAX - t.size(0),0), 0)) for t in text_embeds])
        attention_mask = (text_embeds[:,:,0]!=0).to(torch.int)
        result = mdlm.generate_ar(text_embeds, attention_mask)
        # caption_embs, emb_masks = t5_xxl.get_text_embeddings(caption)
        # valid_caption_embs = caption_embs[:, :emb_masks.sum()]
        # x = valid_caption_embs.to(torch.float32).detach().cpu().numpy()
        # # os.makedirs(os.path.join(args.t5_path, 'Recap_COCO_30K'), exist_ok=True)
        # output_dataset_path = args.output_path
        # current_folder = os.path.join(output_dataset_path, image_id[0][-19:-14])
        # os.makedirs(current_folder, exist_ok=True)
        # save_path = os.path.join(current_folder, '{}-text.npy'.format(image_id[0][-13:-4]))
        # np.save(save_path, x)
        

    dist.destroy_process_group()


if __name__ == '__main__':
    main()

