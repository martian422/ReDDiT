import os

from tqdm import tqdm
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F

import dataloader_t2i as dataloader
import diffusion
import utils
import random

from llamaGen.vq_model import VQ_models

omegaconf.OmegaConf.register_new_resolver(
    'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
    'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config):
    if 'hf' in config.backbone:
        return diffusion.Diffusion(
            config).to('cuda')
    
    return diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        config=config,
        strict=False)

@torch.no_grad()
def generate_samples(config, logger):

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f'cuda:{local_rank}')

    rand_index = random.randint(1,999)
    seed = rand_index
    print(seed)
    torch.manual_seed(seed)

    target_path = '/home/node237/Code/mdlm-c2i/outputs/to_evaluate'
    save_path_images = os.path.join(target_path, config.eval.mark, 'images')
    save_path_codes = os.path.join(target_path, config.eval.mark, 'codes')

    if local_rank == 0:
        os.makedirs(save_path_images, exist_ok=True)
        os.makedirs(save_path_codes, exist_ok=True)
   
    vq_model = VQ_models["VQ-16"](
        codebook_size=16384,
        codebook_embed_dim=8)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(config.repa_loss.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])

    del checkpoint

    for p in vq_model.parameters():
        p.requires_grad = False

    print(f"image tokenizer is loaded")


    model = _load_from_checkpoint(config=config)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    assert not config.sampling.semi_ar

    eps=1e-5
    model.backbone.eval()
    model.noise.eval()
    bs = 6
    class_nums=np.arange(1000)

    logger.info('Generating samples.')

    for class_num in tqdm(class_nums):
        for i in [50]:
            num_steps = i
            timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
            x = model._sample_prior_XX(bs, model.config.model.length).to(model.device)
            # x.shape= bs 256
            dt = (1 - eps) / num_steps
            p_x0_cache = None

            labels=torch.tensor([[class_num]])
            text_embeds = model.lm.cls_embedding(labels.to(model.device)).repeat(bs,1,1)
            
            attention_mask = (text_embeds[:,:,0]!=0).to(torch.int).repeat(bs,1)

            for i in range(num_steps):
                t = timesteps[i] * torch.ones(
                    x.shape[0], 1, device=model.device)
                if model.sampler == 'ddpm_cache':
                    p_x0_cache, x_next = model._ddpm_caching_update_XX(
                        x, text_embeds, attention_mask, t, dt, p_x0=p_x0_cache)
                    if (not torch.allclose(x_next, x)
                            or model.time_conditioning):
                        # Disable caching
                        p_x0_cache = None
                    x = x_next
                else:
                    raise ValueError
            # final step for additional noise removal
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=model.device)
            _, x = model._ddpm_caching_update_XX(x, text_embeds, attention_mask, t, dt, p_x0=None)

            if x.max()>16383:
                x[x>16383] = 16383
                print(f'{class_num} has mistakes, corrected. Pay attention to this.')
            
            x_decode = vq_model.decode_code(x,[x.shape[0],8,16,16])
            x_decode = F.interpolate(x_decode, size=(256, 256), mode='bicubic')
            for k in range(bs):
                # if you want to save the code.
                # torch.save(x[k].unsqueeze(0), os.path.join(save_path_codes, f'gen-cfg-{config.generation_cfg}-c{class_num}_s{num_steps}_r{local_rank}_n{k}.pt'))
                save_image(x_decode[k], os.path.join(save_path_images, f"c-{class_num}-s{seed}-n{k}.png"), normalize=True, value_range=(-1, 1))

    return 0

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    """Main entry point for training."""
    # L.seed_everything(config.seed)
    logger = utils.get_logger(__name__)

    generate_samples(config, logger)


if __name__ == '__main__':
    main()
