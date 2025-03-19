import os

from tqdm import tqdm
import hydra
import lightning as L
import omegaconf

import rich.tree
import torch
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F

import diffusion
import utils
import math


omegaconf.OmegaConf.register_new_resolver(
    'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
    'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config):
    
    return diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        config=config,
        strict=False)

@torch.no_grad()
def generate_samples(config, logger):

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f'cuda:{local_rank}')

    # print(seed)
    seed = 100 + config.seed
    torch.manual_seed(seed)
    sampler = config.sampling.predictor

    target_path = f'/nfs/mtr/code/ddit-c2i/outputs/eval-{sampler}'
    save_path_images = os.path.join(target_path, config.eval.mark, 'images')
    # save_path_codes = os.path.join(target_path, config.eval.mark, 'codes')

    if local_rank == 0:
        os.makedirs(save_path_images, exist_ok=True)
        # os.makedirs(save_path_codes, exist_ok=True)
    if config.vq == 'llamagen':
        from tokenizer.llamagen_vq import VQ_models

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

        print(f"tokenizer tokenizer is loaded")
        vocab = 16384 - 1
        decode_range = (-1, 1)

    elif config.vq == 'maskgit':
        from tokenizer.maskgit import PretrainedTokenizer

        vq_model = PretrainedTokenizer("/nfs/mtr/pretrained/maskgit-vqgan-imagenet-f16-256.bin")
    
        vq_model.eval()
        vq_model.requires_grad_(False)
        vq_model.to(device)
        print(f"MaskGIT tokenizer is loaded")

        vocab = 1024 - 1
        decode_range = (0, 1)

    elif config.vq == 'sdf8':
        from omegaconf import OmegaConf
        from tokenizer.ldm.util import instantiate_from_config
        ldm_config = OmegaConf.load('/nfs/mtr/pretrained/sd-vq-ds8/config.yaml')
        pl_sd = torch.load('/nfs/mtr/pretrained/sd-vq-ds8/model.ckpt', map_location="cpu")
        sd = pl_sd["state_dict"]
        vq_model = instantiate_from_config(ldm_config.model)
        vq_model.load_state_dict(sd, strict=False)
        vq_model.eval()
        vq_model.requires_grad_(False)
        vq_model = vq_model.to(device)
        decode_range = (-1, 1)
        vocab = 16384 - 1
        # raise ValueError("TBD, check it out!")

    elif config.vq == 'IBQ':
        from omegaconf import OmegaConf
        from tokenizer.IBQ.models.ibqgan import IBQ
        config = OmegaConf.load('/nfs/mtr/pretrained/IBQ-16384/imagenet_ibqgan_16384.yaml')
        vq_model = IBQ(**config.model.init_args)

        sd = torch.load('/nfs/mtr/pretrained/IBQ-16384/imagenet256_16384.ckpt', map_location="cpu")["state_dict"]
        missing, unexpected = vq_model.load_state_dict(sd, strict=False)

        vq_model.eval()
        vq_model.requires_grad_(False)
        vq_model = vq_model.to(device)
        decode_range = (-1, 1)
        vocab = 16384 - 1
        # raise ValueError("TBD, check it out!")
    else:
        raise ValueError("Unsupported tokenizer!")

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
    bs = 1
    if config.eval.mode=="all":
        class_nums=np.arange(1000)
    elif config.eval.mode=="sample":
        bs = 3
        class_nums = [1, 207, 360, 387, 88, 417, 279]
        # class_nums = [88]
    else:
        bs=1
        print('Not specified, have a look at our fish!')
        class_nums=[0,1]

    logger.info('Generating samples.')

    sample_step = config.sampling.steps

    for class_num in tqdm(class_nums):
        for i in [sample_step]:
            labels = torch.tensor([[class_num]]).repeat(bs,1).to(model.device)
            bs_all = bs
            num_steps = i
            timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
            # fake the cosine schedule during inference.
            if config.eval.timeline == 'slow-fast':
                timesteps = (math.pi/2 * timesteps).sin()
            elif config.eval.timeline=='fast-slow':
                timesteps = timesteps ** 1.25

            x = model._sample_prior_XX(bs_all, model.config.model.length).to(model.device)
            # x.shape= bs 256
            # dt = (1 - eps) / num_steps
            # p_x0_cache = None
            t_t = torch.ones(x.shape[0], 1, device=model.device)

            for i in range(num_steps):
                t_s = timesteps[i+1] * torch.ones(
                    x.shape[0], 1, device=model.device)
                if model.sampler == 'ddpm':
                    _, x_next = model._ddpm_update_v1(
                        x, labels, t_t, t_t - t_s, p_x0=None)
                    x = model.q_xt_hash(x_next)
                    t_t = t_s
                elif model.sampler == 'maskgit':
                    _, x_next = model._maskgit_update(
                        x, labels, t_t, t_t - t_s, logits_x0=None)
                    x = x_next
                    t_t = t_s
                elif model.sampler == 'flow_matching':
                    _, x_next = model._flow_matching_update(
                        x, labels, t_t, t_t - t_s, p_x0=None)
                    x = x_next
                    t_t = t_s
                else:
                    raise ValueError
            
            # final again for noise removal?
            if config.eval.timeline == 'slow-fast':
                final_dt = (t_t - eps).sin()
            elif config.eval.timeline == 'fast-slow':
                final_dt = (t_t - t_s + eps) ** 1.25
            elif config.eval.timeline == 'linear':
                final_dt = t_t - t_s + eps
            else:
                raise ValueError
            try:
                # in some cases, very rare tokens are still masked, so enforce the unmask schedule.
                # _, final_x = model._ddpm_update_final(x, labels, t_t, final_dt, p_x0=None)
                _, final_x = model._ddpm_update_final(x, labels, t_t, final_dt, p_x0=None)
                x = final_x
            except:
                print('Denoising error!')
                pass
            
            if x.max()>vocab:
                x[x>vocab] = vocab
                print(f'{class_num} has mistakes, corrected. Pay attention to this.')
            if config.vq == 'llamagen':
                x_decode = vq_model.decode_code(x,[x.shape[0],8,16,16]) # [bs, 3, H, W]
            elif config.vq == 'maskgit':
                x_decode = vq_model.decode_tokens(x)
                x_decode = torch.clamp(x_decode, 0.0, 1.0)
            elif config.vq == 'sdf8':
                x_decode = vq_model.decode_tokens(x)
                x_decode = torch.clamp(x_decode, -1.0, 1.0)
            elif config.vq == 'IBQ':
                x_decode = vq_model.decode_code(x)
                x_decode = torch.clamp(x_decode, -1.0, 1.0)
            else:
                raise ValueError('Please check the tokenizer type!')
            # x_decode = F.interpolate(x_decode, size=(256, 256), mode='bicubic')
            for k in range(bs):
                # if you want to save the code.
                # torch.save(x[k].unsqueeze(0), os.path.join(save_path_codes, f'gen-cfg-{config.generation_cfg}-c{class_num}_s{num_steps}_r{local_rank}_n{k}.pt'))
                save_image(x_decode[k], os.path.join(save_path_images, f"c-{class_num}-s{seed}-n{k}.png"), normalize=True, value_range=decode_range)

    return 0

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    print(f'current seed is {config.seed}')
    logger = utils.get_logger(__name__)

    generate_samples(config, logger)


if __name__ == '__main__':
    main()
