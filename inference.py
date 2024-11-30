import os


import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader_t2i as dataloader
import diffusion
import utils

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


def generate_samples(config, logger):

    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    assert not config.sampling.semi_ar

    eps=1e-5
    model.backbone.eval()
    model.noise.eval()
    batch_size_per_gpu = model.config.loader.eval_batch_size
    class_nums=[207, 360, 387, 974, 88, 979, 417, 279]

    logger.info('Generating samples.')

    for class_num in class_nums:
        for i in [10,20,30,40,50,60,70,80,90,100]:
        # for i in [2,4,6,8,10,12,14,16,18,20,22,24]:
            num_steps = i
            timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
            x = model._sample_prior_XX(batch_size_per_gpu, model.config.model.length).to(model.device)
            # x.shape= bs 256
            dt = (1 - eps) / num_steps
            p_x0_cache = None

            input_ids = x.clone()
            import numpy as np
            import torch.nn.functional as F

            labels=torch.tensor([[class_num]])
            text_embeds = model.lm.cls_embedding(labels.to(model.device))
            
            attention_mask = (text_embeds[:,:,0]!=0).to(torch.int)
            
            indices = torch.arange(input_ids.shape[1],device = input_ids.device).repeat(input_ids.shape[0],1)

            intermediate = []
            for i in range(num_steps):
                t = timesteps[i] * torch.ones(
                    x.shape[0], 1, device=model.device)
                if model.sampler == 'ddpm':
                    x = model._ddpm_update_XX(input_ids, indices, x, t, dt)
                elif model.sampler == 'ddpm_cache':
                    p_x0_cache, x_next = model._ddpm_caching_update_XX(
                        input_ids, text_embeds, attention_mask, indices, x, t, dt, p_x0=p_x0_cache)
                    if (not torch.allclose(x_next, x)
                            or model.time_conditioning):
                        # Disable caching
                        p_x0_cache = None
                    x = x_next
                else:
                    raise ValueError
                if model.config.sampling.return_intermediate > 0 and (i + 1) % model.config.sampling.return_intermediate == 0:
                    intermediate.append(x)
            # import copy
            # x_old=copy.deepcopy(x)
            if model.config.sampling.noise_removal:
                t = timesteps[-1] * torch.ones(x.shape[0], 1, device=model.device)
                unet_conditioning = model.noise(t)[0]
                input_ids.scatter_(1, indices, x)
                with torch.no_grad():
                    x = model.forward(
                        input_ids, text_embeds, attention_mask, x, indices, unet_conditioning)
                breakpoint()
                x = x[0].argmax(dim=-1)
            torch.save(x,f'/home/node237/Code/mdlm-c2i/outputs/images/repa/gen-norepa-cfg-{config.generation_cfg}-{class_num}_s{num_steps}.pt')
            print(f'Tensor at {num_steps} steps saved.')

    return 0

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    logger = utils.get_logger(__name__)
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f'cuda:{local_rank}')

    generate_samples(config, logger)


if __name__ == '__main__':
    main()
