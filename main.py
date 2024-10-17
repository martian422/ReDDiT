import os

import fsspec
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


def _load_from_checkpoint(config, tokenizer):
    if 'hf' in config.backbone:
        return diffusion.Diffusion(
            config, tokenizer=tokenizer).to('cuda')
    
    return diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        strict=False)


@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig,
    resolve: bool = True,
    save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    
    Args:
        config (DictConfig): Configuration composed by Hydra.
        resolve (bool): Whether to resolve reference fields of DictConfig.
        save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            '{}/config_tree.txt'.format(
                config.checkpointing.save_dir), 'w') as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=8):
    for dl_type, dl in [
        ('train', train_ds), ('valid', valid_ds)]:
        print(f'Printing {dl_type} dataloader batch.')
        batch = next(iter(dl))
        print('Batch input_ids.shape', batch['input_ids'].shape)
        first = batch['input_ids'][0, :k]
        last = batch['input_ids'][0, -k:]
        print(f'First {k} tokens:', tokenizer.decode(first))
        print('ids:', first)
        print(f'Last {k} tokens:', tokenizer.decode(last))
        print('ids:', last)
        print(f'{batch["input_length"]=}')


def generate_samples_old(config, logger, tokenizer):
    logger.info('Generating samples.')
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    assert not config.sampling.semi_ar
    for _ in range(config.sampling.num_sample_batches):
        # sample from llama
        samples, intermediate = model.restore_model_and_sample(
            config.sampling.input_str, num_steps=config.sampling.steps)
        for x in intermediate:
            print(model.tokenizer.batch_decode(x))
            print('')
        text_samples = model.tokenizer.batch_decode(samples)
        print('Text samples:', text_samples)
    return text_samples


def generate_samples(config, logger, tokenizer):
    # not using now
    logger.info('Generating samples.')
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    assert not config.sampling.semi_ar

    eps=1e-5
    model.backbone.eval()
    model.noise.eval()
    batch_size_per_gpu = model.config.loader.eval_batch_size
    num_steps = model.config.sampling.steps
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    x = model._sample_prior_XX(batch_size_per_gpu, model.config.model.length).to(model.device)
    # x.shape= bs 256
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    input_ids = x.clone()
    import numpy as np
    import torch.nn.functional as F
    PAD_MAX=model.lm.cls_token_num
    text = '/home/MaTianren/Workspace/llamaGen/dataset_files/t5_embeds/Recap_COCO_30K_recap/COCO_val2014_000000496618.jpg.npy'
    text_embeds = torch.tensor(np.load(text), dtype=torch.bfloat16,device=model.device)
    text_embeds = F.pad(text_embeds, (0, 0, PAD_MAX-text_embeds.shape[1], 0))

    text_embeds = model.lm.cls_embedding(text_embeds)
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

    if model.config.sampling.noise_removal:
        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=model.device)
        unet_conditioning = model.noise(t)[0]
        input_ids.scatter_(1, indices, x)
        x = model.forward(
            input_ids, text_embeds, attention_mask, x, indices, unet_conditioning).argmax(dim=-1)
          
    torch.save(x,'./outputs/denoised_tensors/{}.pt'.format(text[-20:-8]))

    return x, intermediate

def _ppl_eval(config, logger, tokenizer):
    logger.info('Starting Zero Shot Eval.')

    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed)
    trainer.validate(model, valid_ds)


def _train(config, logger):
    logger.info('Starting Training.')
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)

    if (config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(
                config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(
        config)
    
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f'cuda:{local_rank}')
    

    tokenizer = dataloader.get_tokenizer(config, device=device)
    model = diffusion.Diffusion(config, tokenizer)

    # print('the local rank is:', device)
    # print('the tokenizer rank is:', model.tokenizer.model.device)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)


    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)
    
    logger = utils.get_logger(__name__)
    # tokenizer = dataloader.get_tokenizer(config)

    if config.mode == 'sample_eval':
        generate_samples(config, logger)
    elif config.mode == 'ppl_eval':
        _ppl_eval(config, logger)
    else:
        _train(config, logger)


if __name__ == '__main__':
    main()
