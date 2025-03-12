import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader_c2i as dataloader
import diffusion
import utils
import timm

from tokenizer.llamagen_vq import VQ_models


omegaconf.OmegaConf.register_new_resolver(
    'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
    'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)

@torch.no_grad()
def load_encoder_dinov2(config):
    encoder_dir= config.repa_loss.dino_model
    latent_size = config.repa_loss.latent_size
    encoder = torch.hub.load('facebookresearch/dinov2', encoder_dir)
    del encoder.head
    encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [latent_size, latent_size],
            )
    encoder.head = torch.nn.Identity()

    return encoder



def _load_from_checkpoint(config):
    if 'hf' in config.backbone:
        return diffusion.Diffusion(
            config).to('cuda')
    
    return diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
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
    
    if config.repa_loss.use_repa==True:
        dino_encoder = load_encoder_dinov2(config)
        dino_encoder = dino_encoder.to(device)
        for p in dino_encoder.parameters():
            p.requires_grad = False
        dino_encoder.eval()

        if config.data.type != 'both':

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
        else:
            vq_model=None
            print("image detokenizer skipped")

    else:
        dino_encoder=None
        vq_model=None
        print(f"image tokenizer is skipped")

    model = diffusion.Diffusion(config, dino_encoder, vq_model)

    # print('the local rank is:', device)
    # print('the tokenizer rank is:', model.tokenizer.model.device)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)


    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)

def _debug(config, logger):
    logger.info('Starting Training.')
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
    
    if config.repa_loss.use_repa==True:
        dino_encoder = load_encoder_dinov2(config)
        dino_encoder = dino_encoder.to(device)
        for p in dino_encoder.parameters():
            p.requires_grad = False
        dino_encoder.eval()

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
    else:
        dino_encoder=None
        vq_model=None
        print(f"image tokenizer is skipped")

    model = diffusion.Diffusion(config, dino_encoder, vq_model)

    # print('the local rank is:', device)
    # print('the tokenizer rank is:', model.tokenizer.model.device)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=None)


    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)
    
    logger = utils.get_logger(__name__)
    # tokenizer = dataloader.get_tokenizer(config)

    if config.mode == 'debug':
        _debug(config, logger)
    else:
        _train(config, logger)


if __name__ == '__main__':
    main()
