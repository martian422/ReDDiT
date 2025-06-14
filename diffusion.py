import itertools
import math
import copy
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

from torchvision import transforms

import dataloader_c2i as dataloader
import models
import noise_schedule
import utils


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import time
from torchvision.transforms import Normalize

# lm_config=LlamaGen_Config()

LOG2 = math.log(2)


# def _sample_categorical(categorical_probs):
#     gumbel_norm = (
#         1e-10
#         - (torch.rand_like(categorical_probs) + 1e-10).log())
#     return (categorical_probs / gumbel_norm).argmax(dim=-1)
CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)



def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def _sample_categorical(categorical_probs):
    # A simple sample function based on probability distribution
    *sample_shape, C = categorical_probs.shape
    return torch.multinomial(categorical_probs.reshape(-1, C), num_samples=1).reshape(*sample_shape)

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

### introducing REPA

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


@dataclass
class Loss:
    total_loss: torch.FloatTensor
    ce_loss: torch.FloatTensor
    repa_loss: torch.FloatTensor
    nlls: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
    pass


class BPD(NLL):
    def compute(self) -> Tensor:
        """Computes the bits per dimension.

        Returns:
            bpd
        """
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> Tensor:
        """Computes the Perplexity.

        Returns:
         Perplexity
        """
        return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
    def __init__(
        self,
        config,
        dino_encoder=None,
        vq_model=None):
        super().__init__()
        self.save_hyperparameters("config")
        self.config = config

        
        self.vocab_size = self.config.lm_vocab_size + self.config.mask_vocab_size
        self.mask_index_range = (self.config.lm_vocab_size, self.vocab_size)
        self.sampler = self.config.sampling.predictor

        self.parameterization = self.config.parameterization

        self.dino = dino_encoder
        self.vq = vq_model
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True)])
        # note: totensors will first use x/255 to get [0, 1] image tensors, so donot do it again.

        # self.embed_tokens = EmbeddingWithMask(
        #     self.input_embedding, self.mask_index_range).bfloat16()
        self.backbone = models.dit.DIT(
            self.config, lm_vocab_size=self.config.lm_vocab_size, vocab_size=self.vocab_size).bfloat16()

        self.seq_len = self.config.model.length

        self.softplus = torch.nn.Softplus()

        self.noise = noise_schedule.get_noise(self.config, dtype=self.dtype)
        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
                itertools.chain(
                    # self.embed_tokens.parameters(),
                    self.backbone.parameters(),
                    self.noise.parameters(),
                ),
                decay=self.config.training.ema,
            )
        else:
            self.ema = None
        
        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.time_conditioning # true recommended
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint['loops'][
            'fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops'][
            'fit_loop']['epoch_loop.batch_progress'][
                'current']['completed']

    def on_save_checkpoint(self, checkpoint):
        keys_to_remove = [key for key in checkpoint['state_dict'].keys() if (key.startswith('lm') or key.startswith('vq'))]
        for key in keys_to_remove:
            del checkpoint['state_dict'][key]
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['total'][
                'completed'] = checkpoint['loops']['fit_loop'][
                    'epoch_loop.automatic_optimization.optim_progress'][
                        'optimizer']['step']['total'][
                            'completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['current'][
                'completed'] = checkpoint['loops']['fit_loop'][
                    'epoch_loop.automatic_optimization.optim_progress'][
                        'optimizer']['step']['current'][
                            'completed'] * self.trainer.accumulate_grad_batches
        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.state_dict'][
                '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
                    'epoch_loop.automatic_optimization.optim_progress'][
                        'optimizer']['step']['total']['completed']
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler,
                             'state_dict'):
            sampler_state_dict = self.trainer.\
                train_dataloader.sampler.state_dict()
            checkpoint['sampler'][
                'random_state'] = sampler_state_dict.get(
                    'random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        # Adapted from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed)
        if distributed:
            sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader.RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, 'shuffle'):
                dl_sampler = sampler_cls(
                    dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)
            if (distributed
                    and self.fast_forward_epochs is not None
                    and self.fast_forward_batches is not None):
                dl_sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': (self.fast_forward_batches
                                            * self.config.loader.batch_size)})
            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=self.config.loader.num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False, ##Note that the shuffle is still turned on in dl_sampler by default.
                    persistent_workers=True))
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(itertools.chain(
                # self.embed_tokens.parameters(),
                self.backbone.parameters(),
                self.noise.parameters()))


    def _subs_parameterization_with_repa(self, logits, xt, zs_tilde=None):
        # log prob at the mask index = - infinity
        # 'Zero Masking Probabilities'
        logits[:, :, self.mask_index_range[0]:] += self.neg_infinity
        
        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens. (result in one-hot for each unmasked token)
        
        # 'Carry-Over Unmasking'
        unmasked_indices = (xt < self.mask_index_range[0])
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0.0 

        return logits, zs_tilde

    def _process_sigma(self, sigma):
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _preprocess_inputs(self, inputs, x):
        print('this is not used!')
        return 0

    def _preprocess_batch(self, batch):

        image_tokens = torch.stack(batch['image_tokens']) # shape [bs, L]
        labels = torch.tensor([int(t[0]) for t in batch['text']]).unsqueeze(1).to(image_tokens.device)

        if self.config.data.type =='both':
            images = [ self.transform(x) for x in batch['images'] ]
            images = torch.stack(images).to(image_tokens.device)

        if self.config.batch_drop_out > 0:
            # for cfg perhaps.
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.config.batch_drop_out
            labels = torch.where(drop_ids[:, None], 1000, labels)

        # text_embeds = torch.stack([F.pad(t[:PAD_MAX,:], (0, 0, max(PAD_MAX - t.size(0),0), 0)) for t in text_embeds])
        # labels = labels.to(image_tokens.device)
        
        # attention_mask = (text_embeds[:,:,0]!=0).to(torch.int)

        if self.config.repa_loss.use_repa==True:
            ls = self.config.repa_loss.latent_size
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    if self.config.data.type != 'both':
                        raw_images = self.vq.decode_code(image_tokens,[image_tokens.shape[0],8,ls,ls]) # remember to change for other resolution.
                        raw_images_ = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(raw_images)
                    else:
                        raw_images_ = images.to(torch.bfloat16)
                    raw_images_ = torch.nn.functional.interpolate(raw_images_, ls * self.config.repa_loss.psz, mode='bicubic')
                    zs = self.dino.forward_features(raw_images_)
                    zs = zs['x_norm_patchtokens'] # [bs, 16*16, 768]
        else:
            zs = None
         

        # indices = (PAD_MAX + torch.arange(block_length, device=text_embeds.device)).repeat(2,1)
        return image_tokens, labels, zs
    
    def forward(self, xt, labels, sigma):
        """dit"""
        sigma = self._process_sigma(sigma)
         # equals to [bs] of zero if time_conditioning is false.
        labels = labels.squeeze(1)
        # noise_xt may introduce extra noise to enforce reconstruction
        logits, zs_tilde = self.backbone(labels, self.noise_xt(xt), sigma) # y x t

        return self._subs_parameterization_with_repa(logits=logits, xt=xt, zs_tilde = zs_tilde)
    
    def sample_forward(self, xt, labels, sigma, temp = 1.0):
        """ddit sample, skipping zero-mask, keep carry-out, donot log."""

        sigma = self._process_sigma(sigma)
         # equals to [bs] of zero if time_conditioning is false.
        labels = labels.squeeze(1)
        logits, zs = self.backbone(labels, xt, sigma) # y x t

        logits = logits / temp
        
        logits[:, :, self.mask_index_range[0]:] += self.neg_infinity

        unmasked_indices = (xt < self.mask_index_range[0])
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] -= self.neg_infinity

        # logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)        

        return logits, zs
    
    def sample_forward_raw(self, xt, labels, sigma, temp = 1.0):
        """ddit sample, skip both zero-mask and carry-out, do not log"""

        sigma = self._process_sigma(sigma)
         # equals to [bs] of zero if time_conditioning is false.
        labels = labels.squeeze(1)
        logits, zs_tilde = self.backbone(labels, xt, sigma) # y x t
        logits[:, :, self.mask_index_range[0]:] += self.neg_infinity
        logits = logits / temp
        
        # logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        return logits, zs_tilde


    def _compute_loss(self, batch, prefix):
        # entrance
        input_ids, text_embeds, zs = self._preprocess_batch(batch)
        losses = self._loss(input_ids, text_embeds, zs)

        return losses
    
    def on_train_epoch_start(self):
        if self.config.resume_modified_scheduler:
            print("Resetting LR to 5e-5 on train epoch start")
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group['lr'] = 5e-5 # write your resume target here.
            # lr = self.trainer.optimizers[0].param_groups[0]['lr']
            # print(f"[Epoch {self.current_epoch}] LR in use: {lr:.2e}")
        self.backbone.train()
        self.noise.train()

    def training_step(self, batch, batch_idx):
        losses = self._compute_loss(batch, prefix='train')
        total_loss = losses.total_loss
        self.log(name='trainer/loss',
                         value=total_loss.item(),
                         on_step=True,
                         on_epoch=True,
                         sync_dist=True)
        self.log(name='trainer/repa_loss',
                         value=losses.repa_loss.item(),
                         on_step=True,
                         on_epoch=True,
                         sync_dist=True)
        self.log(name='trainer/ce_loss',
                         value=losses.ce_loss.item(),
                         on_step=True,
                         on_epoch=True,
                         sync_dist=True)
        return total_loss

    
    def on_train_epoch_end(self):
    # Access the logger
        metrics = self.trainer.callback_metrics

        # Get the loss values (ensure keys match your logged names)
        total_loss = metrics.get("trainer/loss")
        repa_loss = metrics.get("trainer/repa_loss")
        ce_loss = metrics.get("trainer/ce_loss")

        # Format the output
        log_str = (f"Epoch {self.current_epoch} - "
                f"Total Loss: {total_loss:.4f}, "
                f"Repa Loss: {repa_loss:.4f}, "
                f"CE Loss: {ce_loss:.4f}\n")

        # Append to a text file
        with open("training_losses.txt", "a") as f:
            f.write(log_str)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            itertools.chain(
                # self.embed_tokens.parameters(),
                self.backbone.parameters(),
                self.noise.parameters()),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1,
                         self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val/loss',
            'name': 'trainer/lr',
        }

        if self.config.resume_modified_scheduler:
            # if you want to totally reset the lr to const during debugging, use this.
            return [optimizer]
        else:
            return [optimizer], [scheduler_dict]

    def q_xt(self, x, move_chance):
        """Computes the noisy sample xt.

        Args:
            x: int torch.Tensor with shape (batch_size,
                    diffusion_model_input_length), input. 
            move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(
            * x.shape, device=x.device) < move_chance
        
        mask_index = torch.randint(*self.mask_index_range, size=x.shape, dtype=x.dtype, device=x.device)

        xt = torch.where(move_indices, mask_index, x)

        return xt
    
    def noise_xt(self, x):
        """Computes the noisy sample xt.

        Args:
            x: int torch.Tensor with shape (batch_size,
                    diffusion_model_input_length), input. 
            move_chance: float torch.Tensor with shape (batch_size, 1).
        """

        if self.config.random_noise == True:
            
            hash_indices = torch.rand(* x.shape, device=x.device) < 0.1
            hash_index = torch.randint(0, self.config.lm_vocab_size, size=x.shape, dtype=x.dtype, device=x.device)

            xt = torch.where(hash_indices, hash_index, x)
            return xt
        else:
            return x
    
    def q_xt_hash(self, x):
        """
        makes the masks random.
        """
        unmasked = x < self.mask_index_range[0]
        mask_index = torch.randint(*self.mask_index_range, size=x.shape, dtype=x.dtype, device=x.device)
        x_hash = torch.where(unmasked, x, mask_index)
        return x_hash

    def _sample_prior(self, *batch_dims):
        " TBD "
        return torch.randint(*self.mask_index_range, size=batch_dims, dtype=torch.int64)

    
    @torch.no_grad()
    ## trying to replicate the linear growth of cfg as MUSE did.
    def _ddpm_update_v0(self, xt, labels, t, dt, p_x0=None):
        "worse than v1, abandoned!"

        # assert self.config.noise.type == 'loglinear'
        # non-linear cannot be accelerated using this function
        # 
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
      
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]

        assert move_chance_t.ndim == 3, move_chance_t.shape

        if p_x0 is None:
            # a linear cfg growth as t decrease from 1 to 0.
            if self.config.generation_cfg > 1 :
                if self.config.sampling.cfg_schedule == 'linear':
                    current_cfg = (self.config.generation_cfg - 1) * (1 + dt - t[0].item()) + 1
                elif self.config.sampling.cfg_schedule == 'const': 
                    current_cfg = self.config.generation_cfg
                elif self.config.sampling.cfg_schedule == 'biaslinear': 
                    offset = self.config.sampling.cfg_offset # starting from 1.0+ seems not ideal
                    current_cfg = (self.config.generation_cfg - offset) * (1 + dt - t[0].item()) + offset
                # print(f'Current cfg is {current_cfg}.')
                labels_all = torch.cat([labels, 1000 * torch.ones_like(labels)])

                p_x0_all, _ = self.forward(xt.repeat(2,1), labels_all, sigma_t.repeat(2,1))

                p_x0_cond, p_x0_uncond = torch.split(p_x0_all, p_x0_all.shape[0] // 2, dim = 0)
                p_x0 = p_x0_uncond + current_cfg * (p_x0_cond - p_x0_uncond)
                p_x0 = p_x0.exp()
            else:
                p_x0 = self.forward(xt, labels, sigma_t)
                p_x0 = p_x0.exp()

        assert move_chance_t.ndim == p_x0.ndim

        one_hot_x = move_chance_s[:, :, 0, None] * F.one_hot(xt, num_classes=p_x0.shape[2]) * (1.0 / self.config.mask_vocab_size) #
    
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index_range[0]:] = one_hot_x[:, :, self.mask_index_range[0]:]

        _x = _sample_categorical(q_xs)
        
        copy_flag = (xt < self.mask_index_range[0]).to(xt.dtype)

        return p_x0, copy_flag * xt + (1 - copy_flag) * _x
    
    def _ddpm_update_v1(self, xt, labels, t, dt, p_x0=None):
        "removed zero-mask using sample_forward, relies on probabilities. This is our default sampler"
        # assert self.config.noise.type == 'loglinear'

        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)

        if t.ndim > 1:
            t = t.squeeze(-1)

        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)

        # move_chance_s_1 = np.arccos(1-(t[0]-dt[0][0]).item()) / (math.pi * 0.5)
        # move_chance_t_1 = np.arccos(1-t[0].item()) / (math.pi * 0.5)

        k_t = move_chance_t[0]
        k_s = move_chance_s[0]

        # https://github.com/bytedance/1d-tokenizer/blob/main/modeling/maskgit.py#L157
        # ann_temp = (0.5 + t)[0].item() # do not add annealing in llamagen settings!
        ann_temp = 1.0
        # gumble_coeff = 1.2 * t[0].item() + 0.1
        # gumble_coeff = 0.0

        # assert move_chance_t.ndim == 3, move_chance_t.shape

        if p_x0 is None:
            # a linear cfg growth as t decrease from 1 to 0.
            if self.config.generation_cfg > 1 :
                if self.config.sampling.cfg_schedule == 'linear':
                    current_cfg = (self.config.generation_cfg - 1) * (1 + dt[0].item() - t[0].item()) + 1
                elif self.config.sampling.cfg_schedule == 'const': 
                    current_cfg = self.config.generation_cfg
                elif self.config.sampling.cfg_schedule == 'biaslinear': 
                    offset = self.config.sampling.cfg_offset # starting from 1.0+ seems not ideal
                    current_cfg = (self.config.generation_cfg - offset) * (1 + dt[0].item() - t[0].item()) + offset
                # print(f'Current cfg is {current_cfg}.')
                labels_all = torch.cat([labels, 1000 * torch.ones_like(labels)])
                logits_all, _ = self.sample_forward(xt.repeat(2,1), labels_all, sigma_t.repeat(2,1), temp=ann_temp)

                logits_cond, logits_uncond = torch.split(logits_all, logits_all.shape[0] // 2, dim = 0)
                logits_cfg = logits_uncond + current_cfg * (logits_cond - logits_uncond)
                # logits_cfg = logits_cond + current_cfg * (logits_cond - logits_uncond)
                # doing exp() after cfg can lower the FID by ~1.
                # logits_cfg = add_gumbel_noise(logits_cfg, gumble_coeff)
                p_x0 = F.softmax(logits_cfg, dim=-1)
            else:
                logits_, _ = self.sample_forward(xt, labels, sigma_t)
                # logits_with_noise = add_gumbel_noise(logits_, gumble_coeff)
                p_x0 = F.softmax(logits_, dim=-1)
        p_mask = k_s # no div, carry out the first, then use hash to make colorful noise
    
        q_xs = p_x0 * (k_t - k_s) / k_t
        # q_xs[:, :, self.mask_index_range[0]:] = p_mask / k_t

        q_xs[:, :, self.mask_index_range[0]:] = 0.0
        q_xs[:, :, self.mask_index_range[0]] = p_mask / k_t
        # if a certain position of xt is mask, then _x may have mask.
        # if a certain position of xt is not mask, then _x is not mask.
        _x = _sample_categorical(q_xs)
        
        copy_flag = (xt < self.mask_index_range[0]).to(xt.dtype)

        _xs = copy_flag * xt + (1 - copy_flag) * _x

        return p_x0, _xs
    
    def _ddpm_update_final(self, xt, labels, t, dt, p_x0=None):
        "final update to ensure that all tokens are decoded."
        # assert self.config.noise.type == 'loglinear'

        sigma_t, _ = self.noise(t)

        if t.ndim > 1:
            t = t.squeeze(-1)

        if p_x0 is None:
            # a linear cfg growth as t decrease from 1 to 0.
            if self.config.generation_cfg > 1 :
                if self.config.sampling.cfg_schedule == 'linear':
                    current_cfg = (self.config.generation_cfg - 1) * (1 + dt[0].item() - t[0].item()) + 1
                elif self.config.sampling.cfg_schedule == 'const': 
                    current_cfg = self.config.generation_cfg
                elif self.config.sampling.cfg_schedule == 'biaslinear': 
                    offset = self.config.sampling.cfg_offset # starting from 1.0+ seems not ideal
                    current_cfg = (self.config.generation_cfg - offset) * (1 + dt[0].item() - t[0].item()) + offset
                # print(f'Current cfg is {current_cfg}.')
                labels_all = torch.cat([labels, 1000 * torch.ones_like(labels)])
                logits_all, _ = self.sample_forward_raw(xt.repeat(2,1), labels_all, sigma_t.repeat(2,1))

                logits_cond, logits_uncond = torch.split(logits_all, logits_all.shape[0] // 2, dim = 0)
                logits_cfg = logits_uncond + current_cfg * (logits_cond - logits_uncond)
                p_x0 = F.softmax(logits_cfg, dim=-1)
            else:
                logits_cfg, _ = self.sample_forward_raw(xt, labels, sigma_t)
                p_x0 = F.softmax(logits_cfg, dim=-1)

        p_x0[:, :, self.mask_index_range[0]:] = 0.0

        _x = _sample_categorical(p_x0)
        
        copy_flag = (xt < self.mask_index_range[0]).to(xt.dtype)

        _x0 = copy_flag * xt + (1 - copy_flag) * _x

        return p_x0, _x0
    
    def _maskgit_update_old(self, xt, labels, t, dt, logits_x0=None):
        "maskgit sample method, relies on logits. previous low-precision gumbel noise"
        # assert self.config.noise.type == 'loglinear'
        device = xt.device
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)

        mask_ratio = 1 - torch.exp(-sigma_s)
        mask_ratio = mask_ratio[0].item()

        # mask_ratio = np.arccos(1-(t-dt)[0].item()) / (math.pi * 0.5)

        if t.ndim > 1:
            t = t.squeeze(-1)

        ann_temp = (0.5 + 0.8 * t)[0].item()

        if logits_x0 is None:
            # a linear cfg growth as t decrease from 1 to 0.
            if self.config.generation_cfg > 1 :
                if self.config.sampling.cfg_schedule == 'linear':
                    current_cfg = (self.config.generation_cfg - 1) * (1 + dt - t[0].item()) + 1
                elif self.config.sampling.cfg_schedule == 'const': 
                    current_cfg = self.config.generation_cfg
                elif self.config.sampling.cfg_schedule == 'biaslinear': 
                    offset = self.config.sampling.cfg_offset # starting from 1.0+ seems not ideal
                    current_cfg = (self.config.generation_cfg - offset) * (1 + dt - t[0].item()) + offset
                # print(f'Current cfg is {current_cfg}.')
                labels_all = torch.cat([labels, 1000 * torch.ones_like(labels)])
                logits_all, _ = self.sample_forward_raw(xt.repeat(2,1), labels_all, sigma_t.repeat(2,1), temp=ann_temp)

                logits_cond, logits_uncond = torch.split(logits_all, logits_all.shape[0] // 2, dim = 0)
                logits_x0 = logits_uncond + current_cfg * (logits_cond - logits_uncond)

            else:
                logits_x0, _ = self.sample_forward_raw(xt, labels, sigma_t)

        # logits_x0 = logits_x0.to(torch.float64)

        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))
        def gumbel_noise(t):
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))
        def _add_gumbel_noise(t, temperature):
            return t + temperature * gumbel_noise(t)
        
        is_mask = (xt > (self.config.lm_vocab_size -1))

        ratio = t[0].item()
        annealed_temp = 2.0 * ratio + 0.1

        logits_with_noise = _add_gumbel_noise(logits_x0, annealed_temp)

        x0_pred = logits_with_noise.argmax(dim=-1)
        # logits_x0[:, :, self.mask_index_range[0]:] = -np.inf
        # sampled_ids = _sample_categorical(F.softmax(logits_x0, dim=-1))
        x0_pred_logits = torch.squeeze(
                torch.gather(logits_x0, dim=-1, index=torch.unsqueeze(x0_pred, -1)), -1)
        x0_pred = torch.where(is_mask, x0_pred, xt)
        x0_pred_logits = torch.where(is_mask, x0_pred_logits, +np.inf).float()

        
        mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).to(device)
        mask_len = torch.maximum(torch.Tensor([1]).to(device),
                                 torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1, mask_len))[0].squeeze()
        confidence = _add_gumbel_noise(x0_pred_logits, annealed_temp)
        sorted_confidence, _ = torch.sort(confidence, axis=-1)
        cut_off = sorted_confidence[:, mask_len.long() - 1:mask_len.long()]
        to_mask = (confidence <= cut_off)

        _xs = torch.where(to_mask, self.config.lm_vocab_size, x0_pred)

        return logits_x0, _xs
    
    def _maskgit_update(self, xt, labels, t, dt, logits_x0=None):
        "maskgit sample method, relies on logits. using arXiv:2409.02908 new gumbel, merged as default maskgit method."
        # assert self.config.noise.type == 'loglinear'
        device = xt.device
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)

        mask_ratio = 1 - torch.exp(-sigma_s)
        mask_ratio = mask_ratio[0].item()

        # mask_ratio = np.arccos(1-(t-dt)[0].item()) / (math.pi * 0.5)

        if t.ndim > 1:
            t = t.squeeze(-1)

        # ann_temp = (0.5 + 0.8 * t)[0].item()
        ann_temp = 1.0

        if logits_x0 is None:
            # a linear cfg growth as t decrease from 1 to 0.
            if self.config.generation_cfg > 1 :
                if self.config.sampling.cfg_schedule == 'linear':
                    current_cfg = (self.config.generation_cfg - 1) * (1 + dt - t[0].item()) + 1
                elif self.config.sampling.cfg_schedule == 'const': 
                    current_cfg = self.config.generation_cfg
                elif self.config.sampling.cfg_schedule == 'biaslinear': 
                    offset = self.config.sampling.cfg_offset # starting from 1.0+ seems not ideal
                    current_cfg = (self.config.generation_cfg - offset) * (1 + dt - t[0].item()) + offset
                # print(f'Current cfg is {current_cfg}.')
                labels_all = torch.cat([labels, 1000 * torch.ones_like(labels)])
                logits_all, _ = self.sample_forward_raw(xt.repeat(2,1), labels_all, sigma_t.repeat(2,1), temp=ann_temp)

                logits_cond, logits_uncond = torch.split(logits_all, logits_all.shape[0] // 2, dim = 0)
                logits_x0 = logits_uncond + current_cfg * (logits_cond - logits_uncond)
                p_x0 = F.softmax(logits_x0, dim=-1)

            else:
                logits_x0, _ = self.sample_forward_raw(xt, labels, sigma_t)

        
        is_mask = (xt > (self.config.lm_vocab_size -1))

        ratio = t[0].item()
        annealed_temp = 1.0 * ratio + 0.5 # 4.5 for maskgit tokenizer
        # sample_temp = 1.0

        logits_with_noise = add_gumbel_noise(logits_x0, annealed_temp)

        # x0_pred = logits_with_noise.argmax(dim=-1)
        x0_pred = _sample_categorical(p_x0)
        x0_pred_logits = torch.squeeze(
                torch.gather(logits_x0, dim=-1, index=torch.unsqueeze(x0_pred, -1)), -1)
        x0_pred = torch.where(is_mask, x0_pred, xt) # paste the newly decoded tokens to xt, get _x0
        x0_pred_logits = torch.where(is_mask, x0_pred_logits, +np.inf).float() # carry out the old tokens' logit to inf.

        # mask_ratio = np.arccos(1 - ratio + 1e-6) / (math.pi * 0.5)
        
        mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).to(device)
        mask_len = torch.maximum(torch.Tensor([1]).to(device),
                                 torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1, mask_len))[0].squeeze()
        confidence = add_gumbel_noise(x0_pred_logits, annealed_temp)
        sorted_confidence, _ = torch.sort(confidence, axis=-1)
        cut_off = sorted_confidence[:, mask_len.long() - 1:mask_len.long()]
        to_mask = (confidence <= cut_off)

        # count_prev = xt < self.config.lm_vocab_size

        _xs = torch.where(to_mask, self.config.lm_vocab_size, x0_pred)

        # count_new = ids < self.config.lm_vocab_size

        # alter = count_prev.to(torch.int) - count_new.to(torch.int)
        # print(alter.max().item())
        # print(count[0].sum())

        return logits_x0, _xs
    
    def _flow_matching_update(self, xt, labels, t, dt, p_x0=None):
        "flow-matching sample method, relies on probabilities."
        # assert self.config.noise.type == 'loglinear'
        device = xt.device
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        # pi = torch.pi

        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        jump_t = move_chance_t[0] 
        jump_s = move_chance_s[0] 
        
        # kt = ((1 - torch.exp(-sigma_t)**2)**0.5)[0] # sin (pi/2 * 1-t) universal
        # d_kt = (pi/2 * torch.exp(-sigma_t))[0] # pi/2 * cos(pi/2 * 1-t) only for cosine !
        move_chance_t = move_chance_t[:, None]
        move_chance_s = move_chance_s[:, None]

        if t.ndim > 1:
            t = t.squeeze(-1)

        ann_temp = (0.5 + 0.8 * t)[0].item() # important and useful
        # ann_temp = 1.0

        if p_x0 is None:
            # a linear cfg growth as t decrease from 1 to 0.
            if self.config.generation_cfg > 1 :
                if self.config.sampling.cfg_schedule == 'linear':
                    current_cfg = (self.config.generation_cfg - 1) * (1 + dt - t[0].item()) + 1
                elif self.config.sampling.cfg_schedule == 'const': 
                    current_cfg = self.config.generation_cfg
                elif self.config.sampling.cfg_schedule == 'biaslinear': 
                    offset = self.config.sampling.cfg_offset # starting from 1.0+ seems not ideal
                    current_cfg = (self.config.generation_cfg - offset) * (1 + dt - t[0].item()) + offset
                # print(f'Current cfg is {current_cfg}.')
                labels_all = torch.cat([labels, 1000 * torch.ones_like(labels)])
                logits_all, _ = self.sample_forward_raw(xt.repeat(2,1), labels_all, sigma_t.repeat(2,1), temp=ann_temp)

                logits_cond, logits_uncond = torch.split(logits_all, logits_all.shape[0] // 2, dim = 0)
                logits_x0 = logits_uncond + current_cfg * (logits_cond - logits_uncond)
                # p_x0 = torch.softmax(logits_cfg, dim=-1)
            else:
                logits_x0, _ = self.sample_forward_raw(xt, labels, sigma_t)
                # p_x0 = torch.softmax(logits_cfg, dim=-1)
                # p_x0 = p_x0_.exp()
        # flow matching is based on softmax results

        # annealed_temp = 1.5 * t[0].item() + 0.1 # 4.5 for maskgit tokenizer
        # sample_temp = 1.0

        # logits_with_noise = add_gumbel_noise(logits_x0, annealed_temp)
        # p_x0_noised = F.softmax(logits_with_noise,dim=-1)
        p_x0 = F.softmax(logits_x0,dim=-1)

        _x0_pred = _sample_categorical(p_x0)

        # if (t - dt).mean() < 1e-5:
        #     xs = _x0 # this needs to be determined!
        # else:
        xs = xt.clone()
        delta_x0 = F.one_hot(_x0_pred, num_classes=p_x0.shape[2])
        delta_xt = F.one_hot(xt, num_classes=p_x0.shape[2])

        correct = jump_s / jump_t * F.one_hot(xt, num_classes=logits_cond.shape[2]) * (1.0 / self.config.mask_vocab_size)

        # u = (jump_t - jump_s) * p_x0 / jump_t # adapted
        u = (jump_t - jump_s) * delta_x0 / jump_t # vanilla is slightly better!

        u[:, :, self.mask_index_range[0]:] = correct[:, :, self.mask_index_range[0]:]
        
        u = torch.where(delta_xt.to(dtype=torch.bool), torch.zeros_like(u), u)

        # the above calculation is:  u  = coeff * (p_x0 - dirac_xt), coeff = dkt/(1- kt)
        # then, p_xs ~ dirac_xt + h * u

        intensity = u.sum(dim=-1)
        # breakpoint()

        mask_jump = torch.rand(size = xt.shape, device = device)< 1- torch.exp(-intensity)

        # mask_jump = torch.rand(size = xt.shape, device = device) < 1- torch.exp(-dt)*intensity

        xs[mask_jump] = _sample_categorical(u[mask_jump].to(dtype=torch.float32))

        # count_prev = xt < self.config.lm_vocab_size
        # count_new = xs < self.config.lm_vocab_size
        # total_alter = (mask_jump[0].to(torch.int)).sum()
        # newly_unmask = (count_new[0].to(torch.int) - count_prev[0].to(torch.int)).sum()
        # just_updated = total_alter - newly_unmask
        # really_changed = ((xs!=xt)[0].to(torch.int)).sum() - newly_unmask

        # print(f'total alter: {total_alter.item()}, newly unmasked: {newly_unmask}, just updated: {just_updated}, really changed: {really_changed}.')
            
        # count = (xs<self.config.lm_vocab_size)
        # print(f'decoded tokens: {count[0].sum()}')

        return p_x0, xs
    
    def _flow_matching_update_naive(self, xt, labels, t, dt, p_x0=None):
        "naive flow-matching sample method."
        # assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        # pi = torch.pi

        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        jump_t = move_chance_t[0] 
        jump_s = move_chance_s[0] 
        
        # kt = ((1 - torch.exp(-sigma_t)**2)**0.5)[0] # sin (pi/2 * 1-t) universal
        # d_kt = (pi/2 * torch.exp(-sigma_t))[0] # pi/2 * cos(pi/2 * 1-t) only for cosine !
        move_chance_t = move_chance_t[:, None]
        move_chance_s = move_chance_s[:, None]

        if t.ndim > 1:
            t = t.squeeze(-1)

        ann_temp = (0.5 + 0.8 * t)[0].item() # important and useful
        # ann_temp = 1.0

        if p_x0 is None:
            # a linear cfg growth as t decrease from 1 to 0.
            if self.config.generation_cfg > 1 :
                if self.config.sampling.cfg_schedule == 'linear':
                    current_cfg = (self.config.generation_cfg - 1) * (1 + dt - t[0].item()) + 1
                elif self.config.sampling.cfg_schedule == 'const': 
                    current_cfg = self.config.generation_cfg
                elif self.config.sampling.cfg_schedule == 'biaslinear': 
                    offset = self.config.sampling.cfg_offset # starting from 1.0+ seems not ideal
                    current_cfg = (self.config.generation_cfg - offset) * (1 + dt - t[0].item()) + offset
                # print(f'Current cfg is {current_cfg}.')
                labels_all = torch.cat([labels, 1000 * torch.ones_like(labels)])
                logits_all, _ = self.sample_forward_raw(xt.repeat(2,1), labels_all, sigma_t.repeat(2,1), temp=ann_temp)

                logits_cond, logits_uncond = torch.split(logits_all, logits_all.shape[0] // 2, dim = 0)
                logits_x0 = logits_uncond + current_cfg * (logits_cond - logits_uncond)
                # p_x0 = torch.softmax(logits_cfg, dim=-1)
            else:
                logits_x0, _ = self.sample_forward_raw(xt, labels, sigma_t)
                # p_x0 = torch.softmax(logits_cfg, dim=-1)
                # p_x0 = p_x0_.exp()
        # flow matching is based on softmax results

        # annealed_temp = 1.5 * t[0].item() + 0.1 # 4.5 for maskgit tokenizer
        # sample_temp = 1.0

        # logits_with_noise = add_gumbel_noise(logits_x0, annealed_temp)
        # p_x0_noised = F.softmax(logits_with_noise,dim=-1)
        p_x0 = F.softmax(logits_x0,dim=-1)

        delta_xt = F.one_hot(xt, num_classes=p_x0.shape[2])

        u =  (p_x0 - delta_xt) / jump_t 

        p_xs = delta_xt + (jump_t - jump_s) * u

        _xs = _sample_categorical(p_xs)

        return p_x0, _xs


    def _sample_t(self, n, device):
        " get random t "
        _eps_t = torch.rand(n, device=device)
        # Training trick from Kingma to avoid extreme values. See Variational diffusion models. 
        offset = torch.arange(n, device=device) / n
        _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        return t

    def _forward_pass_diffusion(self, input_ids, text_embeds, zs):
        
        x0 = input_ids.clone()
        t = self._sample_t(x0.shape[0], x0.device) # bs

        sigma, dsigma = self.noise(t)
        unet_conditioning = sigma[:, None]
        move_chance = 1 - torch.exp(-sigma[:, None])

        xt = self.q_xt(x0, move_chance) # noised input_ids

        model_output, zs_tilde = self.forward(
            xt, text_embeds, unet_conditioning)
        utils.print_nans(model_output, 'model_output')

        ### repa
        if self.config.repa_loss.use_repa==True:
            proj_loss = 0.
            zs=[zs]
            bsz = zs[0].shape[0]
            for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                    z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                    z_j = torch.nn.functional.normalize(z_j, dim=-1)
                    proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
            proj_loss /= (len(zs) * bsz)
        else:
            proj_loss=0.0
        # deal with zs here or inside subs_para
        
        # SUBS parameterization, continuous time.
        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)
        
        origin_loss = - log_p_theta * (dsigma / (torch.expm1(sigma) + 0.2))[:, None] 
        # 0.2 ( = 1 / 5.0 ) is a min-SNR-5 trick to help convergence for continuous time.
        # https://arxiv.org/pdf/2303.09556
        # origin_loss = - log_p_theta # without re-weighting, this is a classic MLM loss.
        # We observed that MLM loss helps convergence and performance at first, but then gradually makes the output not ideal for MDLM sampling.

        return origin_loss, proj_loss

    def _loss(self, input_ids, text_embeds, zs):

        loss, proj_loss = self._forward_pass_diffusion(input_ids, text_embeds, zs)

        loss_mask = torch.ones(input_ids.shape,dtype=torch.int,device=loss.device)
        nlls = loss * loss_mask
        count = loss_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count
        if self.config.repa_loss.use_repa==True:
            repa_loss = 0.5 * proj_loss.mean()
        else:
            repa_loss = 0.0

        loss_sum = token_nll + repa_loss

        return Loss(
            total_loss=loss_sum, 
            ce_loss=token_nll,
            repa_loss=torch.tensor([repa_loss]),
            nlls=nlls, 
        )
