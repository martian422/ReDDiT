import os
import re
import math
import typing
import datasets
import torch
import transformers

import utils
import numpy as np

LOGGER = utils.get_logger(__name__)

import torch.distributed as dist

def get_dataset(
    dataset_path, 
    image_token_dir,
    cache_dir='/workspace/intern/liaomingxiang/ARG-MDM/data/cache', 
    num_proc=len(os.sched_getaffinity(0)), 
    streaming=True,
):

    data = datasets.load_from_disk(dataset_path)


    def preprocess(text):
        return re.sub(r'<<.*?>>', '', text)

    def preprocess_and_tokenize(example):

        image_dir = image_token_dir
        # my_tokenizer=tokenizer.copy()
        # 'not streaming' will cache the embeddings, make sure there is enough space left on your disk.
        # about 500k per image-text pair
        # caption_embs, emb_masks = my_tokenizer.get_text_embeddings(example['text']) 
        # valid_caption_embs = caption_embs[:, :emb_masks.sum()]
        # text_embeds = valid_caption_embs.to(torch.float32)
        image_file = os.path.join(image_dir, example['image_tokens'])
        try:
            image_tokens = np.load(image_file)
        except:
            image_tokens = np.load('/workspace/intern/liaomingxiang/ARG-MDM/laion-coco/01889/018890002-img.npy') ## FIXME
        return dict(text=example['text'], image_tokens=image_tokens.astype(np.int32))

    if streaming:
        tokenized_dataset = data.map(
            preprocess_and_tokenize,
            batched=False,
            num_proc = num_proc,
            desc='Tokenizing')
    else:
        raise Exception

    return tokenized_dataset


def get_tokenizer_old(config):

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.data.tokenizer_name_or_path, use_fast=False)

    # For wrapped batches:
    #  [BOS] sent1 [EOS] sent2-fragment [EOS]
    #  [BOS] sent2-fragment [EOS] sent3 [EOS]
    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(
                'Tokenizer must have a bos_token or '
                f'cls_token: {tokenizer}')
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(
                'Tokenizer must have a eos_token '
                f'or sep_token: {tokenizer}')
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_tokens([f'[MASK{i:0>6d}]' for i in range(config.mask_vocab_size)], special_tokens=True)

    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    tokenizer.chat_template = 'default template'
    print(f'{tokenizer.chat_template=}')
    return tokenizer


def get_dataloaders(
    config,  
    skip_train=False,
    skip_valid=True, 
    valid_seed=None,
):
    num_gpus = torch.cuda.device_count()
    assert config.loader.global_batch_size == (
        config.loader.batch_size * 
        config.trainer.num_nodes * 
        num_gpus * 
        config.trainer.accumulate_grad_batches
    )
    if config.loader.global_batch_size % (
        num_gpus * config.trainer.accumulate_grad_batches) != 0:
        raise ValueError(
            f'Train Batch Size {config.training.batch_size}'
            f'not divisible by {num_gpus} gpus with accumulation '
            f'{config.trainer.accumulate_grad_batches}.')
    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError(
            f'Eval Batch Size for {config.eval.batch_size} '
            f'not divisible by {num_gpus}.')
    if skip_train:
        train_set = None
    else:
        train_set = get_dataset(
            config.data.dataset_path, 
            image_token_dir=config.data.image_token_dir,
            cache_dir=config.data.cache_dir,
        )
    
    if skip_valid:
        valid_set = None
    else:
        valid_set = get_dataset(
            config.data.val_dataset_path, 
            image_token_dir=config.data.image_token_dir, 
            cache_dir=config.data.cache_dir,
        )

    def collate_fn(batch):

        text = [x['text'] for x in batch] 
        image_tokens = [torch.tensor(x['image_tokens'][0]) for x in batch]
        tokens = {
        'text': text, # [1, L, 2048]
        'image_tokens': image_tokens # [256]
        }
        return tokens

    if skip_train:
        train_loader = None
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.loader.batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=not config.data.streaming,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=True)
    if skip_valid:
        valid_loader = None
    else:
        if valid_seed is None:
            shuffle_valid = False
            generator = None
        else:
            shuffle_valid = True
            generator = torch.Generator().manual_seed(valid_seed)
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.loader.eval_batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=shuffle_valid,
            collate_fn=collate_fn,
            generator=generator)
        # Will be used in generative perplexity calculation

    return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

    def __init__(self, *args, generator=None, **kwargs):
        # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
        # which should be reproducible if pl.seed_everything was called beforehand.
        # This means that changing the seed of the experiment will also change the
        # sampling order.
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        kwargs.pop('shuffle', None)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'random_state': self.generator.get_state(),
                        'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get('random_state'))
        self.counter = state_dict['counter']
        # self.start_counter = self.counter
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.

    def __iter__(self) -> typing.Iterator[int]:
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'epoch': self.epoch, 'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.counter = state_dict['counter']
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(
                    padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0
