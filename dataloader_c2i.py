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

from torchvision import transforms
import numpy as np
from PIL import Image

import torch.distributed as dist

def get_dataset_from_token(
    dataset_path, 
    image_token_dir,
    num_proc=len(os.sched_getaffinity(0)), 
    streaming=True,
):
    "from tokens"

    data = datasets.load_from_disk(dataset_path)


    def preprocess_and_tokenize(example):

        image_dir = image_token_dir
        
        image_token_file = os.path.join(image_dir, example['image_tokens'])
        # image_file = os.path.join(image_dir, example['image_tokens'])
        try:
            image_tokens = np.load(image_token_file)
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

def get_dataset_from_image(
    dataset_path, 
    imagenet_dir,
    num_proc=len(os.sched_getaffinity(0)), 
    streaming=False,
):
    "from images"

    data = datasets.load_from_disk(dataset_path)


    def augment_extract(image):
        image_crop = image
        return image_crop

    def preprocess_and_tokenize(example):
        
        image_path = os.path.join(imagenet_dir, example['image_dir'])

        try:
            img = Image.open(image_path).convert("RGB")
        except:
            breakpoint()
            img = Image.open(image_path).convert("RGB")

        return dict(text=example['class'], image_tokens=img)

    if streaming:
        tokenized_dataset = data.map(
            preprocess_and_tokenize,
            batched=False,
            num_proc = num_proc,
            desc='Tokenizing')
    else:
        raise Exception

    return tokenized_dataset

def get_dataset_separate(
    dataset_path, 
    token_dir,
    image_dir,
    num_proc=len(os.sched_getaffinity(0)), 
    streaming=True,
):
    "from tokens"

    data = datasets.load_from_disk(dataset_path)


    def preprocess(example):
        
        token_path = os.path.join(token_dir, example['image_tokens'])
        image_path = os.path.join(image_dir, example['image_tokens'][:-4] + '.jpg')
        # assume the token and image have same dir structure.
        try:
            tokens = np.load(token_path)
            image = Image.open(image_path).convert("RGB")
        except:
            print('Error when loading datasets.')
            pass

        return dict(text = example['text'], image_tokens = tokens.astype(np.int32), images = image)

    if streaming:
        tokenized_dataset = data.map(
            preprocess,
            batched=False,
            num_proc = num_proc,
            desc='Tokenizing')
    else:
        raise Exception

    return tokenized_dataset

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
        if config.data.type == 'token':
            train_set = get_dataset_from_token(
                config.data.dataset_path, 
                image_token_dir=config.data.image_token_dir
            )
        elif config.data.type == 'image':
            train_set = get_dataset_from_image(
                config.data.dataset_path, 
                image_token_dir=config.data.image_token_dir
            )
        elif config.data.type == 'both':
            train_set = get_dataset_separate(
                config.data.dataset_path, 
                token_dir=config.data.image_token_dir,
                image_dir=config.data.image_dir
            )
        else:
            raise ValueError
    
    if skip_valid:
        valid_set = None
    else:
        raise ValueError

    def collate_fn(batch):

        text = [x['text'] for x in batch] 
        image_tokens = [torch.tensor(x['image_tokens'][0]) for x in batch]
        images = [x['images'] for x in batch]

        tokens = {
        'text': text, # [1, L, 2048]
        'image_tokens': image_tokens, # [256],
        'images': images
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
