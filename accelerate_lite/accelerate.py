import os
import random
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

@dataclass
class DataLoaderConfig:
    '''
    Configuration for torch dataloader options.
    Args:
        num_workers (int): Number of worker processes.
        pin_memory (bool): Whether to pin host memory.
        persistent_workers (bool): Whether to keep workers alive between epochs.
        prefetch_factor (int | None): Number of prefetched batches per worker.
    Returns:
        None
    '''

    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None

class AcceleratorLite:
    '''Lightweight Accelerator (Huggingface) like class to handle device placement and distributed training.'''

    def __init__(self, torch_compile=False, base_seed=42, dataloader_config=None):
        self.torch_compile = torch_compile
        self.dataloader_config = self._parse_dataloader_config(dataloader_config)
        self.running_ddp = "RANK" in os.environ
        if self.running_ddp:
            assert torch.cuda.is_available()
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.device = torch.device(f"cuda:{self.local_rank}")
            dist.init_process_group(backend="nccl", init_method="env://", world_size=self.world_size, rank=self.rank)
            torch.cuda.set_device(self.device)
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_master_process = self.rank == 0
        self.set_seed(self.rank, base_seed=base_seed)

    def __del__(self):
        if self.running_ddp:
            self.print("destroying process group")
            dist.destroy_process_group()

    def broadcast(self, tensor, src=0):
        if self.running_ddp:
            dist.broadcast(tensor, src=src)

    def prepare(self, model, train_dataset, val_dataset, batch_size, make_train_eval_dataloader=False):
        model.to(self.device)
        if torch.cuda.is_available() and self.torch_compile:
            self.print(f"torch compile model")
            model = torch.compile(model)
        dataloader_kwargs = asdict(self.dataloader_config)
        if self.running_ddp:
            model = DDP(model, device_ids=[self.local_rank])
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=True)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, **dataloader_kwargs)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, drop_last=True, **dataloader_kwargs)
            if make_train_eval_dataloader:
                sub_train_dataset = Subset(train_dataset, torch.arange(0, len(val_dataset)))
                train_eval_sampler = DistributedSampler(sub_train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=True)
                train_eval_dataloader = DataLoader(sub_train_dataset, batch_size=batch_size, sampler=train_eval_sampler, drop_last=True, **dataloader_kwargs)
                train_eval_dataloader = DataLoaderOnDevice(train_eval_dataloader, self.device)
            else:
                train_eval_dataloader = None
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **dataloader_kwargs)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **dataloader_kwargs)
            if make_train_eval_dataloader:
                sub_train_dataset = Subset(train_dataset, torch.arange(0, len(val_dataset)))
                train_eval_dataloader = DataLoader(sub_train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **dataloader_kwargs)
                train_eval_dataloader = DataLoaderOnDevice(train_eval_dataloader, self.device)
            else:
                train_eval_dataloader = None
        train_dataloader = DataLoaderOnDevice(train_dataloader, self.device)
        val_dataloader = DataLoaderOnDevice(val_dataloader, self.device)
        return model, train_dataloader, val_dataloader, train_eval_dataloader

    def print(self, *args, **kwargs):
        if self.is_master_process:
            print(*args, **kwargs)

    def set_seed(self, rank, base_seed):
        seed = base_seed + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _parse_dataloader_config(self, dataloader_config):
        '''
        Parse dataloader configuration to a `DataLoaderConfig`.
        Args:
            dataloader_config (dict | DataLoaderConfig | None): Dataloader configuration value.
        Returns:
            DataLoaderConfig: Parsed dataloader configuration.
        '''
        if dataloader_config is None:
            return DataLoaderConfig()
        if isinstance(dataloader_config, DataLoaderConfig):
            return dataloader_config
        if isinstance(dataloader_config, dict):
            return DataLoaderConfig(**dataloader_config)
        raise TypeError("dataloader_config must be a dict or DataLoaderConfig")

class DataLoaderOnDevice:
    '''
    Wrapper that moves dataloader batches to a device.
    '''

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.sampler = dataloader.sampler
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            yield batch.to(self.device)

    def __len__(self):
        return len(self.dataloader)
