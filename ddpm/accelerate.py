import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import os

class AcceleratorLite:
    '''Lightweight Accelerator (Huggingface) like class to handle device placement and distributed training.'''

    def __init__(self, do_compile=False):
        self.do_compile = do_compile
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

    def __del__(self):
        if self.running_ddp:
            self.print("destroying process group")
            dist.destroy_process_group()

    def prepare(self, model, train_dataset, val_dataset, batch_size):
        model.to(self.device)
        if torch.cuda.is_available() and self.do_compile:
            model = torch.compile(model)
        if self.running_ddp:
            model = DDP(model, device_ids=[self.local_rank])
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        train_dataloader = DataLoaderOnDevice(train_dataloader, self.device)
        val_dataloader = DataLoaderOnDevice(val_dataloader, self.device)
        return model, train_dataloader, val_dataloader

    def print(self, *args, **kwargs):
        if self.is_master_process:
            print(*args, **kwargs)

class DataLoaderOnDevice:
    '''Wrapper to place batches on device.'''

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            yield batch.to(self.device)

    def __len__(self):
        return len(self.dataloader)
