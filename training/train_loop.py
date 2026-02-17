from generative_models.agm import AbstractGenerativeModel
from accelerate.accelerate import AcceleratorLite
from utils import unwrap_torch_compile_state_dict

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Union
import copy
import inspect
import time

import torch
import torch.distributed as dist

@dataclass
class TrainConfig:
    batch_size: int
    simul_batch_size: int
    epochs: int
    epochs_per_checkpoint: int
    lr: float
    weight_decay: float = 0.0
    use_ema: bool = False
    ema_decay: float = 0.9999
    grad_clip: float = 0.0
    autocast_dtype: str = "bfloat16"
    lr_scheduler: str = "none"

class TrainLoop:
    
    def __init__(
        self,
        model: AbstractGenerativeModel,  # model.network is the actual neural network to train
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        train_config: TrainConfig,
        model_config: dict,
        accelerator: AcceleratorLite,
        savepath: Union[str, Path],
        resume_checkpoint_path: Union[Path, None] = None,
    ):
        self.train_config = asdict(train_config)
        self.model_config = model_config

        self._unpack_train_config()
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.savepath = Path(savepath)
        self.resume_checkpoint_path = resume_checkpoint_path

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        assert self.train_dataset[0].shape == self.val_dataset[0].shape

        self.model = model
        self.model.move_to(self.device)
        self.network, self.train_dataloader, self.val_dataloader = accelerator.prepare(self.model.network, train_dataset, val_dataset, self.batch_size)
        self.model.network = self.network

        assert self.simul_batch_size % (self.batch_size * accelerator.world_size) == 0
        self.grad_accum_steps = self.simul_batch_size // (self.batch_size * accelerator.world_size)
        if self.grad_accum_steps == 1:
            accelerator.print("No gradient accumulation")
        else:
            accelerator.print(f"Gradient accumulation steps: {self.grad_accum_steps}")

        if self.device.type == "cuda":
            self.autocast_context = torch.autocast(device_type="cuda", dtype=getattr(torch, self.autocast_dtype))
            self.scaler = torch.GradScaler(device="cuda", enabled=self.autocast_dtype=="float16") # Only use gradscaler for float16
        else:
            self.autocast_context = nullcontext()
            self.scaler = torch.GradScaler(device="cpu", enabled=False)

        if accelerator.running_ddp:
            self.raw_network = self.model.network.module
        else:
            self.raw_network = self.model.network

        if self.use_ema:
            self.ema_network = self._ema_create()

        # Use fused AdamW (Adam with weight decay)
        use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and self.device.type == "cuda"
        accelerator.print(f"Using fused AdamW: {use_fused}")
        self.optimizer = torch.optim.AdamW(params=self.raw_network.parameters(), lr=self.lr, weight_decay=self.weight_decay, fused=use_fused)
        self.scheduler = self._get_lr_scheduler(self.optimizer)
        self.start_epoch = 0
        if self.resume_checkpoint_path is not None:
            self._resume_from_checkpoint(self.resume_checkpoint_path)
    
    def train(self):
        accelerator = self.accelerator
        step = 0

        # Drop any incomplete accumulated batch for simplicity
        n_batches = len(self.train_dataloader)
        n_full_batches = n_batches - (n_batches % self.grad_accum_steps)

        for epoch in range(1, self.epochs + 1):
            absolute_epoch = self.start_epoch + epoch
            t0 = time.time()
            if accelerator.running_ddp:
                self.train_dataloader.sampler.set_epoch(absolute_epoch)
            self.network.train()
            self.optimizer.zero_grad()
            train_accum_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            accum_grad_norm = torch.zeros((), dtype=torch.float32, device=self.device)
            n_simul_train_batches = torch.zeros((), dtype=torch.float32, device=self.device)

            for i, x in enumerate(islice(self.train_dataloader, n_full_batches), start=1):
                # There will be no partial last batch since we drop it
                final_grad_accum_step = i % self.grad_accum_steps == 0
                with self.network.no_sync() if accelerator.running_ddp and not final_grad_accum_step else nullcontext():
                    loss = self.model.loss(x, autocast_context=self.autocast_context)
                    loss /= self.grad_accum_steps
                    train_accum_loss += loss.detach()
                    self.scaler.scale(loss).backward()
                if not final_grad_accum_step:
                    continue # Keep accumulating gradients

                if self.grad_clip != 0.0:
                    self.scaler.unscale_(self.optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                    if not torch.isfinite(norm).item():
                        raise RuntimeError("Non-finite gradient norm detected during clipping.")
                    accum_grad_norm += norm
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.use_ema:
                    self._ema_update()
                self.optimizer.zero_grad()
                step += 1
                n_simul_train_batches += 1

            assert n_simul_train_batches.item() == n_full_batches // self.grad_accum_steps, "All batches should be consumed"

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            eval_network = self.ema_network if self.use_ema else self.raw_network
            eval_network.eval()
            with torch.no_grad():
                val_accum_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                n_val_batches = torch.zeros((), dtype=torch.float32, device=self.device)
                for x in self.val_dataloader:
                    val_accum_loss += self.model.loss(x, autocast_context=self.autocast_context, network_override=eval_network)
                    n_val_batches += 1

            assert n_val_batches.item() == len(self.val_dataloader), "All validation batches should be consumed"

            if accelerator.running_ddp:
                for x in train_accum_loss, val_accum_loss, accum_grad_norm, n_simul_train_batches, n_val_batches:
                    dist.all_reduce(x, op=dist.ReduceOp.SUM)
            if accelerator.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            log_msg = (
                f"epoch: {absolute_epoch} "
                f"| step: {step} "
                f"| train loss: {(train_accum_loss / n_simul_train_batches).item():.6f} "
                f"| val loss: {(val_accum_loss / n_val_batches).item():.6f} "
                + (f"| lr: {current_lr:.7f} " if self.lr_scheduler != "none" else "")
                + (f"| grad norm: {(accum_grad_norm / n_simul_train_batches).item():.3f} " if self.grad_clip != 0 else "")
                + f"| dt: {t1 - t0:.1f}"
            )
            accelerator.print(log_msg, flush=True)

            if accelerator.is_master_process and (absolute_epoch % self.epochs_per_checkpoint == 0 or epoch == self.epochs):
                savepath = self.savepath
                name = savepath.stem + f"-epoch{absolute_epoch}" + savepath.suffix
                current_epoch_savepath = savepath.parent / name
                checkpoint = {}
                if self.use_ema:
                    ema_state_dict = unwrap_torch_compile_state_dict(self.ema_network.state_dict())
                    checkpoint["ema_network_state_dict"] = ema_state_dict
                raw_state_dict = unwrap_torch_compile_state_dict(self.raw_network.state_dict())
                checkpoint |= {
                    "epoch": absolute_epoch,
                    "network_state_dict": ema_state_dict if self.use_ema else raw_state_dict,
                    "raw_network_state_dict": raw_state_dict,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),
                    "model_config": self.model_config,
                    "train_config": self.train_config,
                    "train_indices": self.train_dataset.indices,
                    "val_indices": self.val_dataset.indices,
                }
                torch.save(checkpoint, current_epoch_savepath)

    def _resume_from_checkpoint(self, checkpoint_path):
        '''Resume train loop state from checkpoint. Args: checkpoint_path (str|Path). Returns: None.'''
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.raw_network.load_state_dict(checkpoint["raw_network_state_dict"])
        if self.use_ema:
            self.ema_network.load_state_dict(checkpoint["ema_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            self.lr_scheduler = "none"
            self.scheduler = self._get_lr_scheduler(self.optimizer)
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = int(checkpoint["epoch"])

    def _ema_create(self):
        ema_network = copy.deepcopy(self.raw_network)
        ema_network.requires_grad_(False)
        return ema_network

    @torch.no_grad()
    def _ema_update(self):
        for ema_p, raw_p in zip(self.ema_network.parameters(), self.raw_network.parameters()):
            ema_p.mul_(self.ema_decay).add_(raw_p, alpha=1-self.ema_decay)
        for ema_b, raw_b in zip(self.ema_network.buffers(), self.raw_network.buffers()):
            ema_b.copy_(raw_b)

    def _get_lr_scheduler(self, optimizer):
        match self.lr_scheduler:
            case "none":
                return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
            case "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
            case _:
                raise ValueError(f"Unsupported lr scheduler: {self.lr_scheduler}")

    def _unpack_train_config(self):
        for k, v in self.train_config.items():
            setattr(self, k, v)
