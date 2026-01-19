from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
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
        model,
        train_dataset,
        val_dataset,
        train_config,
        model_config, # For checkpoint saving
        accelerator,
        savepath,
    ):
        self.train_config = train_config
        self.model_config = model_config
        self._unpack_train_config()
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.savepath = savepath

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        assert self.train_dataset[0].shape == self.val_dataset[0].shape

        self.model, self.train_dataloader, self.val_dataloader = accelerator.prepare(model, train_dataset, val_dataset, self.batch_size)

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
            self.raw_model = self.model.module
        else:
            self.raw_model = self.model

        if self.use_ema:
            self.ema_model = self._ema_create()

        # Use fused AdamW (Adam with weight decay)
        use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and self.device.type == "cuda"
        accelerator.print(f"Using fused AdamW: {use_fused}")
        self.optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.lr, weight_decay=self.weight_decay, fused=use_fused)
        self.scheduler = self._get_lr_scheduler(self.optimizer)
    
    def train(self):
        accelerator = self.accelerator
        step = 0
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            if accelerator.running_ddp:
                self.train_dataloader.sampler.set_epoch(epoch)
            self.model.train()
            self.optimizer.zero_grad()
            train_accum_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            accum_grad_norm = torch.zeros((), dtype=torch.float32, device=self.device)
            n_steps = torch.zeros((), dtype=torch.float32, device=self.device)
            for i, x in enumerate(self.train_dataloader, start=1):
                final_grad_accum_step = i % self.grad_accum_steps == 0 or i == len(self.train_dataloader)
                loss = self.model.loss(x, self.autocast_context)
                loss /= self.grad_accum_steps
                train_accum_loss += loss.detach()
                if accelerator.running_ddp:
                    self.model.require_backward_grad_sync = final_grad_accum_step
                self.scaler.scale(loss).backward()
                if not final_grad_accum_step:
                    continue # Keep accumulating gradients

                if self.grad_clip != 0.0:
                    self.scaler.unscale_(self.optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    accum_grad_norm += norm
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.use_ema:
                    self._ema_update()
                self.optimizer.zero_grad()
                step += 1
                n_steps += 1

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            eval_model = self.ema_model if self.use_ema else self.model
            eval_model.eval()
            with torch.no_grad():
                val_accum_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                n_val_batches = torch.zeros((), dtype=torch.float32, device=self.device)
                for x in self.val_dataloader:
                    val_accum_loss += eval_model.loss(x, self.autocast_context)
                    n_val_batches += 1

            if accelerator.running_ddp:
                for x in train_accum_loss, val_accum_loss, accum_grad_norm, n_steps, n_val_batches:
                    dist.all_reduce(x, op=dist.ReduceOp.SUM)
            if accelerator.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            log_msg = (
                f"epoch: {epoch} "
                f"| step: {step} "
                f"| train loss: {(train_accum_loss / n_steps).item():.6f} "
                f"| val loss: {(val_accum_loss / n_val_batches).item():.6f} "
                f"| lr: {current_lr:.7f}"
                + (f"| grad norm: {(accum_grad_norm / n_steps).item():.3f} " if self.grad_clip != 0 else "")
                + f"| dt: {t1 - t0:.1f}"
            )
            accelerator.print(log_msg, flush=True)

            if accelerator.is_master_process and (epoch % self.epochs_per_checkpoint == 0 or epoch == self.epochs):
                savepath = Path(self.savepath)
                name = savepath.stem + f"-epoch{epoch}" + savepath.suffix
                current_epoch_savepath = savepath.parent / name
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.ema_model.state_dict() if self.use_ema else self.raw_model.state_dict(),
                    "raw_model_state_dict": self.raw_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),
                    "model_config": self.model_config,
                    "train_indices": self.train_dataset.indices,
                    "val_indices": self.val_dataset.indices,
                }
                torch.save(checkpoint, current_epoch_savepath)

    def _ema_create(self):
        ema_model = copy.deepcopy(self.raw_model)
        ema_model.requires_grad_(False)
        return ema_model

    @torch.no_grad()
    def _ema_update(self):
        for ema_p, raw_p in zip(self.ema_model.parameters(), self.raw_model.parameters()):
            ema_p.mul_(self.ema_decay).add_(raw_p, alpha=1-self.ema_decay)
        for ema_b, raw_b in zip(self.ema_model.buffers(), self.raw_model.buffers()):
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
        for k, v in asdict(self.train_config).items():
            setattr(self, k, v)
