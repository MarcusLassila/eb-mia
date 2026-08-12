from generative_models.agm import AbstractGenerativeModel, AbstractDiffusionModel
from accelerate.accelerate import AcceleratorLite
from utils import unwrap_torch_compile_state_dict

from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Union
import copy
import inspect
import math
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
    lr_scheduler_params: dict = field(default_factory=dict)
    fixed_noise_level_loss: float | None = None

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
        resume_checkpoint: Union[dict, None]=None,
    ):
        self.train_config = train_config
        self.model_config = model_config
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.savepath = Path(savepath)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        assert self.train_dataset[0].shape == self.val_dataset[0].shape

        self.fixed_noise_level_loss = self.train_config.fixed_noise_level_loss
        self.track_fixed_noise_eval = self.fixed_noise_level_loss is not None
        if self.track_fixed_noise_eval:
            assert 0.0 <= self.fixed_noise_level_loss <= 1.0
            assert isinstance(model, AbstractDiffusionModel)

        self.model = model
        self.model.move_to(self.device)
        self.network, self.train_dataloader, self.val_dataloader, self.train_eval_dataloader = accelerator.prepare(
            self.model.network,
            train_dataset,
            val_dataset,
            self.train_config.batch_size,
            make_train_eval_dataloader=True,
        )
        self.model.network = self.network

        assert self.train_config.simul_batch_size % (self.train_config.batch_size * accelerator.world_size) == 0
        self.grad_accum_steps = self.train_config.simul_batch_size // (self.train_config.batch_size * accelerator.world_size)
        if self.grad_accum_steps == 1:
            accelerator.print("No gradient accumulation")
        else:
            accelerator.print(f"Gradient accumulation steps: {self.grad_accum_steps}")

        if self.device.type == "cuda":
            self.autocast_context = torch.autocast(device_type="cuda", dtype=getattr(torch, self.train_config.autocast_dtype))
            self.scaler = torch.GradScaler(device="cuda", enabled=self.train_config.autocast_dtype=="float16") # Only use gradscaler for float16
        else:
            self.autocast_context = nullcontext()
            self.scaler = torch.GradScaler(device="cpu", enabled=False)

        if accelerator.running_ddp:
            self.raw_network = self.model.network.module
        else:
            self.raw_network = self.model.network

        if self.train_config.use_ema:
            self.ema_network = self._ema_create()

        # Use fused AdamW (Adam with weight decay)
        use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and self.device.type == "cuda"
        accelerator.print(f"Using fused AdamW: {use_fused}")
        self.optimizer = torch.optim.AdamW(
            params=self.raw_network.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,
            fused=use_fused,
        )
        self.scheduler = self._get_lr_scheduler(self.optimizer)
        self.start_epoch = 0
        self.train_losses = []
        self.train_eval_losses = []
        self.val_losses = []
        self.fixed_noise_train_losses = []
        self.fixed_noise_val_losses = []
        if resume_checkpoint is not None:
            self._load_resume_checkpoint(resume_checkpoint)
    
    def train(self):
        accelerator = self.accelerator
        step = 0

        # Drop any incomplete accumulated batch for simplicity
        n_batches = len(self.train_dataloader)
        n_full_batches = n_batches - (n_batches % self.grad_accum_steps)
        end_epoch = self.start_epoch + self.train_config.epochs

        for epoch in range(self.start_epoch + 1, end_epoch + 1):
            t0 = time.time()
            if accelerator.running_ddp:
                self.train_dataloader.sampler.set_epoch(epoch)
            self.network.train()
            self.optimizer.zero_grad()
            current_lr = self.optimizer.param_groups[0]["lr"]
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

                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                if self.train_config.grad_clip != 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.train_config.grad_clip)
                else:
                    grad_norm_sq = torch.zeros((), dtype=torch.float32, device=self.device)
                    for parameter in self.network.parameters():
                        if parameter.grad is None:
                            continue
                        grad = parameter.grad.detach()
                        grad_norm_sq += grad.float().pow(2).sum()
                    grad_norm = grad_norm_sq.sqrt()
                if not torch.isfinite(grad_norm).item():
                    raise RuntimeError("Non-finite gradient norm detected.")
                accum_grad_norm += grad_norm
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.train_config.use_ema:
                    self._ema_update()
                if self.train_config.lr_scheduler == "linear":
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                self.optimizer.zero_grad()
                step += 1
                n_simul_train_batches += 1

            assert n_simul_train_batches.item() == n_full_batches // self.grad_accum_steps, "All batches should be consumed"

            if self.train_config.lr_scheduler != "linear":
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]

            eval_network = self.ema_network if self.train_config.use_ema else self.raw_network
            eval_network.eval()
            with torch.no_grad():

                if self.track_fixed_noise_eval:
                    fixed_noise_train_accum_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    fixed_noise_val_accum_loss = torch.zeros((), dtype=torch.float32, device=self.device)

                train_eval_accum_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                n_train_eval_batches = torch.zeros((), dtype=torch.float32, device=self.device)
                for x in self.train_eval_dataloader:
                    train_eval_accum_loss += self.model.loss(x, autocast_context=self.autocast_context, network_override=eval_network)
                    if self.track_fixed_noise_eval:
                        fixed_noise_train_accum_loss += self.model.fixed_noise_level_per_sample_loss(
                            x,
                            noise_level=self.fixed_noise_level_loss,
                            autocast_context=self.autocast_context,
                            network_override=eval_network
                        ).mean()
                    n_train_eval_batches += 1

                val_accum_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                n_val_batches = torch.zeros((), dtype=torch.float32, device=self.device)
                for x in self.val_dataloader:
                    val_accum_loss += self.model.loss(x, autocast_context=self.autocast_context, network_override=eval_network)
                    if self.track_fixed_noise_eval:
                        fixed_noise_val_accum_loss += self.model.fixed_noise_level_per_sample_loss(
                            x,
                            noise_level=self.fixed_noise_level_loss,
                            autocast_context=self.autocast_context,
                            network_override=eval_network
                        ).mean()
                    n_val_batches += 1

            assert n_val_batches.item() == len(self.val_dataloader), "All validation batches should be consumed"

            if accelerator.running_ddp:
                for x in train_accum_loss, train_eval_accum_loss, val_accum_loss, accum_grad_norm, n_simul_train_batches, n_train_eval_batches, n_val_batches:
                    dist.all_reduce(x, op=dist.ReduceOp.SUM)
                if self.track_fixed_noise_eval:
                    for x in fixed_noise_train_accum_loss, fixed_noise_val_accum_loss:
                        dist.all_reduce(x, op=dist.ReduceOp.SUM)

            mean_train_loss = (train_accum_loss / n_simul_train_batches).item()
            mean_train_eval_loss = (train_eval_accum_loss / n_train_eval_batches).item()
            mean_val_loss = (val_accum_loss / n_val_batches).item()
            self.train_losses.append(mean_train_loss)
            self.train_eval_losses.append(mean_train_eval_loss)
            self.val_losses.append(mean_val_loss)
            if self.track_fixed_noise_eval:
                mean_fixed_noise_train_loss = (fixed_noise_train_accum_loss / n_train_eval_batches).item()
                mean_fixed_noise_val_loss = (fixed_noise_val_accum_loss / n_val_batches).item()
                self.fixed_noise_train_losses.append(mean_fixed_noise_train_loss)
                self.fixed_noise_val_losses.append(mean_fixed_noise_val_loss)
            if accelerator.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            log_msg = (
                f"epoch: {epoch} "
                f"| step: {step} "
                f"| train loss: {mean_train_loss:.6f} "
                f"| train eval loss: {mean_train_eval_loss:.6f}"
                f"| val loss: {mean_val_loss:.6f} "
                + (f"| fix train loss: {mean_fixed_noise_train_loss:.6f}" if self.track_fixed_noise_eval else "")
                + (f"| fix val loss: {mean_fixed_noise_val_loss:.6f}" if self.track_fixed_noise_eval else "")
                + (f"| lr: {current_lr:.7f} " if self.train_config.lr_scheduler != "none" else "")
                + f"| grad norm: {(accum_grad_norm / n_simul_train_batches).item():.3f} "
                + f"| dt: {t1 - t0:.1f}"
            )
            accelerator.print(log_msg, flush=True)

            if accelerator.is_master_process and (epoch % self.train_config.epochs_per_checkpoint == 0 or epoch == end_epoch):
                savepath = self.savepath
                name = savepath.stem + f"-epoch{epoch}" + savepath.suffix
                current_epoch_savepath = savepath.parent / name
                checkpoint = {}
                if self.train_config.use_ema:
                    ema_state_dict = unwrap_torch_compile_state_dict(self.ema_network.state_dict())
                    checkpoint["ema_network_state_dict"] = ema_state_dict
                raw_state_dict = unwrap_torch_compile_state_dict(self.raw_network.state_dict())
                checkpoint |= {
                    "epoch": epoch,
                    "network_state_dict": ema_state_dict if self.train_config.use_ema else raw_state_dict,
                    "raw_network_state_dict": raw_state_dict,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),
                    "model_config": self.model_config,
                    "train_config": self.train_config,
                    "train_losses": self.train_losses,
                    "train_eval_losses": self.train_eval_losses,
                    "val_losses": self.val_losses,
                    "train_indices": self.train_dataset.indices,
                    "val_indices": self.val_dataset.indices,
                }
                if self.track_fixed_noise_eval:
                    checkpoint |= {
                        "fixed_noise_train_losses": self.fixed_noise_train_losses,
                        "fixed_noise_val_losses": self.fixed_noise_val_losses,
                        "fixed_noise_level_loss": self.fixed_noise_level_loss,
                    }
                torch.save(checkpoint, current_epoch_savepath)

    def _load_resume_checkpoint(self, checkpoint):
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.train_losses = [float(loss) for loss in checkpoint.get("train_losses", [])]
        self.train_eval_losses = [float(loss) for loss in checkpoint.get("train_eval_losses", [])]
        self.val_losses = [float(loss) for loss in checkpoint.get("val_losses", [])]
        self.fixed_noise_train_losses = [float(loss) for loss in checkpoint.get("fixed_noise_train_losses", [])]
        self.fixed_noise_val_losses = [float(loss) for loss in checkpoint.get("fixed_noise_val_losses", [])]
        if self.fixed_noise_level_loss != checkpoint.get("fixed_noise_level_loss", None):
            msg = "The fixed noise level is not allowed to change when resuming from a checkpoint!\n"
            msg += f"Checkpoint noise level: {checkpoint.get("fixed_noise_level_loss", None)}\n"
            msg += f"Config noise level: {self.fixed_noise_level_loss}"
            raise ValueError(msg)
        raw_state_dict = unwrap_torch_compile_state_dict(checkpoint["raw_network_state_dict"])
        getattr(self.raw_network, "_orig_mod", self.raw_network).load_state_dict(raw_state_dict)
        if self.train_config.use_ema:
            ema_state_dict = unwrap_torch_compile_state_dict(checkpoint.get("ema_network_state_dict", checkpoint["network_state_dict"]))
            getattr(self.ema_network, "_orig_mod", self.ema_network).load_state_dict(ema_state_dict)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.train_config.lr
            group["weight_decay"] = self.train_config.weight_decay
        checkpoint_train_config = checkpoint.get("train_config", None)
        if "scheduler_state_dict" in checkpoint and checkpoint_train_config == self.train_config:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            for group, lr in zip(self.optimizer.param_groups, self.scheduler.get_last_lr()):
                group["lr"] = lr
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

    def _ema_create(self):
        ema_network = copy.deepcopy(self.raw_network)
        ema_network.requires_grad_(False)
        return ema_network

    @torch.no_grad()
    def _ema_update(self):
        for ema_p, raw_p in zip(self.ema_network.parameters(), self.raw_network.parameters()):
            ema_p.mul_(self.train_config.ema_decay).add_(raw_p, alpha=1-self.train_config.ema_decay)
        for ema_b, raw_b in zip(self.ema_network.buffers(), self.raw_network.buffers()):
            ema_b.copy_(raw_b)

    def _get_lr_scheduler(self, optimizer):
        match self.train_config.lr_scheduler:
            case "none":
                return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
            case "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.train_config.epochs, eta_min=1e-6)
            case "linear":
                warmup_steps = self.train_config.lr_scheduler_params.get("warmup_steps", 0)
                min_lr = self.train_config.lr_scheduler_params.get("min_lr", 1e-8)
                max_lr = self.train_config.lr
                total_steps = self.train_config.epochs * math.ceil(len(self.train_dataset) / self.train_config.simul_batch_size)
                assert 0 <= warmup_steps < total_steps
                def lr_multiplier(current_step: int):
                    if current_step < warmup_steps:
                        lr = (max_lr - min_lr) * current_step / warmup_steps + min_lr
                    else:
                        slope = -(max_lr - min_lr) / (total_steps - warmup_steps)
                        intercept = max_lr + (max_lr - min_lr) / (total_steps - warmup_steps) * warmup_steps
                        lr = max(slope * current_step + intercept, min_lr)
                    return lr / max_lr
                return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
            case _:
                raise ValueError(f"Unsupported lr scheduler: {self.train_config.lr_scheduler}")
