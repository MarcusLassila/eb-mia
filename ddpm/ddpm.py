from . import models, utils
from accelerate.accelerate import AcceleratorLite

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import tqdm

import inspect
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from statistics import mean

class DDPM:

    def __init__(
        self,
        beta,
        channel_mult,
        image_dim,
        base_channels=128,
        dropout=0.0,
        resample_with_conv=True,
        accelerator=None,
        torch_compile=False,
    ):
        self.image_dim = image_dim
        assert self.image_dim[1] == self.image_dim[2], "Only square images are supported"
        if accelerator is None:
            self.accelerator = AcceleratorLite(torch_compile=torch_compile)
        else:
            self.accelerator = accelerator
        self.device = self.accelerator.device
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.T = beta.shape[0]
        self.beta = beta.to(self.device)
        self.alpha_bar = torch.cumprod(1 - self.beta, dim=0)
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.dropout = dropout
        self.resample_with_conv = resample_with_conv
        self.model = models.UNet(
            image_size=self.image_dim[1],
            in_channels=self.image_dim[0],
            out_channels=self.image_dim[0],
            base_channels=base_channels,
            channel_mult=channel_mult,
            dropout=dropout,
            resample_with_conv=resample_with_conv,
        ).to(self.device) # Model that predict noise
        self.accelerator.print(f"Using a U-net model with {utils.count_params(self.model)['n_params']:_} parameters")
        self.accelerator.print(f"Num trainabe parameters: {utils.count_params(self.model)['n_trainable_params']:_}")

    def per_sample_loss(self, x, t, autocast_context=nullcontext()):
        eps = torch.randn(x.shape).to(self.device)
        alpha_bar_t = self.alpha_bar[t].view(x.shape[0], 1, 1, 1)
        z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
        with autocast_context:
            noise_pred = self.model(z, t)
            loss = F.mse_loss(input=noise_pred, target=eps, reduction="none").mean(dim=(1, 2, 3))
        return loss
    
    def loss(self, x, autocast_context=nullcontext()):
        t = torch.randint(low=0, high=self.T, size=(x.shape[0],)).to(self.device)
        return self.per_sample_loss(x, t, autocast_context).mean()

    def train(self, train_dataset, val_dataset, batch_size, lr, n_epochs, savepath, simul_batch_size=64, grad_clip=1.0, epochs_per_checkpoint=200, autocast_dtype="bfloat16"):
        accelerator = self.accelerator
        input_dim = train_dataset[0].shape
        assert input_dim == self.image_dim == val_dataset[0].shape

        assert simul_batch_size % (batch_size * accelerator.world_size) == 0
        grad_accum_steps = simul_batch_size // (batch_size * accelerator.world_size)
        if grad_accum_steps == 1:
            accelerator.print("No gradient accumulation")
        else:
            accelerator.print(f"Gradient accumulation steps: {grad_accum_steps}")

        if self.device.type == "cuda":
            autocast_context = torch.autocast(device_type="cuda", dtype=getattr(torch, autocast_dtype))
            scaler = torch.GradScaler(device="cuda", enabled=autocast_dtype=="float16") # Only use gradscaler for float16
        else:
            autocast_context = nullcontext()
            scaler = torch.GradScaler(device="cpu", enabled=False)

        model, train_dataloader, val_dataloader = accelerator.prepare(self.model, train_dataset, val_dataset, batch_size)
        if accelerator.running_ddp:
            raw_model = model.module
        else:
            raw_model = model

        # Use fused AdamW (Adam with weight decay)
        use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and self.device.type == "cuda"
        accelerator.print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, fused=use_fused)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

        step = 0
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            if accelerator.running_ddp:
                train_dataloader.sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad()
            train_accum_loss = 0.0
            accum_grad_norm = 0.0
            n_steps = torch.zeros((), dtype=torch.float32, device=self.device)
            for i, x in enumerate(train_dataloader):
                final_grad_accum_step = (i + 1) % grad_accum_steps == 0
                loss = self.loss(x, autocast_context)
                loss /= grad_accum_steps
                train_accum_loss += loss.detach()
                if accelerator.running_ddp:
                    model.require_backward_grad_sync = final_grad_accum_step
                scaler.scale(loss).backward()
                if not final_grad_accum_step:
                    continue # Keep accumulating gradients

                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    accum_grad_norm += norm
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1
                n_steps += 1

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            model.eval()
            with torch.no_grad():
                val_accum_loss = 0.0
                n_val_batches = torch.zeros((), dtype=torch.float32, device=self.device)
                for x in val_dataloader:
                    val_accum_loss += self.loss(x, autocast_context)
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
                + (f"| grad norm: {(accum_grad_norm / n_steps).item():.3f} " if grad_clip != 0 else "")
                + f"| dt: {t1 - t0:.1f}"
            )
            accelerator.print(log_msg, flush=True)

            if accelerator.is_master_process and (epoch % epochs_per_checkpoint == 0 or epoch == n_epochs):
                savepath = Path(savepath)
                name = savepath.stem + f"-epoch{epoch}" + savepath.suffix
                current_epoch_savepath = savepath.parent / name
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "beta": self.beta,
                    "channel_mult": self.channel_mult,
                    "base_channels": self.base_channels,
                    "image_dim": self.image_dim,
                    "dropout": self.dropout,
                    "resample_with_conv": self.resample_with_conv,
                    "train_indices": train_dataset.indices,
                    "val_indices": val_dataset.indices,
                }
                torch.save(checkpoint, current_epoch_savepath)

    def load(self, state_dict):
        state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.inference_mode()
    def sample(self, batch_size, disable_tqdm=False):
        ''' Sample a batch of images and rescale them to floating point values in [0,1]. '''
        self.model.eval()
        x = torch.randn(batch_size, *self.image_dim).to(self.device)
        for t in tqdm(range(self.T - 1, -1, -1), disable=disable_tqdm, desc=f"Sampling {batch_size} images on {self.device}"):
            if t > 0:
                z = torch.randn(batch_size, *self.image_dim).to(self.device)
            else:
                z = torch.zeros(size=(batch_size, *self.image_dim)).to(self.device)
            sigma_t = torch.sqrt(self.beta[t])
            noise_pred = self.model(x, t * torch.ones(batch_size, dtype=torch.long).to(self.device))
            x = (x - self.beta[t] * noise_pred / torch.sqrt(1 - self.alpha_bar[t])) / torch.sqrt(1 - self.beta[t]) + sigma_t * z
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)
        return x
