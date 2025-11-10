from . import accelerate, models, utils

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

    def __init__(self,
                 beta,
                 channel_mult,
                 image_dim,
                 base_channels=128,
                 dropout=0.0,
                 resample_with_conv=True,
                 do_compile=False,
        ):
        self.image_dim = image_dim
        assert self.image_dim[1] == self.image_dim[2], "Only square images are supported"
        self.accelerator = accelerate.AcceleratorLite(do_compile=do_compile)
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
            loss = F.mse_loss(input=noise_pred, target=eps, reduction="none").sum(dim=(1, 2, 3))
        return loss
    
    def loss(self, x, autocast_context=nullcontext()):
        t = torch.randint(low=0, high=self.T, size=(x.shape[0],)).to(self.device)
        return self.per_sample_loss(x, t, autocast_context).mean()

    def train(self, train_dataset, val_dataset, batch_size, lr, n_epochs, savepath, simul_batch_size=64, grad_clip=1.0, steps_per_print=50):
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
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            scaler = torch.GradScaler(device="cuda")
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

        step = 0
        log_dict = defaultdict(list)
        for epoch in range(1, n_epochs + 1):
            model.train()
            optimizer.zero_grad()
            accum_loss = 0.0
            t0 = time.time()
            for i, x in enumerate(train_dataloader):
                final_grad_accum_step = (i + 1) % grad_accum_steps == 0
                loss = self.loss(x, autocast_context)
                loss /= grad_accum_steps
                accum_loss += loss.detach()
                if accelerator.running_ddp:
                    model.require_backward_grad_sync = final_grad_accum_step
                scaler.scale(loss).backward()
                if not final_grad_accum_step:
                    continue # Keep accumulating gradients

                # Gradient accumulation done, update parameters and log
                if accelerator.running_ddp:
                    dist.all_reduce(accum_loss, op=dist.ReduceOp.AVG)
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize() # Syncronize processes before measuring t1
                t1 = time.time()
                step += 1
                im_per_sec = x.shape[0] * accelerator.world_size / (t1 - t0)
                log_dict["accum_loss"].append(accum_loss.item())
                log_dict["grad_norm"].append(norm.item())
                log_dict["im_per_sec"].append(im_per_sec)
                log_dict["step_time"].append(t1 - t0)
                if step % steps_per_print == 0:
                    assert len(log_dict["accum_loss"]) == steps_per_print
                    log_msg = (
                        f"step: {step} "
                        f"| loss: {mean(log_dict['accum_loss']):.6f} "
                        f"| grad norm: {mean(log_dict['grad_norm']):.2f} "
                        f"| images per sec: {mean(log_dict['im_per_sec']):.1f} "
                        f"| time per step: {mean(log_dict['step_time']):.4f}"
                    )
                    accelerator.print(log_msg, flush=True)
                    log_dict = defaultdict(list)
                accum_loss = 0.0
                t0 = time.time()

            model.eval()
            with torch.no_grad():
                accum_loss = 0.0
                n_batches = torch.tensor([0], dtype=torch.int64) 
                for x in val_dataloader:
                    loss = self.loss(x, autocast_context)
                    accum_loss += loss
                    n_batches += 1
                if accelerator.running_ddp:
                    dist.all_reduce(accum_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)
                val_loss = (accum_loss / n_batches).item()
                print(f"Epoch: {epoch} | validation loss: {val_loss:.6f}")

            if epoch % 10 == 0 or epoch == n_epochs:
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
                }
                torch.save(checkpoint, savepath)

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
