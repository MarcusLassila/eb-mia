import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from pathlib import Path
import time

def train_vae(model, train_dataloader, val_dataloader, epochs, device, lr, savepath, weight_decay=0.01):
    model.to(device)
    total_steps = len(train_dataloader) * epochs
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    early_stopping_counter = 0
    early_stopping_threshold = 50
    train_loss = []
    val_loss = []
    min_val_loss = torch.inf
    total_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        acc_train_loss = 0.0
        avg_step_time = 0.0
        for x in tqdm(train_dataloader, disable=device.type=="cuda", desc=f"epoch: {epoch}"):
            optimizer.zero_grad()
            x = x.to(device)
            loss = model.loss(x)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_item = loss.item()
            acc_train_loss += loss_item
            t1 = time.time()
            avg_step_time += t1 - t0
            t0 = time.time()
        train_loss.append(acc_train_loss / len(train_dataloader))
        avg_step_time /= len(train_dataloader)

        model.eval()
        acc_val_loss = 0.0
        with torch.no_grad():
            for x in val_dataloader:
                x = x.to(device)
                loss = model.loss(x)
                acc_val_loss += loss.item()
        val_loss.append(acc_val_loss / len(val_dataloader))

        log_msg = " | ".join([
            f"epoch: {epoch}/{epochs}",
            f"train loss: {train_loss[-1]:.6f}",
            f"val loss {val_loss[-1]:.6f}",
            f"step time: {avg_step_time:.4f}",
            f"lr: {optimizer.param_groups[0]['lr']:.7f}"
        ])
        print(log_msg, flush=True)
        if val_loss[-1] < min_val_loss:
            early_stopping_counter = 0
            min_val_loss = val_loss[-1]
            model_checkpoint = {
                "model_state_dict": model.state_dict(),
                "train_indices": train_dataloader.dataset.indices,
                "in_ch": model.in_ch,
                "in_dim": model.in_dim,
                "latent_dim": model.latent_dim,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            torch.save(model_checkpoint, savepath)
        else:
            early_stopping_counter += 1
        if early_stopping_counter == early_stopping_threshold:
            print(f"Early stopping at epoch={epoch}")
            break
    total_time = time.time() - total_time
    print(f"Total train time: {total_time}")
