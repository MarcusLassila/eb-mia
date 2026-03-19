import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class BernoulliMixtureModel(nn.Module):

    def __init__(self, in_dim, n_mixtures=256):
        super().__init__()
        self.in_dim = in_dim
        self.n_features = torch.tensor(in_dim).prod()
        self.n_mixtures = n_mixtures
        self.proj_coefficeints = nn.Parameter(torch.randn(n_mixtures, self.n_features))
        self.log_mixture_coefficients = nn.Parameter(torch.randn(n_mixtures))

    def log_Z(self):
        return torch.logsumexp(self.log_mixture_coefficients + self.log_Zc(), dim=0)

    def log_Zc(self):
        return torch.log(1 + self.proj_coefficeints.exp()).sum(dim=1)

    def log_component_mixture_coefficients(self):
        log_coefficients = self.log_mixture_coefficients + self.log_Zc() - self.log_Z()
        return log_coefficients

    def log_likelihood(self, x):
        x = x.view(x.shape[0], self.n_features)
        y = self.log_mixture_coefficients + x @ self.proj_coefficeints.t()
        return torch.logsumexp(y, dim=1) - self.log_Z()

    def regularizer(self):
        #reg = 1e-5 * torch.norm(self.proj_coefficeints)
        log_pi = self.log_component_mixture_coefficients()
        reg = -0.2 * (log_pi.exp() * log_pi).sum()
        return reg

    def per_sample_loss(self, x):
        return -self.log_likelihood(x) + self.regularizer()

    def loss(self, x):
        return self.per_sample_loss(x).mean()

    def forward(self, x):
        return self.log_likelihood(x)

    @torch.inference_mode()
    def sample(self, n_samples):
        components = torch.multinomial(self.log_component_mixture_coefficients().exp(), num_samples=n_samples, replacement=True)
        prob_components = torch.sigmoid(self.proj_coefficeints[components])
        return prob_components.view(n_samples, *self.in_dim)

def train_bmm(model, train_dataloader, val_dataloader, epochs, device, savepath, lr=1e-2):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=1e-4)
    train_loss = []
    val_loss = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        acc_train_loss = 0.0
        for x in train_dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()
            acc_train_loss += loss.detach().item()
        train_loss.append(acc_train_loss / len(train_dataloader))
        lr_scheduler.step()
        model.eval()
        acc_val_loss = 0.0
        with torch.no_grad():
            for x in val_dataloader:
                x = x.to(device)
                loss = model.loss(x)
                acc_val_loss += loss.item()
            val_loss.append(acc_val_loss / len(val_dataloader))
        t1 = time.time()
        log_msg = (
            f"epoch: {epoch}/{epochs} | train loss: {train_loss[-1]:.5f} "
            f"| val loss: {val_loss[-1]:.5f} | time: {t1 - t0:.1f} "
            f"| reg: {model.regularizer():.5f} | lr: {optimizer.param_groups[0]['lr']:.5f}"
        )
        print(log_msg, flush=True)
        if savepath is not None:
            model_checkpoint = {
                "model_state_dict": model.state_dict(),
                "train_indices": train_dataloader.dataset.indices,
                "in_dim": model.in_dim,
                "n_mixtures": model.n_mixtures,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            torch.save(model_checkpoint, savepath)

def plot_image(images, rescale_method="none", name="temp_image"):
    import matplotlib.pyplot as plt
    # Create the 8x8 grid
    plt.style.use("grayscale")
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    axes = axes.flatten()

    for img, ax in zip(images, axes):
        if rescale_method == "tanh":
            img = torch.tanh(img)
        elif rescale_method == "clamp":
            img = torch.clamp(img, 0.0, 1.0)
        elif rescale_method == "none":
            pass
        else:
            raise ValueError("Unsupported rescale method")
        img = img.permute(1, 2, 0)
        ax.imshow(img)
        ax.axis("off")

    fig.tight_layout()
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    from data import datasets as data
    from torch.utils.data import DataLoader, Subset
    device = torch.device("mps")
    dataset = data.BinaryMNIST()
    train_split_index = int(0.5 * len(dataset))
    train_dataset = Subset(dataset, indices=torch.arange(train_split_index))
    val_dataset = Subset(dataset, indices=torch.arange(start=train_split_index, end=len(dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    model = BernoulliMixtureModel(in_dim=dataset[0].shape, n_mixtures=512)
    train_bmm(model, train_dataloader, val_dataloader, epochs=100, device=device, savepath=None, lr=2e-2)
    samples = model.sample(64).cpu().float()
    plot_image(samples)
