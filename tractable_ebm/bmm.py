import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

class BernoulliMixtureModel(nn.Module):

    def __init__(self, in_dim, n_mixtures=300):
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

    def component_mixture_coefficients(self):
        log_coefficients = self.log_mixture_coefficients + self.log_Zc() - self.log_Z()
        return log_coefficients.exp()

    def log_likelihood(self, x):
        x = torch.flatten(x, start_dim=1)
        y = self.log_mixture_coefficients + x @ self.proj_coefficeints.t()
        return torch.logsumexp(y, dim=1) - self.log_Z()

    def per_sample_loss(self, x):
        return -self.log_likelihood(x)

    def loss(self, x):
        return self.per_sample_loss(x).mean()

    def forward(self, x):
        return self.log_likelihood(x)

    @torch.inference_mode()
    def sample(self, n_samples):
        components = torch.multinomial(self.component_mixture_coefficients(), num_samples=n_samples, replacement=True)
        prob_components = torch.sigmoid(self.proj_coefficeints[components])
        return torch.bernoulli(prob_components).view(n_samples, *self.in_dim)

def train_bmm(model, train_dataloader, val_dataloader, epochs, device, savepath, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    val_loss = []
    for epoch in range(1, epochs + 1):
        model.train()
        acc_train_loss = 0.0
        for x in tqdm(train_dataloader, disable=device.type=="cuda", desc=f"epoch: {epoch}"):
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()
            acc_train_loss += loss.detach().item()
        train_loss.append(acc_train_loss / len(train_dataloader))
        model.eval()
        acc_val_loss = 0.0
        with torch.no_grad():
            for x in val_dataloader:
                x = x.to(device)
                loss = model.loss(x)
                acc_val_loss += loss.item()
            val_loss.append(acc_val_loss / len(val_dataloader))
        log_msg = f"epoch: {epoch}/{epochs} | train loss: {train_loss[-1]:.5f} | val loss: {val_loss[-1]:.5f}"
        print(log_msg, flush=True)
        model_checkpoint = {
            "model_state_dict": model.state_dict(),
            "train_indices": train_dataloader.dataset.indices,
            "in_dim": model.in_dim,
            "n_mixtures": model.n_mixtures,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(model_checkpoint, savepath)

if __name__ == "__main__":
    from data import data
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(torch.bernoulli),
    ])
    dataset = data.MNIST(transform=transform)
    n_features = torch.tensor(dataset[0].shape).prod()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    x = next(iter(dataloader))
    model = BernoulliMixtureModel(n_features=n_features)
    samples = model.sample(6)
    print(samples.shape)
