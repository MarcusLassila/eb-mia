import torch
from datasets import load_dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import ConcatDataset, Dataset

class MNIST(Dataset):
    
    def __init__(self, transform):
        self.transform = transform
        self.dataset = ConcatDataset([
            datasets.MNIST(root='./datasets', train=True, download=True, transform=self.transform),
            datasets.MNIST(root='./datasets', train=False, download=True, transform=self.transform),
        ])

    def __getitem__(self, index):
        item, _ = self.dataset[index] # Discard labels
        return item

    def __len__(self):
        return len(self.dataset)

class CIFAR10(Dataset):

    def __init__(self, transform):
        self.transform = transform
        self.dataset = ConcatDataset([
            datasets.CIFAR10(root='./datasets', train=True, download=True, transform=self.transform),
            datasets.CIFAR10(root='./datasets', train=False, download=True, transform=self.transform),
        ])

    def __getitem__(self, index):
        item, _ = self.dataset[index] # Discard labels
        return item

    def __len__(self):
        return len(self.dataset)
    
class CelebAHQ(Dataset):

    def __init__(self, transform):
        self.transform = transform
        self.dataset = ConcatDataset([
            load_dataset("korexyz/celeba-hq-256x256", split="train"),
            load_dataset("korexyz/celeba-hq-256x256", split="validation"),
        ])

    def __getitem__(self, index):
        image = self.dataset[index]["image"]
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.dataset)

class Flowers(Dataset):

    def __init__(self, transform):
        self.transform = transform
        self.dataset = load_dataset("huggan/flowers-102-categories")["train"]

    def __getitem__(self, index):
        image = self.dataset[index]["image"]
        image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    transform = T.ToTensor()
    dataset = CIFAR10(transform)
    sample = dataset[0]
    print(len(dataset))
    print(sample.shape)
    import matplotlib.pyplot as plt
    plt.imshow(sample.permute(1, 2, 0))
    plt.show()