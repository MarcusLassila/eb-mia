import torch
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import ConcatDataset, DataLoader, Dataset

class MNIST(Dataset):
    
    def __init__(self):
        self.transform = T.Compose([
            T.ToTensor(),
        ])
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
    
    def __init__(self):
        self.transform = T.Compose([
            T.ToTensor(),
        ])
        self.dataset = ConcatDataset([
            datasets.CIFAR10(root='./datasets', train=True, download=True, transform=self.transform),
            datasets.CIFAR10(root='./datasets', train=False, download=True, transform=self.transform),
        ])

    def __getitem__(self, index):
        item, _ = self.dataset[index] # Discard labels
        return item

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    dataset = CIFAR10()
    sample = dataset[0]
    print(len(dataset))
    print(sample.shape)
    import matplotlib.pyplot as plt
    plt.imshow(sample.permute(1,2,0))
    plt.show()