from abc import ABC, abstractmethod
from collections import defaultdict

import torch
from datasets import concatenate_datasets, load_dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import ConcatDataset, Dataset

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x * 2.0 - 1.0),
])

def _normalize_entity_ids(entity_ids: torch.Tensor):
    '''
    Map entity ids to 0,1,...,n_entities-1.
    Args:
        entity_ids (torch.Tensor): Entity id of data point `i` is `entity_ids[i]`.
    Returns:
        torch.Tensor: Normalized entity-id tensor.
    '''
    _, normalized_entity_ids = torch.unique(entity_ids, sorted=True, return_inverse=True)
    return normalized_entity_ids.to(dtype=torch.long)

class EntityDataset(Dataset, ABC):

    @property
    @abstractmethod
    def entity_ids(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def n_entities(self):
        raise NotImplementedError

    def get_entity_index_table(self):
        table = defaultdict(list)
        for idx, id in enumerate(self.entity_ids):
            table[id.item()].append(idx)
        return table

class MNIST(Dataset):
    
    def __init__(self, data_dir="./datasets", transform=None):
        if transform is None:
            self.transform = T.Compose([
                T.Resize(32),  # Resize to 32 for architectural convenience
                TRANSFORM,
            ])
        else:
            self.transform = transform
        self.dataset = ConcatDataset([
            datasets.MNIST(root=data_dir, train=True, download=True, transform=self.transform),
            datasets.MNIST(root=data_dir, train=False, download=True, transform=self.transform),
        ])

    def __getitem__(self, index):
        item, _ = self.dataset[index] # Discard labels
        return item

    def __len__(self):
        return len(self.dataset)

class CIFAR10(Dataset):

    def __init__(self, data_dir="./datasets", transform=None, random_horizontal_flip=True):
        if transform is None:
            transform = []
            if random_horizontal_flip:
                transform.append(T.RandomHorizontalFlip(p=0.5))
            transform.append(TRANSFORM)
            self.transform = T.Compose(transform)
        else:
            self.transform = transform
        self.dataset = ConcatDataset([
            datasets.CIFAR10(root=data_dir, train=True, download=True, transform=self.transform),
            datasets.CIFAR10(root=data_dir, train=False, download=True, transform=self.transform),
        ])

    def __getitem__(self, index):
        item, _ = self.dataset[index] # Discard labels
        return item

    def __len__(self):
        return len(self.dataset)

class CelebA(EntityDataset):
    '''
    CelebA dataset with optional preprocessing.
    Args:
        data_dir (str): Dataset cache directory.
        transform (callable | None): Optional transform applied to each image.
        size (int): Output image size.
        grayscale (bool): Whether to convert images to grayscale.
        random_horizontal_flip (bool): Whether to apply random flips.
    Returns:
        None
    '''

    def __init__(self, data_dir="./datasets", transform=None, size=128, grayscale=False, random_horizontal_flip=True):
        '''
        Initialize the CelebA dataset.
        Args:
            data_dir (str): Dataset cache directory.
            transform (callable | None): Optional transform applied to each image.
            size (int): Output image size.
            grayscale (bool): Whether to convert images to grayscale.
            random_horizontal_flip (bool): Whether to apply random flips.
        Returns:
            None
        '''
        if transform is None:
            transforms = [
                T.CenterCrop((178, 178)),
                T.Resize(
                    (size, size),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
            if grayscale:
                transforms.append(T.Grayscale(num_output_channels=1))
            if random_horizontal_flip:
                transforms.append(T.RandomHorizontalFlip(p=0.5))
            transforms.append(TRANSFORM)
            self.transform = T.Compose(transforms)
        else:
            self.transform = transform
        ds = load_dataset("flwrlabs/celeba", cache_dir=data_dir)
        self.dataset = concatenate_datasets([ds["train"], ds["valid"], ds["test"]])
        self._entity_ids = _normalize_entity_ids(torch.tensor(self.dataset["celeb_id"], dtype=torch.long))
        self._n_entities = int(self._entity_ids.max().item()) + 1

    @property
    def entity_ids(self):
        return self._entity_ids

    @property
    def n_entities(self):
        return self._n_entities

    def __getitem__(self, index):
        '''
        Return a transformed image.
        Args:
            index (int): Sample index.
        Returns:
            torch.Tensor: Transformed image tensor.
        '''
        image = self.dataset[int(index)]["image"]
        image = self.transform(image)
        return image

    def __len__(self):
        '''
        Return the dataset size.
        Returns:
            int: Number of samples in the dataset.
        '''
        return len(self.dataset)

class CelebAHQ(Dataset):

    def __init__(self, data_dir="./datasets", transform=None, random_horizontal_flip=True):
        if transform is None:
            transforms = []
            if random_horizontal_flip:
                transforms.append(T.RandomHorizontalFlip(p=0.5))
            transforms.append(TRANSFORM)
            self.transform = T.Compose(transforms)
        else:
            self.transform = transform
        self.dataset = ConcatDataset([
            load_dataset("korexyz/celeba-hq-256x256", split="train", cache_dir=data_dir),
            load_dataset("korexyz/celeba-hq-256x256", split="validation", cache_dir=data_dir),
        ])

    def __getitem__(self, index):
        image = self.dataset[int(index)]["image"]
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.dataset)

class Flowers(Dataset):

    def __init__(self, data_dir="./datasets", transform=None, random_horizontal_flip=True):
        if transform is None:
            transforms = []
            if random_horizontal_flip:
                transforms.append(T.RandomHorizontalFlip(p=0.5))
            transforms.append(TRANSFORM)
            self.transform = T.Compose(transforms)
        else:
            self.transform = transform
        self.dataset = load_dataset("huggan/flowers-102-categories", cache_dir=data_dir)["train"]

    def __getitem__(self, index):
        image = self.dataset[index]["image"]
        image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    dataset = CelebA()
    sample = dataset[1]
    sample = 0.5 * (sample + 1.0)
    print(len(dataset))
    print(sample.shape)
    import matplotlib.pyplot as plt
    plt.imshow(sample.permute(1, 2, 0))
    plt.show()
