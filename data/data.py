from abc import ABC, abstractmethod

import torch
from datasets import concatenate_datasets, load_dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import ConcatDataset, Dataset

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x * 2.0 - 1.0),
])

class Entity(Dataset, ABC):

    @property
    @abstractmethod
    def entity_ids(self):
        raise NotImplementedError

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

    def __init__(self, data_dir="./datasets", transform=None):
        if transform is None:
            self.transform = TRANSFORM
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

class CelebA(Entity):
    '''CelebA dataset with optional celeb_id filtering. Args: data_dir (str), transform (callable|None), min_celeb_samples (int), size (int), grayscale (bool). Returns: None.'''

    def __init__(self, data_dir="./datasets", transform=None, min_celeb_samples=0, size=128, grayscale=False, random_horizontal_flip=True):
        '''Initialize CelebA dataset and optionally filter celeb_ids. Args: data_dir (str), transform (callable|None), min_celeb_samples (int), size (int), grayscale (bool). Returns: None.'''
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
        dataset = concatenate_datasets([ds["train"], ds["valid"], ds["test"]])
        celeb_ids = torch.tensor(dataset["celeb_id"], dtype=torch.long)
        if min_celeb_samples > 1:
            unique_ids, counts = torch.unique(celeb_ids, return_counts=True)
            keep_ids = set(unique_ids[counts >= min_celeb_samples].tolist())
            celeb_ids_list = celeb_ids.tolist()
            keep_indices = [
                index
                for index, celeb_id in enumerate(celeb_ids_list)
                if celeb_id in keep_ids
            ]
            dataset = dataset.select(keep_indices)
            celeb_ids = torch.tensor(
                [celeb_id for celeb_id in celeb_ids_list if celeb_id in keep_ids],
                dtype=torch.long,
            )
        self.dataset = dataset
        self.celeb_ids = celeb_ids

    @property
    def entity_ids(self):
        return self.celeb_ids

    def __getitem__(self, index):
        '''Return transformed image. Args: index (int). Returns: torch.Tensor.'''
        image = self.dataset[int(index)]["image"]
        image = self.transform(image)
        return image

    def __len__(self):
        '''Return dataset size. Args: None. Returns: int.'''
        return len(self.dataset)

class CelebA2(CelebA):
    '''CelebA dataset filtered to celeb_ids with at least 2 samples. Args: data_dir (str), transform (callable|None), size (int), grayscale (bool). Returns: None.'''

    def __init__(self, data_dir="./datasets", transform=None, size=128, grayscale=False):
        '''Initialize CelebA2 dataset. Args: data_dir (str), transform (callable|None), size (int), grayscale (bool). Returns: None.'''
        super().__init__(
            data_dir=data_dir,
            transform=transform,
            min_celeb_samples=2,
            size=size,
            grayscale=grayscale,
        )

class CelebA2LowRes(CelebA2):
    '''CelebA2 dataset resized and optionally grayscaled. Args: data_dir (str), transform (callable|None), size (int), grayscale (bool). Returns: None.'''

    def __init__(self, data_dir="./datasets", transform=None, size=32, grayscale=True):
        '''Initialize low resolution CelebA2 dataset. Args: data_dir (str), transform (callable|None), size (int), grayscale (bool). Returns: None.'''
        super().__init__(data_dir=data_dir, transform=transform, size=size, grayscale=grayscale)

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
