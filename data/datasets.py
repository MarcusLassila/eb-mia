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

class EntityDataset(Dataset, ABC):

    @property
    @abstractmethod
    def entity_ids(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def unique_entity_ids(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def n_entities(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def max_entity_id(self):
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

class CelebA(EntityDataset):
    '''CelebA dataset with optional celeb_id filtering. Args: data_dir (str), transform (callable|None), size (int), grayscale (bool). Returns: None.'''

    def __init__(self, data_dir="./datasets", transform=None, size=128, grayscale=False, random_horizontal_flip=True):
        '''Initialize CelebA dataset and optionally filter celeb_ids. Args: data_dir (str), transform (callable|None), size (int), grayscale (bool). Returns: None.'''
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
        celeb_ids = torch.tensor(self.dataset["celeb_id"], dtype=torch.long)
        self._celeb_ids = celeb_ids
        self._unique_entity_ids = torch.unique(celeb_ids, sorted=True)
        self._n_entities = self._unique_entity_ids.shape[0]
        self._max_entity_id = self._unique_entity_ids[-1]

    @property
    def entity_ids(self):
        return self._celeb_ids

    @property
    def unique_entity_ids(self):
        return self._unique_entity_ids

    @property
    def n_entities(self):
        return self._n_entities

    @property
    def max_entity_id(self):
        return self._max_entity_id

    def __getitem__(self, index):
        '''Return transformed image. Args: index (int). Returns: torch.Tensor.'''
        image = self.dataset[int(index)]["image"]
        image = self.transform(image)
        return image

    def __len__(self):
        '''Return dataset size. Args: None. Returns: int.'''
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


########################
####### LEGACY #########
########################

class CelebA2(CelebA):
    '''CelebA dataset filtered to celeb_ids with at least 2 samples. Args: data_dir (str), transform (callable|None), size (int), grayscale (bool). Returns: None.'''

    def __init__(self, data_dir="./datasets", transform=None, size=128, grayscale=False, random_horizontal_flip=True):
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
        min_celeb_samples = 2
        ds = load_dataset("flwrlabs/celeba", cache_dir=data_dir)
        dataset = concatenate_datasets([ds["train"], ds["valid"], ds["test"]])
        celeb_ids = torch.tensor(dataset["celeb_id"], dtype=torch.long)
        unique_ids, counts = torch.unique(celeb_ids, return_counts=True)
        if min_celeb_samples > 1:
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
        self._celeb_ids = celeb_ids
        unique_id_list = sorted(torch.unique(celeb_ids).tolist())
        self._n_entities = len(unique_id_list)
        self._max_entity_id = unique_id_list[-1]
        self._index_to_entity_id_map = unique_id_list
        self._entity_id_to_index_map = {entity_id: i for i, entity_id in enumerate(unique_id_list)}

    @property
    def entity_ids(self):
        return self._celeb_ids

    @property
    def n_entities(self):
        return self._n_entities

    @property
    def max_entity_id(self):
        return self._max_entity_id

    @property
    def index_to_entity_id_map(self):
        return self._index_to_entity_id_map

    @property
    def entity_id_to_index_map(self):
        return self._entity_id_to_index_map

    def __getitem__(self, index):
        '''Return transformed image. Args: index (int). Returns: torch.Tensor.'''
        image = self.dataset[int(index)]["image"]
        image = self.transform(image)
        return image

    def __len__(self):
        '''Return dataset size. Args: None. Returns: int.'''
        return len(self.dataset)
