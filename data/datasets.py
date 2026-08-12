from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
import zipfile

import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
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

class ImageNet(Dataset):
    '''
    ImageNet dataset with optional preprocessing.
    Args:
        data_dir (str): Dataset root directory.
        transform (callable | None): Optional transform applied to each image.
        size (int): Output image size.
        grayscale (bool): Whether to convert images to grayscale.
        random_horizontal_flip (bool): Whether to apply random flips.
    Returns:
        None
    '''

    def __init__(self, data_dir="./datasets", transform=None, size=128, grayscale=False, random_horizontal_flip=True):
        '''
        Initialize the ImageNet dataset.
        Args:
            data_dir (str): Dataset root directory.
            transform (callable | None): Optional transform applied to each image.
            size (int): Output image size.
            grayscale (bool): Whether to convert images to grayscale.
            random_horizontal_flip (bool): Whether to apply random flips.
        Returns:
            None
        '''
        if transform is None:
            transforms = [
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
        self.dataset = ConcatDataset([
            datasets.ImageNet(root=data_dir, split="train", transform=self.transform),
            datasets.ImageNet(root=data_dir, split="val", transform=self.transform),
        ])

    def __getitem__(self, index):
        '''
        Return a transformed image.
        Args:
            index (int): Sample index.
        Returns:
            torch.Tensor: Transformed image tensor.
        '''
        item, _ = self.dataset[index]
        return item

    def __len__(self):
        '''
        Return the dataset size.
        Returns:
            int: Number of samples in the dataset.
        '''
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

class CelebA2(CelebA):
    '''
    CelebA dataset without entities that have fewer than two images.
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
        Initialize the filtered CelebA2 dataset.
        Args:
            data_dir (str): Dataset cache directory.
            transform (callable | None): Optional transform applied to each image.
            size (int): Output image size.
            grayscale (bool): Whether to convert images to grayscale.
            random_horizontal_flip (bool): Whether to apply random flips.
        Returns:
            None
        '''
        super().__init__(
            data_dir=data_dir,
            transform=transform,
            size=size,
            grayscale=grayscale,
            random_horizontal_flip=random_horizontal_flip,
        )
        entity_counts = torch.bincount(self._entity_ids)
        keep_mask = entity_counts[self._entity_ids] >= 2
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
        keep_index_list = keep_indices.tolist()
        self.dataset = self.dataset.select(keep_index_list)
        filtered_entity_ids = self._entity_ids[keep_indices]
        normalized_entity_ids = _normalize_entity_ids(filtered_entity_ids)
        self._entity_ids = normalized_entity_ids
        self._n_entities = int(normalized_entity_ids.max().item()) + 1

class VGGFace2(EntityDataset):
    '''
    VGGFace2 dataset with optional preprocessing.
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
        Initialize the VGGFace2 dataset.
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
        local_repo_dir = Path(data_dir) / "logasja___VGGFace2"
        snapshot_download(
            repo_id="logasja/VGGFace2",
            repo_type="dataset",
            local_dir=str(local_repo_dir),
        )
        train_split = load_dataset(str(local_repo_dir), "256", split="train", cache_dir=data_dir)
        test_split = load_dataset(str(local_repo_dir), "256", split="test", cache_dir=data_dir)
        self.dataset = concatenate_datasets([train_split, test_split])
        entity_ids = torch.tensor(self.dataset["class_id"], dtype=torch.long)
        self._entity_ids = _normalize_entity_ids(entity_ids)
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

class MSMT17(EntityDataset):
    '''
    MSMT17 dataset with optional preprocessing.
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
        Initialize the MSMT17 dataset.
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
                T.Lambda(lambda image: self._resize_and_pad_square(image, size)),
            ]
            if grayscale:
                transforms.append(T.Grayscale(num_output_channels=1))
            if random_horizontal_flip:
                transforms.append(T.RandomHorizontalFlip(p=0.5))
            transforms.append(TRANSFORM)
            self.transform = T.Compose(transforms)
        else:
            self.transform = transform
        local_repo_dir = Path(data_dir) / "xianpeijie___MSMT17_V1"
        snapshot_download(
            repo_id="xianpeijie/MSMT17_V1",
            repo_type="dataset",
            local_dir=str(local_repo_dir),
        )
        archive_path = local_repo_dir / "MSMT17_V1.zip"
        extracted_root = local_repo_dir / "MSMT17_V1"
        if archive_path.exists() and not extracted_root.exists():
            with zipfile.ZipFile(archive_path) as archive_file:
                archive_file.extractall(local_repo_dir)
        search_root = extracted_root
        if not search_root.exists():
            search_root = local_repo_dir
        dataset_root = self._get_dataset_root(search_root)
        train_dir = dataset_root / "train"
        test_dir = dataset_root / "test"
        train_samples = self._load_split_samples(train_dir, dataset_root / "list_train.txt")
        val_samples = self._load_split_samples(train_dir, dataset_root / "list_val.txt")
        query_samples = self._load_split_samples(test_dir, dataset_root / "list_query.txt")
        gallery_samples = self._load_split_samples(test_dir, dataset_root / "list_gallery.txt")
        train_val_entity_ids = {sample["entity_id"] for sample in train_samples + val_samples}
        train_val_entity_count = len(train_val_entity_ids)
        for sample in query_samples:
            sample["entity_id"] += train_val_entity_count
        for sample in gallery_samples:
            sample["entity_id"] += train_val_entity_count
        self.dataset = train_samples + val_samples + query_samples + gallery_samples
        entity_ids = torch.tensor([sample["entity_id"] for sample in self.dataset], dtype=torch.long)
        self._entity_ids = _normalize_entity_ids(entity_ids)
        self._n_entities = int(self._entity_ids.max().item()) + 1

    @staticmethod
    def _resize_and_pad_square(image, size):
        '''
        Resize an image to fit within a square and center-pad the remainder.
        Args:
            image (PIL.Image.Image): Input image.
            size (int): Output square size.
        Returns:
            PIL.Image.Image: Square image with preserved aspect ratio.
        '''
        width, height = image.size
        longer_side = max(width, height)
        scale = size / longer_side
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized_image = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BICUBIC,
        )
        square_image = Image.new(image.mode, (size, size), color=0)
        left = (size - resized_width) // 2
        top = (size - resized_height) // 2
        square_image.paste(resized_image, (left, top))
        return square_image

    @staticmethod
    def _get_dataset_root(extracted_root):
        '''
        Locate the directory containing the MSMT17 split files.
        Args:
            extracted_root (Path): Directory produced by archive extraction.
        Returns:
            Path: Directory containing the split metadata files.
        '''
        split_path = extracted_root / "list_train.txt"
        if split_path.exists():
            return extracted_root
        matching_paths = sorted(extracted_root.rglob("list_train.txt"))
        if matching_paths:
            return matching_paths[0].parent
        raise FileNotFoundError(f"Unable to find MSMT17 split metadata in {extracted_root}.")

    @staticmethod
    def _load_split_samples(image_dir, list_path):
        '''
        Load image paths and entity ids for one MSMT17 split.
        Args:
            image_dir (Path): Directory that contains the split images.
            list_path (Path): Metadata file with image names and entity ids.
        Returns:
            list[dict]: Image paths paired with entity ids.
        '''
        samples = []
        with list_path.open() as handle:
            for raw_line in handle:
                stripped_line = raw_line.strip()
                if not stripped_line:
                    continue
                image_name, entity_id_text = stripped_line.split()
                entity_id = int(entity_id_text)
                image_path = image_dir / image_name
                sample = {
                    "image_path": image_path,
                    "entity_id": entity_id,
                }
                samples.append(sample)
        return samples

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
        sample = self.dataset[int(index)]
        image_path = sample["image_path"]
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
        transformed_image = self.transform(rgb_image)
        return transformed_image

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
