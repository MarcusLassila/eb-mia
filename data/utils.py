from .datasets import *

def load_dataset(dataset_name, data_dir="./datasets", transform=None, **kwargs):
    '''
    Load a dataset by name.
    Args:
        dataset_name (str): Name of the dataset to load.
        data_dir (str): Directory used for dataset storage and caching.
        transform (callable | None): Optional transform applied to each sample.
        kwargs (dict): Dataset-specific keyword arguments.
    Returns:
        Dataset: Loaded dataset instance.
    '''
    match dataset_name:
        case "MNIST":
            dataset = MNIST(data_dir=data_dir, transform=transform)
        case "CIFAR10":
            dataset = CIFAR10(
                data_dir=data_dir,
                transform=transform,
                random_horizontal_flip=kwargs.get("random_horizontal_flip", True),
            )
        case "CelebA":
            dataset = CelebA(
                data_dir=data_dir,
                transform=transform,
                size=kwargs.get("size", 128),
                grayscale=kwargs.get("grayscale", False),
                random_horizontal_flip=kwargs.get("random_horizontal_flip", True),
            )
        case "CelebA2":
            dataset = CelebA2(
                data_dir=data_dir,
                transform=transform,
                size=kwargs.get("size", 128),
                grayscale=kwargs.get("grayscale", False),
                random_horizontal_flip=kwargs.get("random_horizontal_flip", True),
            )
        case "VGGFace2":
            dataset = VGGFace2(
                data_dir=data_dir,
                transform=transform,
                size=kwargs.get("size", 128),
                grayscale=kwargs.get("grayscale", False),
                random_horizontal_flip=kwargs.get("random_horizontal_flip", True),
            )
        case "MSMT17":
            dataset = MSMT17(
                data_dir=data_dir,
                transform=transform,
                size=kwargs.get("size", 128),
                grayscale=kwargs.get("grayscale", False),
                random_horizontal_flip=kwargs.get("random_horizontal_flip", True),
            )
        case "CelebAHQ":
            dataset = CelebAHQ(
                data_dir=data_dir,
                transform=transform,
                random_horizontal_flip=kwargs.get("random_horizontal_flip", True),
            )
        case "Flowers":
            dataset = Flowers(
                data_dir=data_dir,
                transform=transform,
                random_horizontal_flip=kwargs.get("random_horizontal_flip", True),
            )
        case _:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset
