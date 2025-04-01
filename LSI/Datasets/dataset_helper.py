from torch.utils.data import DataLoader
from Datasets.datasets_full import (
    CIFAR10,
    CIFAR100,
    Imagenet,
    Prima,
    ImagenetteDataset,
)
from Datasets.dataset_compressed import CompressedDataset

def get_dataset(keyword, base_path):
    dataset_paths = {
        "cifar10": (CIFAR10, f"{base_path}/CIFAR10/cifar-10-batches-py"),
        "cifar100": (CIFAR100, f"{base_path}/CIFAR100/cifar-100-python"),
        "cifar10compressed": (CompressedDataset, f"{base_path}/cifar10"),
        "cifar100compressed": (CompressedDataset, f"{base_path}/cifar100"),
        "cifar100compressedgray": (CompressedDataset, f"{base_path}/cifar100_grayscale"),
        "Imagenet": (Imagenet, "/vol/aimspace/projects/ILSVRC2012/"),
        "Prima": (Prima, f"{base_path}/PRIMA"),
        "Primacompressed": (CompressedDataset, f"{base_path}/Prima"),
        "Imagenette": (ImagenetteDataset, f"{base_path}/imagenette2-160"),
        "Imagenettecompressed": (CompressedDataset, f"{base_path}/Imagenette"),
        "Imagewoof": (ImagenetteDataset, f"{base_path}/imagewoof2-160"),
        "Imagewoofcompressed": (CompressedDataset, f"{base_path}/Imagewoof"),
        "Imdbcompressed": (CompressedDataset, f"{base_path}/imdb"),
        "Loremcompressed": (CompressedDataset, f"{base_path}/LoremIpsum2"),
    }

    if keyword in dataset_paths:
        return dataset_paths[keyword]
    else:
        raise ValueError(f"Invalid keyword '{keyword}'. Please provide a valid keyword.")
    


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, firstbatchnum=None, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        if firstbatchnum is None:
            self.batch_order = list(range(len(self)))
        else:
            batch_order = list(range(len(self)))
            [batch_order[firstbatchnum]] + batch_order[:firstbatchnum] + batch_order[firstbatchnum+1:]
            self.batch_order = batch_order

    def __iter__(self):
        return iter(self.batch_order)