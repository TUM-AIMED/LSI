from torch.utils.data import DataLoader
from LSI.Datasets.datasets_full import (
    CIFAR10,
    CIFAR100,
    Imagenet,
    Prima,
    ImagenetteDataset,
)
from LSI.Datasets.dataset_compressed import CompressedDataset
import json

def set_path_compressed(name, path, json_path):
    try:
        with open(json_path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    data[name + "_compressed"] = path
    with open(json_path, "w") as file:
        json.dump(data, file, indent=4)

def get_dataset_compressed(keyword, base_path, json_path):
    with open(json_path, "r") as file:
        dataset_config = json.load(file)

    if keyword + "_compressed" not in dataset_config:
        return None, None
    return f"{base_path}{dataset_config[keyword + '_compressed']}", f"{dataset_config[keyword + '_compressed']}"

def get_dataset(keyword, base_path, json_path):
    with open(json_path, "r") as file:
        dataset_config = json.load(file)

    dataset_paths = {
        "cifar10": (CIFAR10, f"{base_path}{dataset_config['cifar10']}"),
        "cifar100": (CIFAR100, f"{base_path}{dataset_config['cifar100']}"),
        "Imagenet": (Imagenet, f"{base_path}{dataset_config['Imagenet']}"),
        "Prima": (Prima, f"{base_path}/{dataset_config['Prima']}"),
        "Imagenette": (ImagenetteDataset, f"{base_path}{dataset_config['Imagenette']}"),
        "Imagewoof": (ImagenetteDataset, f"{base_path}{dataset_config['Imagewoof']}"),

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