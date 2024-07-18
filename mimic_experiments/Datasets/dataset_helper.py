import torch
from torch.utils.data import DataLoader
from Datasets.dataset_diabetes import DiabetesDataset
from Datasets.dataset_fairface import FairfaceDataset
from Datasets.dataset_mnist import MNISTDataset
from Datasets.dataset_medmnist import (Adrenalmnist3d, 
                                       Bloodmnist, 
                                       Breastmnist, 
                                       Chestmnist, 
                                       Dermamnist, 
                                       Fracturemnist3d, 
                                       Nodulemnist3d, 
                                       Octmnist, 
                                       Organamnist, 
                                       Organcmnist, 
                                       Organmnist3d, 
                                       Organsmnist, 
                                       Pathmnist, 
                                       Pneumoniamnist, 
                                       Retinamnist, 
                                       Synapsemnist3d, 
                                       Tissuemnist, 
                                       Vesselmnist3d)
from Datasets.dataset_cifar10 import CIFAR10
from Datasets.dataset_cifar100 import CIFAR100
from Datasets.dataset_compressed import Compressed
from Datasets.dataset_imagenet import Imagenet
from Datasets.dataset_prima import Prima
from Datasets.dataset_imagenette import ImagenetteDataset

def get_dataset(keyword):
    if keyword == "diabetes":
        return ValueError("Not implemented on server")
    elif keyword == "fairface":
        return FairfaceDataset, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/fairface"
    elif keyword == "mnist":
        return MNISTDataset, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MNIST/MNIST/raw"
    elif keyword == "adrenalmnist3d":
        return Adrenalmnist3d, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "bloodmnist":
        return Bloodmnist, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "breastmnist":
        return Breastmnist, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "chestmnist":
        return Chestmnist, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "dermamnist":
        return Dermamnist, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "fracturemnist3d":
        return Fracturemnist3d, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "nodulemnist3d":
        return Nodulemnist3d, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "octmnist":
        return Octmnist, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "organamnist":
        return Organamnist, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "organcmnist":
        return Organcmnist, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "organmnist3d":
        return Organmnist3d, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "organsmnist":
        return Organsmnist, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "pathmnist":
        return Pathmnist, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "pneumoniamnist":
        return Pneumoniamnist, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "retinamnist":
        return Retinamnist, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "synapsemnist3d":
        return Synapsemnist3d, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "tissuemnist":
        return Tissuemnist, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "vesselmnist3d":
        return Vesselmnist3d, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/MEDMNIST"
    elif keyword == "cifar10":
        return CIFAR10, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/CIFAR10/cifar-10-batches-py"
    elif keyword == "cifar100":
        return CIFAR100, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/CIFAR100/cifar-100-python"
    elif keyword == "cifar10compressed":
        return Compressed, "/vol/aimspace/users/kaiserj/Datasets/Datasets_compressed_by_resnet18_headless/cifar10"
    elif keyword == "cifar100compressed":
        return Compressed, "/vol/aimspace/users/kaiserj/Datasets/Datasets_compressed_by_resnet18_headless/cifar100"
    elif keyword == "cifar100compressedgray":
        return Compressed, "/vol/aimspace/users/kaiserj/Datasets/Datasets_compressed_by_resnet18_headless/cifar100_grayscale"
    elif keyword == "Imagenet":
        return Imagenet, "/vol/aimspace/projects/ILSVRC2012/"
    elif keyword == "Prima":
        return Prima, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/PRIMA"
    elif keyword == "Primacompressed":
        return Compressed, "/vol/aimspace/users/kaiserj/Datasets/Datasets_compressed_by_resnet18_headless/Prima"
    elif keyword == "Imagenette":
        return ImagenetteDataset, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/imagenette2-160"
    elif keyword == "Imagenettecompressed":
        return Compressed, "/vol/aimspace/users/kaiserj/Datasets/Datasets_compressed_by_resnet18_headless/Imagenette"
    elif keyword == "Imagewoof":
        return ImagenetteDataset, "/vol/aimspace/users/kaiserj/Datasets/Datasets_Raw/imagewoof2-160"
    elif keyword == "Imagewoofcompressed":
        return Compressed, "/vol/aimspace/users/kaiserj/Datasets/Datasets_compressed_by_resnet18_headless/Imagewoof"
    elif keyword == "Imdbcompressed":
        return Compressed, "/vol/aimspace/users/kaiserj/Datasets/Datasets_compressed_by_bert_headless/imdb"
    elif keyword == "Lorem":
        return Compressed, "/vol/aimspace/users/kaiserj/Datasets/Datasets_compressed_by_bert_headless/LoremIpsum"
    else:
        raise ValueError("Invalid keyword. Please provide a valid keyword.")
    


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