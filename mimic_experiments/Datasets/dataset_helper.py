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


def get_dataset(keyword):
    if keyword == "diabetes":
        return ValueError("Not implemented on server")
    elif keyword == "fairface":
        return FairfaceDataset, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/fairface"
    elif keyword == "mnist":
        return MNISTDataset, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MNIST/MNIST/raw"
    elif keyword == "adrenalmnist3d":
        return Adrenalmnist3d, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "bloodmnist":
        return Bloodmnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "breastmnist":
        return Breastmnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "chestmnist":
        return Chestmnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "dermamnist":
        return Dermamnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "fracturemnist3d":
        return Fracturemnist3d, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "nodulemnist3d":
        return Nodulemnist3d, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "octmnist":
        return Octmnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "organamnist":
        return Organamnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "organcmnist":
        return Organcmnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "organmnist3d":
        return Organmnist3d, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "organsmnist":
        return Organsmnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "pathmnist":
        return Pathmnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "pneumoniamnist":
        return Pneumoniamnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "retinamnist":
        return Retinamnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "synapsemnist3d":
        return Synapsemnist3d, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "tissuemnist":
        return Tissuemnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "vesselmnist3d":
        return Vesselmnist3d, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/MEDMNIST"
    elif keyword == "cifar10":
        return CIFAR10, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/CIFAR10/cifar-10-batches-py"
    elif keyword == "cifar100":
        return CIFAR100, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/Datasets_Raw/CIFAR100/cifar-100-python"
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