from Datasets.dataset_diabetes import DiabetesDataset
from Datasets.dataset_fairface import FairfaceDataset
from Datasets.dataset_mnist_unbalanced import MNISTDataset
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
def get_dataset(keyword):
    if keyword == "diabetes":
        return ValueError("Not implemented on server")
    elif keyword == "fairface":
        return FairfaceDataset, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/fairface"
    elif keyword == "mnist":
        return ValueError("Not implemented on server")
    elif keyword == "adrenalmnist3d":
        return Adrenalmnist3d, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "bloodmnist":
        return Bloodmnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "breastmnist":
        return Breastmnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "chestmnist":
        return Chestmnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "dermamnist":
        return Dermamnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "fracturemnist3d":
        return Fracturemnist3d, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "nodulemnist3d":
        return Nodulemnist3d, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "octmnist":
        return Octmnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "organamnist":
        return Organamnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "organcmnist":
        return Organcmnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "organmnist3d":
        return Organmnist3d, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "organsmnist":
        return Organsmnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "pathmnist":
        return Pathmnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "pneumoniamnist":
        return Pneumoniamnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "retinamnist":
        return Retinamnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "synapsemnist3d":
        return Synapsemnist3d, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "tissuemnist":
        return Tissuemnist, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    elif keyword == "vesselmnist3d":
        return Vesselmnist3d, "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/MEDMNIST"
    else:
        raise ValueError("Invalid keyword. Please provide a valid keyword.")