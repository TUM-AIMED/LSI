from Datasets.dataset_diabetes import DiabetesDataset
from Datasets.dataset_fairface import FairfaceDataset
from Datasets.dataset_mnist_unbalanced import MNISTDataset

def get_dataset(keyword):
    if keyword == "diabetes":
        return DiabetesDataset, "C:\Promotion\Code\Individual_DP\DIABETES_dataset"
    elif keyword == "fairface":
        return FairfaceDataset, "C:\Promotion\Code\Individual_DP\FAIRFACE"
    elif keyword == "mnist":
        return MNISTDataset, "C:\Promotion\Code\Individual_DP\MNIST_ORG"
    else:
        raise ValueError("Invalid keyword. Please provide a valid keyword.")