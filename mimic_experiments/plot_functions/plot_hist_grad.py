import os
import io
import pickle
import torch
import sys
sys.path.append("/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from matplotlib import animation
from Datasets.dataset_helper import get_dataset
import cv2
from collections import defaultdict


plt.rcParams.update({
    'font.family': 'serif',
})

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def get_grad_data(final_path):
    with open(final_path, 'rb') as file:
        grad_data = pickle.load(file)
        grad_data = [data.numpy() for data in grad_data]
    return np.array(grad_data)



def main():
    kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_idp/results_cifar10_resnet18_10_100.0__500_10000/results.pkl"
    file_name = "z_hist_grad_results_cifar10_resnet18_10_100.0__500_100000"
    grad_data = get_grad_data(kl_path)

    medians = np.median(grad_data, axis=0)

    plt.figure(figsize=(8, 6))
    plt.hist(medians, bins="auto", edgecolor='blue', alpha=0.7)
    plt.xlabel("KL1 - Diag")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
    plt.savefig(file_name + ".jpg")
    print(f"saving fig as ./{file_name}.jpg")

main()