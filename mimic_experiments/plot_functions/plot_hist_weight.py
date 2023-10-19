import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from Datasets.dataset_helper import get_dataset
import cv2
plt.rcParams.update({
    'font.family': 'serif',
})

file_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_test/gradients/"
file_name = "test_weight_lr_1e_05_2" 
for file_name in os.listdir(file_path):
    file_name = file_name[:-4]
    try:
        with open(file_path + file_name + ".pkl", 'rb') as file:
            data = pickle.load(file)
    except Exception as e:
        print(f"Error loading data from '{file_path}': {str(e)}")
    data = data[-1]
    hist = data["weight_hist"]
    bins = data["weight_bins"]

    bar_widths = [bins[i+1] - bins[i] for i in range(len(bins) - 1)]

    # Create the bar plot
    plt.figure()
    plt.bar(bins[:-1], hist, width=bar_widths, align='edge', color='blue', alpha=0.7)
    plt.title('Weight Distribution Histogram')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(file_path + file_name + ".jpg")
    print(f"saved fig at {file_path + file_name + '.jpg'}")
