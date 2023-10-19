import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import cv2
plt.rcParams.update({
    'font.family': 'serif',
})


def get_kl_data(final_path, agg_type):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = final_dict[agg_type]
    idx = list(kl_data.keys())
    idx = [int(index) for index in idx]
    idx.sort()
    kl_data = [kl_data[str(index)] for index in idx]
    return kl_data, idx


kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_final_kl_all/final.pkl"

kl_data_max, idx = get_kl_data(kl_path, "kl2_max")

    # Create the bar plot
plt.figure()
plt.hist(kl_data_max, bins="auto")
plt.title('KL Distribution Histogram')
plt.xlabel('KL Value')
plt.ylabel('Count')
plt.grid(True)
plt.savefig("z_kl2_max_hist" + ".jpg")
print(f"saved fig at {'z_kl2_max_hist' + '.jpg'}")
