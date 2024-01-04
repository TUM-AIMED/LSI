import sys
sys.path.append("/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
from collections import defaultdict
from Datasets.dataset_helper import get_dataset

cifar10_classes = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

def get_kl_data(final_path, agg_type, rand=False):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    idx = final_dict["idx"][0]
    kl_data_list = final_dict["kl"][0]
    return kl_data_list, idx, None

def plot_images(image_paths_set1, image_paths_set2, label1, label2, kl_lower, kl_higher, file_name):
    fig, axes = plt.subplots(2, len(image_paths_set1), figsize=(40, 1))
    fontdict = {'fontsize': 2}
    for i, (img, lab, kl) in enumerate(zip(image_paths_set1, label1, kl_lower)):
        axes[0, i].imshow(img)
        axes[0, i].set_title(lab + " " + "{:.4e}".format(kl), fontdict=fontdict)
        axes[0, i].axis('off')

    for i, (img, lab, kl) in enumerate(zip(image_paths_set2, label2, kl_higher)):
        axes[1, i].imshow(img)
        axes[1, i].set_title(lab + " " + "{:.4e}".format(kl), fontdict=fontdict)
        axes[1, i].axis('off')

    plt.savefig(file_name + ".jpg", dpi = 300)
    print(f"saving fig as ./{file_name}.jpg")



def get_images_form_idx(idxs):
    dataset_class, data_path = get_dataset("cifar10")
    data_set = dataset_class(data_path, train=True)
    # data_set._set_classes([0, 5])
    # data_set._set_classes([4, 9]) # mnist
    images = []
    labels =[]
    labels = data_set.labels[idxs]
    images = data_set.data[idxs]
    images = [im.transpose(1, 2, 0) for im in images]
    return images, labels


# Example usage
kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax/kl_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000_.pkl"
file_name = "z_kl_chain_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000"

kl_data_list, idx, _ = get_kl_data(kl_path, "kl2_diag")
kl1_diag = kl_data_list



combined_data = list(zip(kl1_diag, idx))
sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
kl1_diag, idx = zip(*sorted_data)

idx_lower = list(idx[0:100])
idx_higher = list(idx[-100:])

kl_lower = list(kl1_diag[0:100])
kl_higher = list(kl1_diag[-100:])

images1, label1 = get_images_form_idx(idx_lower)
images2, label2 = get_images_form_idx(idx_higher)

label1 = [cifar10_classes[lab] for lab in label1]
label2 = [cifar10_classes[lab] for lab in label2]

plot_images(images1, images2, label1, label2, kl_lower, kl_higher, file_name)