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
from tqdm import tqdm
import jax.numpy as jnp

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


plt.rcParams.update({
    'font.family': 'serif',
})

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def get_kl_data(final_path, agg_type, rand=False):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    idx = final_dict["idx"][0]
    kl_data_list = final_dict["kl"][0]
    return kl_data_list, idx, None

def get_accurate(accurate_path, idx_batches):
    with open(accurate_path, 'rb') as file:
        final_dict = pickle.load(file)
    prediction = final_dict["predictions"]
    idx = final_dict["idx"]
    ys = final_dict["label"]

    prediction = prediction.transpose((1, 0, 2))
    accuracy = []
    idx_sanity = []
    for i, pred, label in tqdm(zip(idx, prediction, ys)):
        corr_pred = pred[:, int(label)]
        accuracy.append(corr_pred)
        idx_sanity.append(i)
    assert idx == idx_sanity
    return jnp.array(accuracy)

def plot_images(image_paths_set1, image_paths_set2, label1, label2, add_val1, add_val2, file_name):
    fig, axes = plt.subplots(2, len(image_paths_set1), figsize=(60, 2))
    fontdict = {'fontsize': 2}
    for i, (img, lab, addv) in enumerate(zip(image_paths_set1, label1, add_val1)):
        axes[0, i].imshow(img)
        axes[0, i].set_title(lab + " " + "{:.4e}".format(addv), fontdict=fontdict)
        axes[0, i].axis('off')

    for i, (img, lab, addv) in enumerate(zip(image_paths_set2, label2, add_val2)):
        axes[1, i].imshow(img)
        axes[1, i].set_title(lab + " " + "{:.4e}".format(addv), fontdict=fontdict)
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



def main():
    kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax/kl_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000_.pkl"
    predictions_path ="/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_normal_train/kl_jax_epochs_1000_dataset_cifar10compressed_subset_50000_.pkl"
    file_name = "z_heat_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000"
    file_name2 = "z_image_string_intr_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000"

    rand = True
    accuracy = get_accurate(predictions_path, None)
    kl1_diag, init_idx, rand_labels_idx = get_kl_data(kl_path, "kl1_diag", rand=rand)

    combined_data = list(zip(kl1_diag, accuracy, init_idx))
    sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
    kl1_diag, accuracy, idx = zip(*sorted_data)
    accuracy = jnp.array(list(accuracy))

    plt.figure(figsize=(50, 6))
    plt.imshow(accuracy, cmap='gray', aspect='auto') # interpolation='none',
    plt.xlabel("id")
    plt.ylabel("train_steps")
    plt.savefig(file_name + ".jpg")
    print(f"saving fig as ./{file_name}.jpg")

    num_images = 100
    wrong_first_idx = []
    correct_last_idx = []
    correct_last_kl = []
    wrong_first_kl = []
    for i, acc, kl in zip(idx, accuracy, kl1_diag):
        if acc[-1] < 0.01:
            wrong_first_idx.append(i)
            wrong_first_kl.append(kl)
        if len(wrong_first_idx) > num_images:
            break


    for i, acc, kl in zip(reversed(idx), reversed(accuracy), reversed(kl1_diag)):
        if acc[-1] > 0.99:
            correct_last_idx.append(i)
            correct_last_kl.append(kl)
        if len(correct_last_idx) > num_images:
            break
    print("")

    images1, label1 = get_images_form_idx(wrong_first_idx)
    images2, label2 = get_images_form_idx(correct_last_idx)

    label1 = [cifar10_classes[lab] for lab in label1]
    label2 = [cifar10_classes[lab] for lab in label2]

    plot_images(images1, images2, label1, label2, wrong_first_kl, correct_last_kl, file_name2)

main()