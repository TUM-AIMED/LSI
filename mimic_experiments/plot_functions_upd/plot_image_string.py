import sys
sys.path.append("/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
from collections import defaultdict
from Datasets.dataset_helper import get_dataset


def get_kl_data(final_path, agg_type, rand=False):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = np.array(final_dict["kl"])[0]
    idx = np.array(final_dict["idx"])[0]
    return kl_data, idx, None

def plot_images(image_paths_set1, image_paths_set2, save_dir, selected_label):
    rows = 5
    cols = 4
    fig, axes = plt.subplots(rows, cols*2 + 1, figsize=(20, 10))
    fontdict = {'fontsize': 2}
    for i, (img1, img2) in enumerate(zip(image_paths_set1, image_paths_set2)):
        row = int(i/cols)
        col = i%cols
        axes[row, col].imshow(img1)
        axes[row, col].axis('off')
        axes[row, col + cols + 1].imshow(img2)
        axes[row, col + cols + 1].axis('off')
    
    for i in range(rows):
        axes[i, cols].axis('off')

    # Add titles to the left block (col 0-3)
    plt.subplot2grid((5, 9), (0, 0), colspan=4, fig=None, box_aspect=0.0001, position=[0.24, 0.9, 0.1, 0.02])
    plt.axis('off')
    plt.title('Low $\mathsf{LSI}$',
          fontsize = 15)

    # Add titles to the right block (col 5-8)
    plt.subplot2grid((5, 9), (0, 5), colspan=4, fig=None, box_aspect=0.0001, position=[0.68, 0.9, 0.1, 0.02])
    plt.axis('off')
    plt.title('High $\mathsf{LSI}$',
          fontsize = 15)

    save_name = save_dir + "images_" + str(selected_label)
    plt.savefig(save_name + ".pdf", format="pdf", dpi=1000)
    print(f"saving fig as {save_name}.pdf")




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
path_str = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2/kl_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000_corrupt_0.0_.pkl"
save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results/"

selected_label = 5


kl_data_list, idx, _ = get_kl_data(path_str, "kl2_diag")

combined_data = list(zip(kl_data_list, idx))
sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
grad_sorted, idx = zip(*sorted_data)

_, labels = get_images_form_idx(list(range(50000)))

idx = [index for index in idx if labels[index] == selected_label]

idx_lower = list(idx[0:20])
idx_higher = list(idx[-20:])
images1, label1 = get_images_form_idx(idx_lower)
images2, label2 = get_images_form_idx(idx_higher)

plot_images(images1, images2, save_dir, selected_label)