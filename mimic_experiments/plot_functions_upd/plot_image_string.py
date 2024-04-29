import sys
sys.path.append("/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
from collections import defaultdict
from Datasets.dataset_helper import get_dataset


# def get_kl_data(final_path, agg_type, rand=False):
#     with open(final_path, 'rb') as file:
#         final_dict = pickle.load(file)
#     kl_data = np.array(final_dict["kl"])[0]
#     idx = np.array(final_dict["idx"])[0]
#     return kl_data, idx, None

def get_kl_data(paths, len_dataset=50000):
    idxs = []
    kls = []
    for batch, path in enumerate(paths):
        with open(path, 'rb') as file:
            final_dict = pickle.load(file)
            kl_data = np.array(final_dict["kl"])
            kl_data = np.mean(kl_data, axis=0)
            idx = list(range(len(kl_data)))
            idx = [idx_i + 10000 * batch for idx_i in idx]
            kls.extend(kl_data)
            idxs.extend(idx)
    while len(kls) < len_dataset:
        kls.append(kls[-1])
        idxs.append(idxs[-1] + 1)
    sorted_data = sorted(zip(kls, idxs), key=lambda x: x[0])
    kl_data, idx = zip(*sorted_data)
    idx = list(idx)
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
    plt.savefig(save_name + ".png", format="png", dpi=1000)
    print(f"saving fig as {save_name}.png")




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

base_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2/"
paths = [
    "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_0_10000_corrupt_0.0_torch.pkl",
    "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_10000_20000_corrupt_0.0_torch.pkl",
    "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_20000_30000_corrupt_0.0_torch.pkl",
    "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_30000_40000_corrupt_0.0_torch.pkl",
    "kl_jax_torch_1000_remove_10000_dataset_cifar10compressed_subset_50000_range_40000_49999_corrupt_0.0_torch.pkl"
]

# paths = [
#     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_0_10000_corrupt_0.0_torch.pkl",
#     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_10000_20000_corrupt_0.0_torch.pkl",
#     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_20000_30000_corrupt_0.0_torch.pkl",
#     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_30000_40000_corrupt_0.0_torch.pkl",
#     "kl_jax_torch_1000_remove_10000_dataset_cifar100compressed_subset_50000_range_40000_49999_corrupt_0.0_torch.pkl"
# ]

paths = [base_path + path for path in paths]


# Example usage
save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results_new/"

selected_label = 5

 
kl_data_list, idx, _ = get_kl_data(paths)

combined_data = list(zip(kl_data_list, idx))
sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
grad_sorted, idx = zip(*sorted_data)

_, labels = get_images_form_idx(list(range(50000)))

for selected_label in range(10):
    idx_c = [index for index in idx if labels[index] == selected_label]

    idx_lower = list(idx_c[0:20])
    idx_higher = list(idx_c[-20:])
    images1, label1 = get_images_form_idx(idx_lower)
    images2, label2 = get_images_form_idx(idx_higher)

    plot_images(images1, images2, save_dir, selected_label)