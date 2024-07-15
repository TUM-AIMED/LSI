import sys
sys.path.append("/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
from collections import defaultdict
from Datasets.dataset_helper import get_dataset
cifar100_classes = {
    0: 'apple',
    1: 'aquarium_fish',
    2: 'baby',
    3: 'bear',
    4: 'beaver',
    5: 'bed',
    6: 'bee',
    7: 'beetle',
    8: 'bicycle',
    9: 'bottle',
    10: 'bowl',
    11: 'boy',
    12: 'bridge',
    13: 'bus',
    14: 'butterfly',
    15: 'camel',
    16: 'can',
    17: 'castle',
    18: 'caterpillar',
    19: 'cattle',
    20: 'chair',
    21: 'chimpanzee',
    22: 'clock',
    23: 'cloud',
    24: 'cockroach',
    25: 'couch',
    26: 'crab',
    27: 'crocodile',
    28: 'cup',
    29: 'dinosaur',
    30: 'dolphin',
    31: 'elephant',
    32: 'flatfish',
    33: 'forest',
    34: 'fox',
    35: 'girl',
    36: 'hamster',
    37: 'house',
    38: 'kangaroo',
    39: 'keyboard',
    40: 'lamp',
    41: 'lawn_mower',
    42: 'leopard',
    43: 'lion',
    44: 'lizard',
    45: 'lobster',
    46: 'man',
    47: 'maple_tree',
    48: 'motorcycle',
    49: 'mountain',
    50: 'mouse',
    51: 'mushroom',
    52: 'oak_tree',
    53: 'orange',
    54: 'orchid',
    55: 'otter',
    56: 'palm_tree',
    57: 'pear',
    58: 'pickup_truck',
    59: 'pine_tree',
    60: 'plain',
    61: 'plate',
    62: 'poppy',
    63: 'porcupine',
    64: 'possum',
    65: 'rabbit',
    66: 'raccoon',
    67: 'ray',
    68: 'road',
    69: 'rocket',
    70: 'rose',
    71: 'sea',
    72: 'seal',
    73: 'shark',
    74: 'shrew',
    75: 'skunk',
    76: 'skyscraper',
    77: 'snail',
    78: 'snake',
    79: 'spider',
    80: 'squirrel',
    81: 'streetcar',
    82: 'sunflower',
    83: 'sweet_pepper',
    84: 'table',
    85: 'tank',
    86: 'telephone',
    87: 'television',
    88: 'tiger',
    89: 'tractor',
    90: 'train',
    91: 'trout',
    92: 'tulip',
    93: 'turtle',
    94: 'wardrobe',
    95: 'whale',
    96: 'willow_tree',
    97: 'wolf',
    98: 'woman',
    99: 'worm'
}


def get_kl_data(final_path, agg_type, rand=False):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    kl_data = final_dict[agg_type]
    idx = list(kl_data[0].keys())
    idx = [int(index) for index in idx]
    idx.sort()
    kl_data_dict = defaultdict(list)
    for seed in kl_data:
        for index in idx:
            kl_data_dict[index].append(kl_data[seed][index])
    kl_data_list = [value for key, value in kl_data_dict.items()]
    if rand: 
        return kl_data_list, idx, final_dict["random_labels_idx"]
    return kl_data_list, idx, None

def plot_images(image_paths_set1, image_paths_set2, label1, label2, file_name):
    fig, axes = plt.subplots(2, len(image_paths_set1), figsize=(200, 1))
    fontdict = {'fontsize': 2}
    for i, (img, lab) in enumerate(zip(image_paths_set1, label1)):
        axes[0, i].imshow(img)
        axes[0, i].set_title(lab, fontdict=fontdict)
        axes[0, i].axis('off')

    for i, (img, lab) in enumerate(zip(image_paths_set2, label2)):
        axes[1, i].imshow(img)
        axes[1, i].set_title(lab, fontdict=fontdict)
        axes[1, i].axis('off')

    plt.savefig(file_name + ".jpg", dpi = 300)
    print(f"saving fig as ./{file_name}.jpg")


def get_correct(final_path):
    with open(final_path, 'rb') as file:
        final_dict = pickle.load(file)
    pred_data = final_dict["correct_data"]
    idx = list(pred_data[0].keys())
    pred_data_dict = defaultdict(list)
    for seed in pred_data:
        for index in idx:
            pred_data_dict[index].append(np.concatenate(pred_data[seed][index]))
    predict_own = {}
    for index, index_data in pred_data_dict.items():
        correct = 0
        for run_data in index_data:
            test = run_data[idx.index(index)]
            if run_data[idx.index(index)] == True:
                correct += 1
            else:
                continue
        predict_own[index] = correct
    predict_own_list = [value/len(pred_data) for key, value in predict_own.items()]
    all_pred = []
    for index, index_data in pred_data_dict.items():
        for run_data in index_data:
            all_pred.append(run_data)
    all_pred = np.stack(all_pred)
    count_per_column = np.sum(all_pred, axis=0)
    count_per_column = count_per_column/all_pred.shape[0]
    idx = list(predict_own.keys())
    return predict_own_list, count_per_column, idx


def get_images_form_idx(idxs):
    dataset_class, data_path = get_dataset("cifar100")
    data_set = dataset_class(data_path, train=True)
    # data_set._set_classes([0, 5])
    # data_set._set_classes([4, 9]) # mnist
    images = []
    labels =[]
    labels = data_set.labels[idxs]
    images = data_set.data[idxs]
    images = [im.transpose(1, 2, 0) for im in images]
    return images, labels

def get_grad_data(final_path):
    with open(final_path, 'rb') as file:
        grad_data = pickle.load(file)
    idxs = []
    result = []
    for seed, seed_data in grad_data.items():
        seed_wise_results = []
        idx = []
        for index, idx_data in seed_data.items():
            idx.append(index)
            seed_wise_results.append(np.sqrt(np.sum(idx_data)))
        result.append(seed_wise_results)
        idxs.append(idx)
    if not all(inner_list == idxs[0] for inner_list in idxs):
        raise Exception("some mix up")
    return np.array(idxs[0]), np.mean(result, axis=0)



# Example usage
kl_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_script/results_cifar100compressedgray_logreg4_3_1000__200_1000_0.9500convex_SGD_Streetcar_vs_Lobster_DELETE/results_all.pkl"
file_name = "z_kl_chain_grads"
grad_path = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_idp/results_cifar100compressed_logreg4_1_100.0_1e-09__1000_50000/results.pkl"

kl_data_list, idx, _ = get_kl_data(kl_path, "kl2_diag")
kl1_diag = np.abs(kl_data_list)
kl1_diag = [np.mean(data) for data in kl1_diag]

predict_own_list, count_per_column, idx2 = get_correct(final_path=kl_path)

idx, grad = get_grad_data(grad_path)

combined_data = list(zip(grad, idx))
sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
grad_sorted, idx = zip(*sorted_data)


# # Sanity check
# if idx != idx2:
#     raise Exception("Sanity check failed")

# count_per_column = [count_per_column[i] for i in range(len(idx2)) if idx[i] == idx2[i] ]



# combined_data = list(zip(kl1_diag, idx, predict_own_list, count_per_column))
# # sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
# sorted_data = sorted(combined_data, key=lambda x: x[2])

# kl1_diag, idx, predict_own_list, count_per_column = zip(*sorted_data)

idx_lower = list(idx[0:1000])
idx_higher = list(idx[-1000:])

grad_sorted_lower = list(grad_sorted[0:500])
grad_sorted_higher = list(grad_sorted[-500:])

images1, label1 = get_images_form_idx(idx_lower)
images2, label2 = get_images_form_idx(idx_higher)

label1 = [cifar100_classes[lab] for lab in label1]
label2 = [cifar100_classes[lab] for lab in label2]

plot_images(images1, images2, label1, label2, file_name)