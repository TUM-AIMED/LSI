import sys
sys.path.append("/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments")

import numpy as np
import pickle
from datasets import load_dataset
from Datasets.dataset_helper import get_dataset
import torch

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

# paths = ["/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_after_workshop/kl_jax_torch_1000_remove_25000_dataset_Imdbcompressed_model_Tinymodel_subset_25000_range_0_25000_corrupt_0.0_corrupt_data_0.0_0_torch.pkl"]
paths = ["/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_torch_upd2_after_workshop/kl_jax_torch_1000_remove_25000_dataset_Imdbcompressed_model_Tinymodel_subset_25000_range_0_25000_corrupt_0.0_corrupt_data_0.2_0_torch.pkl"]
length_dataset = 30000
dataset = load_dataset('imdb')
datatext = dataset["train"]["text"]
datalabels = dataset["train"]["label"]


n_corrupt = int(25000 * 0.2)
lorem_data_set_class, lorem_data_path = get_dataset("Loremcompressed")
lorem_dataset = lorem_data_set_class(lorem_data_path, train=True)
X_corrupt = lorem_dataset.data[0:n_corrupt]
y_corrupt = torch.cat((torch.ones(n_corrupt // 2, dtype=torch.int), torch.zeros(n_corrupt // 2, dtype=torch.int)))
y_corrupt = y_corrupt[torch.randperm(n_corrupt)]



# Example usage
save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results_new/"

selected_label = 0

 
kl_data_list, idx, _ = get_kl_data(paths, len_dataset=length_dataset)

high_lsi_idx = idx[-25:]
low_lsi_idx = idx[0:25]

high_lsi = kl_data_list[-25:]
low_lsi = kl_data_list[0:25]

high_lsi_text = [datatext[i] for i in high_lsi_idx]
high_lsi_labels = [datalabels[i] for i in high_lsi_idx]

low_lsi_text = [datatext[i] for i in low_lsi_idx]
low_lsi_labels = [datalabels[i] for i in low_lsi_idx]

print("HIGH LSI")
for (lsi, label, text) in zip(high_lsi, high_lsi_labels, high_lsi_text):
    print(f"{lsi:.10f} with label {label} \n      {text}")
print("\n \n")
print("LOW LSI")
for (lsi, label, text) in zip(low_lsi, low_lsi_labels, low_lsi_text):
    print(f"{lsi:.10f} with label {label} \n      {text} \n")
