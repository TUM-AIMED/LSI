import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
######################
# No Spearman correlation detected
######################
# Replace 'your_file.npz' with the actual file path you want to open
file_path = '/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/C_Score/cifar10-cscores-orig-order.npz'

# Load the NPZ file
data = np.load(file_path)
c_data = data["scores"]

kl_path_str = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_jax_upd2/kl_jax_epochs_1000_remove_50000_dataset_cifar10compressed_subset_50000_corrupt_0.0_.pkl"
with open(kl_path_str, 'rb') as file:
    final_dict = pickle.load(file)
kl_data = np.array(final_dict["kl"])
kl_data = np.squeeze(kl_data)

rho, p_value = spearmanr(c_data, kl_data)

# Print the result
print(f"Spearman rank coefficient: {rho}")
print(f"P-value: {p_value}")
# Create a scatterplot
plt.scatter(kl_data, c_data, marker='.', color='blue', alpha=0.5)

# Add labels and title
plt.xlabel('LSI')
plt.ylabel('C-Score')

save_dir = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/plot_functions_upd/results/"
# Save the scatterplot as a PDF
save_name = save_dir + "c_vs_kl"
plt.savefig(save_name + ".png", format="png", dpi=1000)
print(f"saving fig as {save_name}.png")

print("")