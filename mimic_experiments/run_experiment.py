
from run_test_filtered_dpsgd_compare_loss import train_with_params
import json
from itertools import permutations
from copy import deepcopy
import pandas as pd

save_path = "C:\Promotion\Code\Individual_DP\individual-accounting-gdp\mimic_experiments\results\exp_runs"

with open('./params/params.json', 'r') as file:
        params = json.load(file)

with open('./params/params_experiments.json', 'r') as file:
        params_experiment = json.load(file)

origin = []
data = []
for topic in params_experiment:
        for key in params_experiment[topic]:
                origin.append([topic, key])
                data.append(params_experiment[topic][key])

name = "exp"
all_combinations = []
all_combinations = data[0]
all_combinations = [[data] for data in all_combinations]
all_names = [name + "_" + str(i) for i, data in enumerate(all_combinations)]
for i in range(1, len(data)):
    n_lists = []
    all_names_new = []
    for j in range(len(data[i])):
        n_lists.extend(deepcopy(all_combinations))
        all_names_new.extend(deepcopy(all_names))
    for j, added_attribute in enumerate(data[i]):
        for k in range(len(all_combinations)):
               n_lists[j*len(all_combinations) + k].append(added_attribute)
               all_names_new[j*len(all_combinations) + k] = all_names_new[j*len(all_combinations) + k] + "_" + str(j)
    all_combinations = n_lists
    all_names = all_names_new

columnname = [name[1] for name in origin]
df = pd.DataFrame(data=all_combinations, columns=columnname)
df["name"] = all_names

df.to_pickle(save_path + name +".pkl")

for (exp_params, name) in zip(all_combinations, all_names):
    for (exp_param, location) in zip(exp_params, origin):
        params[location[0]][location[1]] = exp_param
    params["model"]["name"] = name
    train_with_params(params)
