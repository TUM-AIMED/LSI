
from run_filtered_dpsgd import train_with_params
import json
from itertools import permutations
from copy import deepcopy
import pandas as pd
import os

params_path = '/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/params/params.json'
params_exp_path = '/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/params/params_experiments.json'

with open(params_path, 'r') as file:
        params = json.load(file)

with open(params_exp_path, 'r') as file:
        params_experiment = json.load(file)

experiment_nr = params_experiment["experiment_nr"]
params_experiment["experiment_nr"] += 1
with open(params_exp_path, 'w') as file:
    json.dump(params_experiment, file, indent=4)

save_path = params_experiment["save_path"]
if not os.path.exists(save_path):
        os.makedirs(save_path)

origin = []
data = []
for topic in params_experiment["variables"]:
        for key in params_experiment["variables"][topic]:
                origin.append([topic, key])
                data.append(params_experiment["variables"][topic][key])

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
    params["model"]["name"] = "experiment_" + str(experiment_nr) + "_" + name
    params["Paths"]["gradient_save_path"] = os.path.join(save_path, "gradients", "experiment_" + str(experiment_nr))
    params["Paths"]["stats_save_path"] = os.path.join(save_path, "stats", "experiment_" + str(experiment_nr))
    train_with_params(params)
