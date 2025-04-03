from compute_anything import train_with_params
import json
import os

if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    NTIMES = 50


    os.environ['WANDB_CONFIG_DIR'] = '/vol/aimspace/users/kaiserj/wandb_config/'
    json_file_path = '/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/params/params.json'
    """with open('/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/params/params.json', 'r') as file:
        params = json.load(file)
    wandb.login()"""


    for i in range(NTIMES):
        print("-----------------")
        print(f"Run number {i}")
        print("-----------------")
        with open(json_file_path, 'r') as file:
            params = json.load(file)
        params["model"]['name_num'] += 1
        params["model"]["name"] = params["model"]["name_base"] + str(params["model"]["name_num"])
        if params["Inform"]["remove"]:
            params["Inform"]["idx"] +=1
            params["model"]["name"] = params["model"]["name_base"] + str(params["model"]["name_num"]) + "_remove_idx_" + str(params["Inform"]["idx"])
        with open(json_file_path, 'w') as file:
            json.dump(params, file, indent=4)
        # TODO change compare_file_to_correct_one
        train_with_params(
            params,
            json_file_path
        )