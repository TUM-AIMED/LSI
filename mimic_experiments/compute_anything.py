import sys
import os


from models.model_helper import get_model
from Datasets.dataset_helper import get_dataset, CustomDataLoader
from train_methods.train_methods import train
from opacus.validators.module_validator import ModuleValidator

from opacus import PrivacyEngine
from copy import deepcopy

import torch
import numpy as np
import warnings
import json
import argparse
import wandb


def train_with_params(
    params : dict,
    json_file_path
):
    """
    train_with_params initializes the main training parts (model, criterion, optimizer and makes private)

    :param params: dict with all parameters
    """ 
    run = wandb.init(
        project="individual_privacy",
        config = params,
        dir=params["Paths"]["wandb_save_path"],
        tags=["initial_testing"],
        name=params["model"]["name"],
        entity="jkaiser",
        settings=wandb.Settings(_service_wait=8000),
        mode="disabled"
    )

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(params["model"]["seed"])
    warnings.filterwarnings("ignore", message=r".*Using a non-full backward hook.*")

    dataset_class, data_path = get_dataset(params["model"]["dataset_name"])
    model_class = get_model(params["model"]["model"],)

    budget = (params["DP"]["T"] * params["DP"]["max_per_sample_grad_norm"] ** 2) + 1e-3 
    
    # This train loader is for the full batch and for checking all the individual gradient norms
    if params["model"]["split_data"]:
        data_set = dataset_class(data_path, train=True, classes=params["training"]["selected_labels"], portions=params["training"]["balancing_of_labels"], shuffle=False)
    else:
        data_set = dataset_class(data_path, train=True)
    
    if params["Inform"]["remove"] and params["Inform"]["class"] != None:
        class_found = False
        while not class_found:
            _, label, _, _ = data_set.__getitem__(params["Inform"]["idx"])
            if label == params["Inform"]["class"]:
                class_found = True
                print(f"found class {params['Inform']['class']} at index {params['Inform']['idx']}")
                params["model"]["name"] = params["model"]["name_base"] + str(params["model"]["name_num"]) + "_remove_idx_" + str(params["Inform"]["idx"])
                with open(json_file_path, 'w') as file:
                    json.dump(params, file, indent=4)
            else:
                params["Inform"]["idx"] += 1
    if params["Inform"]["remove"]:
        data_set.remove_index_from_data(params["Inform"]["idx"])
        print(f'Remove index {params["Inform"]["idx"]}')
    
    if params["Inform"]["remove"] and params["Inform"]["reorder"]:
        params["Inform"]["firstbatchnum"] = int(params["Inform"]["idx"] / params["training"]["batch_size"])
        compare_folder = params["Paths"]["compare_model_path_base"] + str(params["Inform"]["firstbatchnum"]) + ".pkl"
        for file_name in os.listdir(compare_folder):
            params["Paths"]["compare_model_path"] = os.path.join(compare_folder, file_name)
        print(f'Reordering batches with batch {params["Inform"]["firstbatchnum"]} at first with idx {params["Inform"]["idx"]}')
        print(f'Will compare to {params["Paths"]["compare_model_path"]}')


    if params["Inform"]["reorder"]:
        data_set.batchwise_reorder(params["training"]["batch_size"], params["Inform"]["firstbatchnum"], remove=True)
        print(f'Reordering batches with batch {params["Inform"]["firstbatchnum"]}')
        

    train_loader_0 = torch.utils.data.DataLoader(
        data_set,
        batch_size=params["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


    # This loader contains the data of the test datset
    if params["model"]["split_data"]:
        data_set = dataset_class(data_path, train=False, classes=params["training"]["selected_labels"])
    else:
        data_set = dataset_class(data_path, train=False)
    test_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


    train_X_example, _, _, _ = train_loader_0.dataset[0]
    N = len(train_loader_0.dataset)

    N_CLASSES =  len(np.unique(train_loader_0.dataset.labels))

    if N < params["training"]["batch_size"]:
        raise Exception(f"Batchsize of {params['training']['batch_size']} is larger than the size of the dataset of {N}")
    
    if params["model"]["model"] == "mlp":
        model = ModuleValidator.fix_and_validate(model_class(len(torch.flatten(train_X_example)), N_CLASSES).to(DEVICE))
    else:
        model = ModuleValidator.fix_and_validate(model_class(len(train_X_example), N_CLASSES).to(DEVICE))

    if params["model"]["ELU"]:
        model.replace_all_relu_with_elu()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["training"]["learning_rate"],
        weight_decay=params["training"]["l2_regularizer"],
    )

    privacy_engine = None
    if params["model"]["private"]:
        secure_rng = False   
        privacy_engine = PrivacyEngine(secure_mode=secure_rng)

        model, optimizer, train_loader_0 = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader_0,
            noise_multiplier=params["DP"]["sigma_tilde"],
            max_grad_norm=params["DP"]["max_per_sample_grad_norm"],
            poisson_sampling=False,
        )

        # this is used for computing individual privacy with estimates of gradient norms
        idp_accoutant = PrivacyLossTracker(n_training, args.batchsize, noise_multiplier, init_norm=args.clip, delta=args.delta, rounding=args.rounding)
        idp_accoutant.update_rdp()

    if not os.path.exists(params["Paths"]["gradient_save_path"]):
        os.makedirs(params["Paths"]["gradient_save_path"])
    if not os.path.exists(params["Paths"]["stats_save_path"]):
        os.makedirs(params["Paths"]["stats_save_path"])
    stats_save_path = os.path.join(params["Paths"]["stats_save_path"])


    return train(
        params,
        model,
        DEVICE,
        train_loader_0,
        test_loader,
        optimizer,
        budget,
        criterion,
        N,
        stats_path=stats_save_path,
        privacy_engine=privacy_engine,
        stop_epsilon=None,
    )






if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    os.environ['WANDB_CONFIG_DIR'] = '/vol/aimspace/users/kaiserj/wandb_config/'
    json_file_path = '/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/params/'
    """with open('/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/params/params.json', 'r') as file:
        params = json.load(file)
    wandb.login()"""


    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    
    parser.add_argument("--learning_rate", type=float, help="Value for lerining_rate (optional)")
    parser.add_argument("--l2_regularizer", type=float, help="Value for l2_regularizer (optional)")
    parser.add_argument("--epochs", type=float, help="Value for num_epochs (optional)")
    parser.add_argument("--batchsize", type=float, help="Value for batch_size (optional)")
    parser.add_argument("--clip", type=float, help="Value for clipping threshold (optional)")
    parser.add_argument("--model", type=str, help="Value for clipping threshold (optional)")
    parser.add_argument("--nosave", dest='params["save"]', action='store_false')    
    parser.add_argument("--balancing", action="store", type=float, nargs="+", help="balancing of the dataset (optional)")
    parser.add_argument("--params", type=str, default=None, help="Value for clipping threshold (optional)")
    args = parser.parse_args()

    if args.params != None:
        json_file_path = os.path.join(json_file_path, str(args.params) + ".json")
    else:
        json_file_path = os.path.join(json_file_path, "params.json")
    with open(json_file_path, 'r') as file:
        params = json.load(file)
    params["model"]['name_num'] += 1
    params["model"]["name"] = params["model"]["name_base"] + str(params["model"]["name_num"])
    if params["Inform"]["remove"]:
        params["Inform"]["idx"] +=1
        params["model"]["name"] = params["model"]["name_base"] + str(params["model"]["name_num"]) + "_remove_idx_" + str(params["Inform"]["idx"])
    with open(json_file_path, 'w') as file:
        json.dump(params, file, indent=4)

    if args.learning_rate != None:
        params["training"]["learning_rate"] = args.learning_rate
    if args.l2_regularizer != None:
        params["training"]["l2_regularizer"] = args.l2_regularizer
    if args.epochs != None:
        params["training"]["num_epochs"] = int(args.epochs)
    if args.batchsize != None:
        params["training"]["batch_size"] = int(args.batchsize)
    if args.clip != None:
        params["DP"]["max_per_sample_grad_norm"] = args.clip
    if args.model != None:
        params["model"]["model"] = args.model
    if args.balancing != None:
        params["training"]["balancing_of_labels"] = args.balancing

    print("main")

    train_with_params(
        params,
        json_file_path
    )
