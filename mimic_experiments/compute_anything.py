import sys
import os


from models.model_helper import get_model
from Datasets.dataset_helper import get_dataset
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
    params : dict
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
        settings=wandb.Settings(_service_wait=3000)
    )

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(params["model"]["seed"])
    warnings.filterwarnings("ignore", message=r".*Using a non-full backward hook.*")

    dataset_class, data_path = get_dataset(params["model"]["dataset_name"])
    model_class = get_model(params["model"]["model"],)

    budget = (params["DP"]["T"] * params["DP"]["max_per_sample_grad_norm"] ** 2) + 1e-3 
    
    # This train loader is for the full batch and for checking all the individual gradient norms
    if params["model"]["split_data"]:
        data_set = dataset_class(data_path, train=True, classes=params["training"]["selected_labels"], portions=params["training"]["balancing_of_labels"], shuffle=True)
    else:
        data_set = dataset_class(data_path, train=True)
    train_loader_0 = torch.utils.data.DataLoader(
        data_set,
        batch_size=params["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # This train loader is the one that will be filtered. It will be replaced by a new one at each iteration
    if params["model"]["split_data"]:
        data_set = dataset_class(data_path, train=True, classes=params["training"]["selected_labels"], portions=params["training"]["balancing_of_labels"], shuffle=True)
    else:
        data_set = dataset_class(data_path, train=True)
    train_loader = torch.utils.data.DataLoader(
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

    N_CLASSES =  len(np.unique(train_loader.dataset.labels))

    if N < params["training"]["batch_size"]:
        raise Exception(f"Batchsize of {params['training']['batch_size']} is larger than the size of the dataset of {N}")
    
    if params["model"]["private"]:
        model = ModuleValidator.fix_and_validate(model_class(len(train_X_example), N_CLASSES).to(DEVICE))
    else:
        model = model_class(len(train_X_example), N_CLASSES).to(DEVICE)

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
    json_file_path = '/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/params/params.json'
    """with open('/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/mimic_experiments/params/params.json', 'r') as file:
        params = json.load(file)
    wandb.login()"""

    with open(json_file_path, 'r') as file:
        params = json.load(file)
    params["model"]['name_num'] += 1
    params["model"]["name"] = params["model"]["name_base"] + str(params["model"]["name_num"])
    with open(json_file_path, 'w') as file:
        json.dump(params, file, indent=4)

    

    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    
    parser.add_argument("--learning_rate", type=float, default=params["training"]["learning_rate"], help="Value for lerining_rate (optional)")
    parser.add_argument("--l2_regularizer", type=float, default=params["training"]["l2_regularizer"], help="Value for l2_regularizer (optional)")
    parser.add_argument("--epochs", type=float, default=params["training"]["num_epochs"], help="Value for num_epochs (optional)")
    parser.add_argument("--batchsize", type=float, default=params["training"]["batch_size"], help="Value for batch_size (optional)")
    parser.add_argument("--clip", type=float, default=params["DP"]["max_per_sample_grad_norm"], help="Value for clipping threshold (optional)")
    parser.add_argument("--model", type=str, default=params["model"]["model"], help="Value for clipping threshold (optional)")
    parser.add_argument("--nosave", dest='params["save"]', action='store_false')    
    parser.add_argument("--balancing", action="store", type=float, nargs="+", default=params["training"]["balancing_of_labels"], help="balancing of the dataset (optional)")
    args = parser.parse_args()

    params["training"]["learning_rate"] = args.learning_rate
    params["training"]["l2_regularizer"] = args.l2_regularizer
    params["training"]["num_epochs"] = int(args.epochs)
    params["training"]["batch_size"] = int(args.batchsize)
    params["DP"]["max_per_sample_grad_norm"] = args.clip
    params["model"]["model"] = args.model
    params["training"]["balancing_of_labels"] = args.balancing


    train_with_params(
        params
    )
