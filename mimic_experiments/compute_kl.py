import sys
import os


from models.model_helper import get_model
from Datasets.dataset_helper import get_dataset, CustomDataLoader
from opacus.validators.module_validator import ModuleValidator
from utils.idp_tracker import PrivacyLossTracker
from utils.data_utils import pretty_print_dict
from utils.data_logger import (
    log_data_epoch, log_data_final, save_data_to_pickle, log_realized_gradients)

from opacus import PrivacyEngine
from copy import deepcopy

import torch
import numpy as np
import warnings
import json
import argparse
import wandb
import time


def test(DEVICE, model, test_set):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_set:
            inputs, target, _, _ = data
            inputs, target = inputs.to(DEVICE), target.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    print(f"testing accuracy: {accuracy}")
    return 


def normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               train_loader_active):
    # Run full batch, sum up the gradients. Then we filter and replace the train_loader
    total = 0
    correct = 0
    max_physical_batch_size = 100
    if params["Inform"]["remove"]:
        batchsize = params["training"]["batch_size"]
        iteration_containing_dropped_idx = int(params["Inform"]["idx"]/batchsize)
        dataset = train_loader_active.dataset
        dataset_pre = deepcopy(dataset)
        dataset_mid = deepcopy(dataset)
        dataset_aft = deepcopy(dataset)
        pre = [*range(0, iteration_containing_dropped_idx * batchsize, 1)]
        mid = [*range(iteration_containing_dropped_idx * batchsize, (iteration_containing_dropped_idx + 1) * batchsize - 1)]
        aft = [*range((iteration_containing_dropped_idx + 1) * batchsize - 1, len(dataset.data))]
        dataset_pre.reduce_to_active(pre)
        dataset_mid.reduce_to_active(mid)
        dataset_aft.reduce_to_active(aft)
        train_loader_pre = torch.utils.data.DataLoader(dataset_pre, batch_size=params["training"]["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
        train_loader_mid = torch.utils.data.DataLoader(dataset_mid, batch_size=params["training"]["batch_size"] - 1, shuffle=False, num_workers=0, pin_memory=False)
        train_loader_aft = torch.utils.data.DataLoader(dataset_aft, batch_size=params["training"]["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
        loss_list = []
        for data_loader in [train_loader_pre, train_loader_mid, train_loader_aft]:
            for _, (data, target, idx, _) in enumerate(data_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()

                output = model(data)
                loss = criterion(output, target)
                loss_list.append(loss)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                loss.backward()
                optimizer.step()
                if model.clip_weights != None:
                    model.clip_weights(params["training"]["clip_weight_value"])

    else:
        loss_list = []
        train_loader_new = train_loader_active
        for _, (data, target, idx, _) in enumerate(train_loader_new):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss_list.append(loss)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            loss.backward()
            optimizer.step()
            if model.clip_weights != None:
                model.clip_weights(params["training"]["clip_weight_value"])
    accuracy = correct/total
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    print(f"training accuracy {accuracy}")


def train(
    params,
    model,
    DEVICE,
    train_loader_0,
    test_loader,
    optimizer,
    criterion
):

    # Compute all the individual norms (actually the squared norms squares are saved here)
    print('Hello world', file=sys.stderr, flush=True)
    print(f'File name: {params["model"]["name"]}')
    print(f'lr: {params["training"]["learning_rate"]}', flush=True)
    print(f'l2: {params["training"]["l2_regularizer"]}', flush=True)
    print(f'epochs: {params["training"]["num_epochs"]}', flush=True)
    print(f'batchsize: {params["training"]["batch_size"]}', flush=True)
    recorded_data = []

    step_count = 0

    for epoch in range(1, params["training"]["num_epochs"] + 1):
        start_epoch = time.time()
        model.train()
        print(epoch, flush=True)
        normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               train_loader_0)

        if epoch % params["testing"]["test_every"] == 0:
            test(DEVICE, model, test_loader)
        
        end_epoch = time.time()
        recorded_data.append(log_data_epoch(
            params,
            DEVICE,
            model,
            optimizer,
            train_loader_0,
            train_loader_0,
            test_loader,
            criterion,
            epoch,
            0,
            100,
        ))
        print(f"Epoch took {end_epoch-start_epoch}")


    recorded_data.append(log_data_final(params, 
                                        DEVICE,
                                        train_loader_0, 
                                        model, 
                                        None))
    if params["save"]:
        save_data_to_pickle(
            params, 
            recorded_data,
        )
    return 0


def train_with_params(
    params,
    train_loader_0,
    test_loader
):
    """
    train_with_params initializes the main training parts (model, criterion, optimizer and makes private)

    :param params: dict with all parameters
    """ 

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(params["model"]["seed"])
    warnings.filterwarnings("ignore", message=r".*Using a non-full backward hook.*")

    model_class = get_model(params["model"]["model"])
    
    # This train loader is for the full batch and for checking all the individual gradient norms

    train_X_example, _, _, _ = train_loader_0.dataset[0]
    N = len(train_loader_0.dataset)
    print(f"Length = {N}")

    N_CLASSES =  len(np.unique(train_loader_0.dataset.labels))

    if N < params["training"]["batch_size"]:
        raise Exception(f"Batchsize of {params['training']['batch_size']} is larger than the size of the dataset of {N}")
    
    if params["model"]["model"] == "mlp":
        model = ModuleValidator.fix_and_validate(model_class(len(torch.flatten(train_X_example)), N_CLASSES).to(DEVICE))
    else:
        model = ModuleValidator.fix_and_validate(model_class(len(train_X_example), N_CLASSES).to(DEVICE))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["training"]["learning_rate"],
        weight_decay=params["training"]["l2_regularizer"],
    )

    if not os.path.exists(params["Paths"]["gradient_save_path"]):
        os.makedirs(params["Paths"]["gradient_save_path"])


    return train(
        params,
        model,
        DEVICE,
        train_loader_0,
        test_loader,
        optimizer,
        criterion,
    )




if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    print("--------------------------")
    print("--------------------------")
    N_REMOVE = 399
    try_num = 0

    params = {}
    params["model"] = {}
    params["training"] = {}
    params["testing"] = {}
    params["Inform"] = {}
    params["Paths"] = {}
    params["logging"] = {}
    params["save"] = True
    params["model"]["seed"] = 472168
    params["model"]["model"] = "mlp"
    params["model"]["dataset_name"] = "mnist"
    params["training"]["batch_size"] = 400
    params["training"]["learning_rate"] = 1e-04
    params["training"]["l2_regularizer"] = 1e-04
    params["testing"]["test_every"] = 100
    params["training"]["num_epochs"] = 400
    params["model"]["name_base"] = "laplace_"
    params["model"]["name"] = "laplace_"
    params["Inform"]["remove"] = False
    params["Inform"]["idx"] = 0
    params["Inform"]["approximation"] = "AsdlGGN"
    params["Inform"]["representation"] = "diag"
    params["training"]["clip_weight_value"] = 0.05
    params["Paths"]["gradient_save_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_reorder_mnist3/compare"

    params["logging"]["final"] = ["model_laplace", "hist_weights", "per_class_accuracies"]
    params["logging"]["every_epoch"] = ["per_class_accuracies"]
    params["model"]["name"] = params["model"]["name_base"] + str(try_num)

    print("--------------------------")
    print("Load data")
    print("--------------------------")
    dataset_class, data_path = get_dataset(params["model"]["dataset_name"])
    n_length = params["training"]["batch_size"]
    print(f"Limit to first {n_length} of data")
    data_set = dataset_class(data_path, train=True)
    print("here0", flush=True)
    print('Hello world', file=sys.stderr, flush=True)
    data_set.reduce_to_active([*range(n_length)])
    print("here1")
    train_loader_0 = torch.utils.data.DataLoader(
        data_set,
        batch_size=params["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    print("here2")
    # This loader contains the data of the test datset

    data_set_train = dataset_class(data_path, train=False)
    test_loader = torch.utils.data.DataLoader(
        data_set_train,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print("--------------------------")
    print("Compute Compare Model")
    print("--------------------------")
    train_with_params(
        params,
        train_loader_0,
        test_loader
    )

    params["Paths"]["compare_model_path"] = params["Paths"]["gradient_save_path"] + "/" + str(params["model"]["name"]) + ".pkl"
    params["Paths"]["gradient_save_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_reorder_mnist3/gradients"
    params["Inform"]["remove"] = True
    params["Inform"]["idx"] = 0
    params["logging"]["every_epoch"] = []
    params["logging"]["final"] = ["model_laplace", "compare_laplace", "hist_weights"]
    params["training"]["batch_size"] = params["training"]["batch_size"] - 1

    for i in range(N_REMOVE):
        print("--------------------------")
        print(f"Compute Remove Model {i}", flush=True)
        print("--------------------------")
        data_set_rm = deepcopy(data_set)
        data_set_rm.zero_index_from_data(params["Inform"]["idx"])
        print("------")
        print(f'Remove index {params["Inform"]["idx"]}')
        print("------")
        train_loader_0 = torch.utils.data.DataLoader(
            data_set_rm,
            batch_size=params["training"]["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        
        params["model"]["name"] = params["model"]["name_base"] + str(try_num) + "_remove_" + str(params["Inform"]["idx"])
        train_with_params(
            params,
            train_loader_0,
            test_loader
        )
        params["Inform"]["idx"] += 1