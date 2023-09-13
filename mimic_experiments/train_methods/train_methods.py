from utils.data_logger import (
    log_data_epoch, log_data_final, save_data_to_pickle, log_realized_gradients)
from utils.data_utils import determine_active_set
from opacus.utils.batch_memory_manager import BatchMemoryManager

from opacus import PrivacyEngine
from copy import deepcopy

import torch
import wandb
import time


def test(DEVICE, model, test_set):
    """
    test gets the output of the model an computes the accuracy

    :param DEVICE: available device
    :param model: pytorch model
    :param test_set: dataloader of the test data
    """ 
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
    wandb.log({"test_accuracy": accuracy})
    return 

def normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               privacy_engine,
               stop_epsilon,
               train_loader_active):
    """
    train runs the epoch loop and calls train_step and test

    :param params: params dict
    :param model: pytorch model
    :param optimizer: optimizer fct (Adam, AdamW etc.)
    :param DEVICE: available device
    :param criterion: loss function
    :param privacy_engine: opacus privacy engine
    :param stop_epsilon: idk
    :param train_loader_active: dataloader of the filtered dataset
    """ 

    # Run full batch, sum up the gradients. Then we filter and replace the train_loader
    total = 0
    correct = 0
    if params["model"]["private"]:
        with BatchMemoryManager(data_loader=train_loader_active, max_physical_batch_size=50, optimizer=optimizer) as train_loader_new:
            loss_list = []
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
            accuracy = correct/total
            print(f"training accuracy {accuracy}")
            wandb.log({"train_accuracy": accuracy})
            wandb.log({"loss": sum(loss_list)})
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
        accuracy = correct/total
        print(f"training accuracy {accuracy}")
        wandb.log({"train_accuracy": accuracy})
        wandb.log({"loss": sum(loss_list)})



    # rdp_epsilon = privacy_engine.get_epsilon(1e-5)

    # if stop_epsilon and float(stop_epsilon) < rdp_epsilon:
    #     return 0

    # noise_multiplier = optimizer.noise_multiplier

    optimizer.zero_grad()
    torch.cuda.empty_cache()



def train(
    params,
    model,
    DEVICE,
    train_loader_0,
    test_loader,
    optimizer,
    budget,
    criterion,
    N,
    stats_path,
    privacy_engine=None,
    stop_epsilon=None,
):
    """
    train runs the epoch loop and calls train_step and test

    :param params: params dict
    :param model: pytorch model
    :param DEVICE: available device
    :param train_loader_0: dataloader of the whole, unfiltered dataset
    :param test_loader: dataloader with the test data
    :param optimizer: optimizer fct (Adam, AdamW etc.)
    :param budget: privacy budget of data elements
    :param criterion: loss function
    :param N: number of training samples
    :param stats_path: path to location of stats save_path
    :param gradient_save_path: path to location of gradient save files
    :param privacy_engine: opacus privacy engine
    :param stop_epsilon: idk
    """ 

    # Compute all the individual norms (actually the squared norms squares are saved here)
    grad_norms = torch.zeros(N).to(DEVICE)
    recorded_data = []

    print("here")
    for epoch in range(1, params["training"]["num_epochs"] + 1):
        start_epoch = time.time()
        model.train()
        print(epoch, flush=True)
        if params["model"]["private_filter"]:
            train_loader_active = determine_active_set(params,
                            model,
                            DEVICE,
                            optimizer,
                            criterion,
                            train_loader_0,
                            grad_norms,
                            budget,
                            N)
        else:
            data_set_active = deepcopy(train_loader_0.dataset)
            train_loader_active = torch.utils.data.DataLoader(
                data_set_active,
                batch_size=params["training"]["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
        if "gradients" in params["logging"]["every_epoch"]:    
            log_realized_gradients(params,
                            model,
                            DEVICE,
                            optimizer,
                            criterion,
                            train_loader_0,
                            grad_norms,
                            budget,
                            N)
        print(len(train_loader_active.dataset), flush=True)
        normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               privacy_engine,
               stop_epsilon,
               train_loader_active)
        if params["save"]:
            recorded_data.append(log_data_epoch(
                params,
                model,
                optimizer, 
                train_loader_active, 
                train_loader_0, 
                test_loader,
                criterion, 
                epoch=epoch, 
                max_grad_norm=params["DP"]["max_per_sample_grad_norm"],
                test_every=params["testing"]["test_every"],
            ))

        if epoch % params["testing"]["test_every"] == 0:
            test(DEVICE, model, test_loader)
        
        end_epoch = time.time()
        print(f"Epoch took {end_epoch-start_epoch}")


    recorded_data.append(log_data_final(params, train_loader_0, model))
    if params["save"]:
        save_data_to_pickle(
            params, 
            recorded_data,
        )
    print(f'lr: {params["training"]["learning_rate"]}')
    print(f'l2: {params["training"]["l2_regularizer"]}')
    print(f'epochs: {params["training"]["num_epochs"]}')
    print(f'batchsize: {params["training"]["batch_size"]}')
    return 0