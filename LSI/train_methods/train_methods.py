from utils.data_logger import (
    log_data_epoch, log_data_final, save_data_to_pickle, log_realized_gradients)
from utils.data_utils import determine_active_set
from opacus.utils.batch_memory_manager import BatchMemoryManager
from utils.idp_tracker import update_norms
from utils.idp_tracker_utils import process_grad_batch
from opacus.grad_sample.grad_sample_module import GradSampleModule
from copy import deepcopy

import sys
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
               epoch,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               stop_epsilon,
               train_loader_active,
               idp_accountant,
               step_count):
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
    max_physical_batch_size = 100
    if params["model"]["private"]:
        dataset = deepcopy(train_loader_active.dataset)
        train_loader_gradient_computation = torch.utils.data.DataLoader(dataset, batch_size=max_physical_batch_size, shuffle=False, num_workers=0, pin_memory=False)
        with BatchMemoryManager(data_loader=train_loader_active, max_physical_batch_size=max_physical_batch_size, optimizer=optimizer) as train_loader_new:
            loss_list = []
            loop_iter_counter = 0
            for _, (data, target, idx, _) in enumerate(train_loader_new):
                loop_iter_counter += 1
                data, target = data.to(DEVICE), target.to(DEVICE)
                last_data_of_batch = not optimizer._step_skip_queue[0]
                if(step_count % params["DP"]["update_norms_every"] == 0 and last_data_of_batch):
                    start_time = time.time()
                    print(f"updating norms at epoch {epoch} with step (batch) {step_count}", flush=True)
                    update_norms(DEVICE,
                                epoch,
                                model,
                                optimizer,
                                criterion,
                                train_loader_gradient_computation,
                                idp_accountant)
                    print(f"Updating Norms took {time.time() - start_time}", flush=True)
                    model.train()
                optimizer.zero_grad()

                output = model(data)
                loss = criterion(output, target)
                loss_list.append(loss)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                loss.backward()
                optimizer.step()
                if last_data_of_batch:
                    print(f"Index of Batch {idx[0]}")
                    idp_accountant.update_loss()
                    step_count += 1
                    loop_iter_counter = 0
                    if params["training"]["clip_weight_value"] != None:
                        if isinstance(model, GradSampleModule):
                            model._module.clip_weights(params["training"]["clip_weight_value"])
                        else:
                            model.clip_weights(params["training"]["clip_weight_value"])
    elif not params["model"]["private"] and params["Inform"]["remove"]:
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
    wandb.log({"train_accuracy": accuracy})
    wandb.log({"loss": sum(loss_list)})




def train(
    params,
    model,
    DEVICE,
    train_loader_0,
    test_loader,
    optimizer,
    criterion,
    stop_epsilon=None,
    idp_accountant = None
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
               epoch,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               stop_epsilon,
               train_loader_0,
               idp_accountant,
               step_count)
        
        if params["save"]:
            recorded_data.append(log_data_epoch(
                params,
                DEVICE,
                model,
                optimizer, 
                train_loader_0, 
                train_loader_0, 
                test_loader,
                criterion, 
                epoch=epoch, 
                max_grad_norm=params["DP"]["max_per_sample_grad_norm"],
                test_every=params["testing"]["test_every"]
            ))

        if epoch % params["testing"]["test_every"] == 0:
            test(DEVICE, model, test_loader)
        
        end_epoch = time.time()
        print(f"Epoch took {end_epoch-start_epoch}")


    recorded_data.append(log_data_final(params, 
                                        DEVICE,
                                        train_loader_0, 
                                        model, 
                                        idp_accountant))
    if params["save"]:
        save_data_to_pickle(
            params, 
            recorded_data,
        )
    return 0