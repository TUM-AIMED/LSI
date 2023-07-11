import sys
import os

from recording_utils.stats_recorder import write_stats
from recording_utils.data_logger import (
    log_data, save_gradients_to_pickle)
from models.model_helper import get_model
from Datasets.dataset_helper import get_dataset
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.data_loader import DPDataLoader
from opacus import PrivacyEngine
from copy import deepcopy

import torch
import numpy as np
import warnings
import json


def determine_active_set(params,
                         model,
                         DEVICE,
                         optimizer,
                         criterion,
                         train_loader_0,
                         grad_norms,
                         budget,
                         N):

    # Train loader returns also indices (vector idx)
    with BatchMemoryManager(data_loader=train_loader_0, max_physical_batch_size=5000, optimizer=optimizer) as train_loader_0_new:
        for _, (data, target, idx) in enumerate(train_loader_0_new):
            # print(idx[0])
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            batch_grad_norms = torch.zeros(len(target)).to(DEVICE)
            # Clip each parameter's per-sample gradient
            for (ii, p) in enumerate(model.parameters()):

                per_sample_grad = p.grad_sample

                # dimension across which we compute the norms for this gradient part
                # (here is difference e.g. between biases and weight matrices)
                per_sample_grad = torch.reshape(per_sample_grad, (per_sample_grad.shape[0], per_sample_grad.shape[1], -1))
                dims = list(range(1, len(per_sample_grad.shape)))

                # compute the clipped norms. Gradients will be clipped in .backward()
                per_sample_grad_norms = per_sample_grad.norm(dim=dims)
                batch_grad_norms += per_sample_grad_norms ** 2


            # compute the clipped norms. Gradients will be then clipped in .backward()
            # clipped, per sample gradient norms, to track privacy
            grad_norms[idx] += (
                torch.sqrt(batch_grad_norms).clamp(max=params["DP"]["max_per_sample_grad_norm"])
            ) ** 2

    del batch_grad_norms
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    # filter elements that are allowed to continue
    active_indices = [idx for idx in range(N) if grad_norms[idx] < budget]

    if len(active_indices) == 0:
        # print("all budgets exceeded, stopping at epoch: " + str(epoch))
        return

    data_set_active = deepcopy(train_loader_0.dataset)
    data_set_active.reduce_to_active(active_indices)
    train_loader_active = torch.utils.data.DataLoader(
        data_set_active,
        batch_size=params["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    # unique_values, value_counts = np.unique(train_loader_active.dataset.class_assignment_list, return_counts=True)
    # total_count = len(train_loader_active.dataset.class_assignment_list)
    # portions = value_counts / total_count
    # print(unique_values)
    # print(portions)

    return train_loader_active



def train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               privacy_engine,
               stop_epsilon,
               train_loader_active):
    
    T = params["DP"]["T"]
    
    # Run full batch, sum up the gradients. Then we filter and replace the train_loader
    with BatchMemoryManager(data_loader=train_loader_active, max_physical_batch_size=50, optimizer=optimizer) as train_loader_new:
        correct = 0
        total = 0
        
        for _, (data, target, idx) in enumerate(train_loader_new):

            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            loss.backward()
            optimizer.step()
        accuracy = correct/total
        print(f"training accuracy {accuracy}")

    rdp_epsilon = privacy_engine.get_epsilon(1e-5)

    if stop_epsilon and float(stop_epsilon) < rdp_epsilon:
        return 0

    noise_multiplier = optimizer.noise_multiplier

    optimizer.zero_grad()
    torch.cuda.empty_cache()
    """    _save_stats(
        max(grad_norms),
        budget,
        epoch,
        active_indices,
        stats_path,
        max_grad_norm,
        T,
        rdp_epsilon,
        noise_multiplier,
    )"""


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
    gradient_save_path,
    privacy_engine=None,
    stop_epsilon=None,
):
    

    # Compute all the individual norms (actually the squared norms squares are saved here)
    grad_norms = torch.zeros(N).to(DEVICE)
    recorded_gradients = []



    for epoch in range(1, params["training"]["num_epochs"] + 1):
        model.train()
        print(epoch, flush=True)
        train_loader_active = determine_active_set(params,
                         model,
                         DEVICE,
                         optimizer,
                         criterion,
                         train_loader_0,
                         grad_norms,
                         budget,
                         N)
        print(len(train_loader_active.dataset))
        train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               privacy_engine,
               stop_epsilon,
               train_loader_active)
        recorded_gradients.append(log_data(
            model,
            optimizer, 
            train_loader_active, 
            train_loader_0, 
            test_loader,
            criterion, 
            iteration=epoch, 
            max_grad_norm=params["DP"]["max_per_sample_grad_norm"],
            test_every=params["testing"]["test_every"],
        ))

        if epoch % params["testing"]["test_every"] == 0:
            test(model, test_loader)

    # TODO implement running numbering
    save_gradients_to_pickle(
        params, recorded_gradients, path=gradient_save_path
    )
    return 0


def test(model, test_set):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_set:
            inputs, labels, _ = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"testing accuracy: {accuracy}")
    return 


def _save_stats(
    auc_dict: dict,
    max_grad_norm_sum: float,
    budget: float,
    epoch: int,
    active_indices: list,
    stats_path: str,
    clipping_bound: float,
    T: int,
    rdp_epsilon: float,
    noise_multiplier: float,
):
    columns = [
        "epoch",
        "active_elements",
        "max_current_grad_norm_sum",
        "clipping_bound",
        "T",
        "budget",
        "rdp_epsilon",
        "noise_multiplier",
    ]
    stats_dict = dict()
    stats_dict["epoch"] = epoch
    stats_dict["max_current_grad_norm_sum"] = max_grad_norm_sum.item()
    stats_dict["clipping_bound"] = clipping_bound
    stats_dict["T"] = T
    stats_dict["budget"] = budget
    stats_dict["active_elements"] = len(active_indices)
    stats_dict["rdp_epsilon"] = rdp_epsilon
    stats_dict["noise_multiplier"] = noise_multiplier
    columns.append("ave_auc_macro")
    """stats_dict["ave_auc_macro"] = auc_dict.get("ave_auc_macro")
    for i, v in enumerate(auc_dict.get("auc_scores")):
        title = "auc_{}".format(i)
        columns.append(title)
        stats_dict[title] = v"""

    write_stats(stats_dict, path=os.path.join(stats_path, str(params["model"]["name"]) + ".csv"), columns=columns)


def train_with_params(
    params : dict
):

    dataset_class, data_path = get_dataset(params["model"]["dataset_name"])
    model_class = get_model(params["model"]["model"],)

    torch.manual_seed(472368)
    warnings.filterwarnings("ignore", message=r".*Using a non-full backward hook.*")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sigma = params["DP"]["sigma_tilde"]
    T = params["DP"]["T"]
    max_per_sample_grad_norm = params["DP"]["max_per_sample_grad_norm"]

    budget = (
        T * max_per_sample_grad_norm ** 2
    ) + 1e-3  # This will correspond to (eps,delta)-DP filtering with noise std sigma_tilde/sqrt(T)

    # This train loader is for the full batch and for checking all the individual gradient norms
    data_set = dataset_class(data_path, train=True, classes=params["training"]["selected_labels"], portions=params["training"]["balancing_of_labels"])
    train_loader_0 = torch.utils.data.DataLoader(
        data_set,
        batch_size=params["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # This train loader is the one that will be filtered. It will be replaced by a new one at each iteration
    train_loader = torch.utils.data.DataLoader(
        dataset_class(data_path, train=True, classes=params["training"]["selected_labels"], portions=params["training"]["balancing_of_labels"]),
        batch_size=params["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_class(data_path, train=False, classes=params["training"]["selected_labels"]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    train_X_example, train_y_example, idx = train_loader.dataset[0]
    N = len(train_loader.dataset)
    if N < params["training"]["batch_size"]:
        raise Exception(f"Batchsize of {params['training']['batch_size']} is larger than the size of the dataset of {N}")
    

    model = model_class(len(train_X_example), len(np.unique(data_set.labels))).to(DEVICE)

    criterion = torch.nn.NLLLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["training"]["learning_rate"],
        weight_decay=params["training"]["l2_regularizer"],
    )

    secure_rng = False
    privacy_engine = None   
    privacy_engine = PrivacyEngine(secure_mode=secure_rng)

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=max_per_sample_grad_norm,
        poisson_sampling=False,
    )
    if not os.path.exists(params["Paths"]["gradient_save_path"]):
        os.makedirs(params["Paths"]["gradient_save_path"])
    gradient_save_path = os.path.join(params["Paths"]["gradient_save_path"])
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
        gradient_save_path=gradient_save_path,
        privacy_engine=privacy_engine,
        stop_epsilon=None,
    )


if __name__ == "__main__":
    with open('./params/params.json', 'r') as file:
        params = json.load(file)

    train_with_params(
        params
    )
