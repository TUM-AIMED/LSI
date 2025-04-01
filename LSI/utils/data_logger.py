import warnings
import os
import torch
import pickle
import numpy as np
import pandas as pd
import wandb
import time
from Datasets.dataset_helper import get_dataset
from tqdm import tqdm
from opacus.utils.batch_memory_manager import BatchMemoryManager
from laplace import Laplace
from utils.helper import laplace_backed_helper, representation_helper
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from opacus.grad_sample.grad_sample_module import GradSampleModule


def log_data_final(
    params,
    DEVICE,
    train_loader,
    model: torch.nn.Module,
    idp_accountant
):
    laplace_approx_mean = None
    results_dict = {}
    if params["Inform"]["remove"]:
        results_dict["removed_idx"] = params["Inform"]["idx"]
    if "model_laplace" in params["logging"]["final"]:
        laplace_approx_mean, laplace_approx_precision = _create_laplace_approx(params, model, train_loader)
        results_dict["laplace_approx_mean"] = laplace_approx_mean
        results_dict["laplace_approx_precision"] = laplace_approx_precision
    if "compare_laplace" in params["logging"]["final"]:
        if laplace_approx_mean is not None:
            mean_KL_divergence_model_vs_compare, mean_KL_divergence_compare_vs_model, kl_test = _compare_laplace(params, mean1=laplace_approx_mean, precision1=laplace_approx_precision)
        else:
            mean_KL_divergence_model_vs_compare, mean_KL_divergence_compare_vs_model, kl_test = _compare_laplace(params, model=model, train_loader=train_loader)
        results_dict["mean_KL_divergence_model_vs_compare"] = mean_KL_divergence_model_vs_compare
        results_dict["mean_KL_divergence_compare_vs_model"] = mean_KL_divergence_compare_vs_model
        results_dict["kl_test"] = kl_test
    if "get_accuracies" in params["logging"]["final"]:
        idx_list, accuracies = get_accuracies(model, train_loader)
        results_dict["accuracies_idx_list"] = idx_list
        results_dict["per_idx_accuracies"] = accuracies
    if "get_idx_accuracy" in params["logging"]["final"]:
        idx, accuracy, label = get_idx_accuracy(params, model)
        idx_list, accuracies = get_accuracies(model, train_loader)
        results_dict["idx"] = idx
        results_dict["accuracy"] = accuracy
    if "hist_weights" in params["logging"]["final"]:
        hist, bins = log_weights(model)
        results_dict["weight_hist"] = hist
        results_dict["weight_bins"] = bins
    if "idp_accountant" in params["logging"]["final"]:
        print("Gettint RDP values")
        up_eps = idp_accountant.return_epsilon(1e-5)        
        results_dict["idp_accountant_indiv"] = idp_accountant.indiv_Accountants
        results_dict["idp_accountant"] = up_eps
    if "per_class_accuracies" in params["logging"]["final"]:
        print("Getting per class accuracies")
        results_dict["per_class_accuracies"] = log_per_class_accuracies_test(DEVICE, model, train_loader)
    if "labels" in params["logging"]["final"]:
        print("Getting labels")
        results_dict["labels"] = log_labels(DEVICE, model, train_loader)
    return results_dict


def log_data_epoch(
    params,
    DEVICE,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader_active: torch.utils.data.DataLoader,
    train_loader_complete: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    epoch: int,
    max_grad_norm: float,
    test_every: int,
):
    results_dict = {
        "epoch": epoch
    }
    if "gradient_df_active" in params["logging"]["every_epoch"]:
        gradient_df_active = _create_df_of_gradients(
            model,
            optimizer,
            train_loader_active,
            criterion,
            max_grad_norm=max_grad_norm,
        )
        results_dict["gradient_df_active"] = gradient_df_active
    if "gradient_df_complete" in params["logging"]["every_epoch"]:
        gradient_df_complete = _create_df_of_gradients(
            model,
            optimizer,
            train_loader_complete,
            criterion,
            max_grad_norm=max_grad_norm,
        )
        results_dict["gradient_df_complete"] = gradient_df_complete
    if "losses_df_active" in params["logging"]["every_epoch"]:
        losses_df_active = _create_df_of_losses(
            model,
            train_loader_active,
        )
        results_dict["losses_df_active"] = losses_df_active
    if "losses_df_complete" in params["logging"]["every_epoch"]:
        losses_df_complete = _create_df_of_losses(
            model,
            train_loader_complete,
        )
        results_dict["losses_df_complete"] = losses_df_complete
    if epoch % test_every == 0:
        if "losses_df_test" in params["logging"]["every_epoch"]:
            model.eval()
            losses_df_test = _create_df_of_losses(
                model,
                test_loader,    
            )
            model.train()
            results_dict["test_loss"] = losses_df_test
    if "per_class_accuracies" in params["logging"]["every_epoch"]:
        # print("Getting per class accuracies")
        results_dict["per_class_accuracies"] = log_per_class_accuracies_test(DEVICE, model, train_loader_complete)

    return results_dict

def save_data_to_pickle(params, results_dict: dict):
    path = params["Paths"]["gradient_save_path"]
    path = os.path.join(path, str(params["model"]["name"]) + ".pkl")
    with open(path, 'wb') as file:
        pickle.dump(results_dict, file)
    print("Save results to", path)


def _create_df_of_gradients(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    max_grad_norm: float
):
    total_gradient_norm_list = []
    id_list = []
    sample_ids_list = []
    sample_ids= data_loader.dataset.labels
    dataset_idx= data_loader.dataset.active_indices


    # compute gradient for single data points
    # a naive, but simple solution
    with BatchMemoryManager(data_loader=data_loader, max_physical_batch_size=50, optimizer=optimizer) as data_loader:
        for _, (data, target, id, _) in enumerate(data_loader):

            # ignore pytorch warnings about hooks
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # compute gradients for current sample
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

            # concatenate gradients to compute l2-norm per sample
            total_gradient_norm = 0
            batch_grad_norms = 0
            for (ii, p) in enumerate(model.parameters()):
                if p.requires_grad:
                    per_sample_grad = p.grad_sample

                    # dimension across which we compute the norms for this gradient part
                    # (here is difference e.g. between biases and weight matrices)
                    per_sample_grad = torch.reshape(per_sample_grad, (per_sample_grad.shape[0], per_sample_grad.shape[1], -1))
                    dims = list(range(1, len(per_sample_grad.shape)))

                    # compute the clipped norms. Gradients will be clipped in .backward()
                    per_sample_grad_norms = per_sample_grad.norm(dim=dims)

                    batch_grad_norms += per_sample_grad_norms ** 2

                    # compute the clipped norms. Gradients will be then clipped in .backward()
            total_gradient_norm += (
                torch.sqrt(batch_grad_norms).clamp(max=max_grad_norm)
            ) ** 2
            # Save in list for batchsizes smaller than whole dataset
            total_gradient_norm_list.append(total_gradient_norm)
            id_list.append(id)
            sample_ids_list.append(sample_ids[id])
        # concatenate all batches
    total_gradient_norm_list = torch.concatenate(total_gradient_norm_list)
    id_list = np.concatenate(id_list)
    sample_ids_list = np.concatenate(sample_ids_list)

    
    df = pd.DataFrame({"ids" : torch.tensor(id_list),
                       "classes": sample_ids_list, 
                       "total_gradient_norm": total_gradient_norm_list.cpu(),
                       "idx": torch.tensor(dataset_idx)})

    optimizer.zero_grad()
    return df

def _create_df_of_losses(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
):
    sample_ids= data_loader.dataset.labels
    dataset_idx= data_loader.dataset.active_indices
    loss = torch.nn.NLLLoss(reduction='none')

    # compute gradient for single data points
    # a naive, but simple solution
    losses_list = []
    id_list = []
    sample_ids_list = []
    for _, (data, target, id, _) in enumerate(data_loader):

        # ignore pytorch warnings about hooks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data, target = data.to(DEVICE), target.to(DEVICE)
            with torch.no_grad():
                y_pred = model(torch.atleast_2d(data))
                losses_list.extend(loss(torch.atleast_2d(y_pred), torch.atleast_1d(target)))
                id_list.extend(id.tolist())
    sample_ids_list = sample_ids[id_list]
    losses_list = [tensor.item() for tensor in losses_list]
    df = pd.DataFrame({"ids" : torch.tensor(id_list),
                    "classes": torch.tensor(sample_ids_list), 
                    "losses": torch.tensor(losses_list).cpu(),
                    "idx": torch.tensor(dataset_idx)})
    df.avg_loss = torch.mean(torch.tensor(losses_list))

    return df


def log_realized_gradients(params,
                         model,
                         DEVICE,
                         optimizer,
                         criterion,
                         train_loader_0,
                         grad_norms,
                         budget,
                         N):
    """
    train runs the epoch loop and calls train_step and test

    :param params: params dict
    :param model: pytorch model
    :param DEVICE: available device
    :param optimizer: optimizer fct (Adam, AdamW etc.)
    :param criterion: loss function
    :param train_loader_0: dataloader of the whole, unfiltered dataset
    :param grad_norms: tensor to save all the accumulated gradient norms - used for logging
    :param budget: privacy budget of data elements
    :param N: number of training samples

    """ 
    start_time = time.time()
    # Train loader returns also indices (vector idx)
    if params["model"]["private"]:
        with BatchMemoryManager(data_loader=train_loader_0, max_physical_batch_size=50, optimizer=optimizer) as train_loader_0_new:
            label_wise_gradients = {}
            label_wise_item_count = {}
            total_gradient_norm_list = []
            id_list = []
            sample_ids_list = []
            sample_ids= train_loader_0.dataset.labels
            for _, (data, target, id, _) in enumerate(train_loader_0_new):
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
                clipped_grads = (
                    torch.sqrt(batch_grad_norms).clamp(max=params["DP"]["max_per_sample_grad_norm"])
                        ) ** 2
                for label in target.unique():
                    mask = target == label.item()
                    num_with_label = torch.sum(mask.long()).item()
                    if str(label.item()) not in label_wise_gradients:
                        label_wise_gradients[str(label.item())] = 0
                    if str(label.item()) not in label_wise_item_count:
                        label_wise_item_count[str(label.item())] = 0
                    label_wise_gradients[str(label.item())] += sum(clipped_grads[target == label.item()])
                    label_wise_item_count[str(label.item())] += num_with_label
                total_gradient_norm_list.append(clipped_grads)
                id_list.append(id)
                sample_ids_list.append(sample_ids[id])
        for key, value in label_wise_item_count.items():
            label_wise_gradients[key] = label_wise_gradients[key]/value

        for key, value in label_wise_gradients.items():
            wandb.log({"gradients_" + key: value})

        del batch_grad_norms
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        

    else:
        train_loader_0_new = train_loader_0
        label_wise_gradients = {}
        label_wise_item_count = {}
        for _, (data, target, idx, _) in enumerate(train_loader_0_new):
            # print(idx[0])
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            #TODO This does not compute the real per sample grads
            batch_grad_norms = torch.zeros(len(target)).to(DEVICE)
            # Clip each parameter's per-sample gradient
            for (ii, p) in enumerate(model.parameters()):

                per_sample_grad = p.grad
                # batch_grad_norms += per_sample_grad_norms ** 2

            # compute the clipped norms. Gradients will be then clipped in .backward()
            # clipped, per sample gradient norms, to track privacy
            clipped_grads = (
                torch.sqrt(batch_grad_norms)) ** 2
            for label in target.unique():
                mask = target == label.item()
                num_with_label = torch.sum(mask.long()).item()
                if str(label.item()) not in label_wise_gradients:
                    label_wise_gradients[str(label.item())] = 0
                if str(label.item()) not in label_wise_item_count:
                    label_wise_item_count[str(label.item())] = 0
                label_wise_gradients[str(label.item())] += sum(clipped_grads[target == label.item()])
                label_wise_item_count[str(label.item())] += num_with_label
        for key, value in label_wise_item_count.items():
            label_wise_gradients[key] = label_wise_gradients[key]/value

        for key, value in label_wise_gradients.items():
            wandb.log({"gradients_" + key: value})

        del batch_grad_norms
        optimizer.zero_grad()
        torch.cuda.empty_cache()
    

    return 


def get_accuracies(model, train_loader):
    model.eval()
    output_list = []
    idx_list = []
    for _, (data, target, idx, _) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        output_list.append(output)
        idx_list.append(idx)
    return np.array(idx_list), np.array(output_list)


def get_idx_accuracy(params, model):
    dataset_class, data_path = get_dataset(params["model"]["dataset_name"])
    if params["model"]["split_data"]:
        data_set = dataset_class(data_path, train=True, classes=params["training"]["selected_labels"], portions=params["training"]["balancing_of_labels"], shuffle=False)
    else:
        data_set = dataset_class(data_path, train=True)

    x, y, idx, _ = data_set.__getitem__(idx)
    output = model(x)
    return idx, output, y


def compute_log_det(diag_array):
    det = np.sum(np.log(diag_array))
    return det


def computeKL(mean1, mean2, precision1, precision2):
    inv_precision1 = 1/precision1
    inv_precision2 = 1/precision2
    mean_difference = mean2 - mean1
    # test1 = np.sum(np.multiply(precision2, inv_precision1)) 
    # test2 = np.sum(np.multiply(mean_difference, np.multiply(precision2, mean_difference))) 
    # test3 = len(mean1) 
    # test4 = compute_log_det(inv_precision2) - compute_log_det(inv_precision1)
    # test5 = compute_log_det(inv_precision2)
    # test6 = compute_log_det(inv_precision1)
    # print("------")
    # print(test1)
    # print(test2)
    # print(test3)
    # print(test4)
    # print(test5)
    # print(test6)
    # print("------")
    kl = 0.5*(np.sum(np.multiply(precision2, inv_precision1)) 
              + np.sum(np.multiply(mean_difference, np.multiply(precision2, mean_difference))) 
              - len(mean1) 
              + compute_log_det(inv_precision2) - compute_log_det(inv_precision1))
    return kl   

def _create_laplace_approx(params, model, train_loader):
    print("Replacing ReLus inplace variable")
    if isinstance(model, GradSampleModule):
        model._module.ReLU_inplace_to_False(model.features)
        model.disable_hooks()
    else:
        model.ReLU_inplace_to_False(model.features)
    # model.ReLU_inplace_to_False(model.features)
    # model = model.features
    start_time = time.time()
    print("create laplace")
    backend_class = laplace_backed_helper(params["Inform"]["approximation"])
    representation = representation_helper(params["Inform"]["representation"])
    if params["Inform"]["representation"] != "diag":
        raise Exception("Determinante Computation only works for diagonal representation at the moment")
    param_values = []
    # for param in model.parameters():
    #     param_values.extend(param.data.cpu().numpy().flatten())
    la = Laplace(model, 'classification',
                 subset_of_weights='all',
                 hessian_structure=representation,
                 backend=backend_class)
    print(f"fit laplace - laplace creation took {time.time() - start_time}s")
    start_time = time.time()
    la.fit(train_loader)
    print(f"computed laplace - laplace fitting took {time.time() - start_time}s")
    mean = la.mean.cpu().numpy()
    post_prec = la.posterior_precision.cpu().numpy()
    print("got mean and prec on cpu")
    return mean, post_prec


def _compare_laplace(params, model=None, train_loader=None, mean1=None, precision1=None):
    if mean1 is None:
        mean1, precision1 = _create_laplace_approx(params, model, train_loader)
    
    compare_path = params["Paths"]["compare_model_path"]
    with open(compare_path, 'rb') as f:
        data = pickle.load(f) 
        data = data[-1]
        mean2 = data["laplace_approx_mean"]
        precision2 = data["laplace_approx_precision"]
    # KL(model||model_compare)
    test_precision = np.ones(precision1.shape)
    print("---------")
    kl1 = computeKL(mean1, mean2, precision1, precision2)
    print(f"kl1: {kl1}")
    # KL(model_compare||model)
    kl2 = computeKL(mean2, mean1, precision2, precision1)
    print(f"kl2: {kl2}")
    print("---------")
    kl_test = computeKL(mean1, mean2, test_precision, test_precision)
    print(f"kl_test: {kl_test}")
    print("---------")
    return kl1, kl2, kl_test

def log_weights(model):
    print("getting hists of the weights")
    num_bins = 20  # You can adjust the number of bins as needed
    weight_range = (-1.0, 1.0)  # Specify the range for weight values
    all_weights = []

    for param in model.parameters():
        if param.requires_grad:
            weights = param.data.cpu().numpy().flatten()  # Convert to NumPy array
            all_weights.extend(weights)

    # Create a histogram of weights
    hist, bins = np.histogram(all_weights, bins=100, range=(-0.1, 0.1))
    return hist, bins

def log_per_class_accuracies_test(DEVICE, model, data_set):
    model.eval()  # Set the model to evaluation mode
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for data in data_set:
            inputs, target, _, _ = data
            inputs, target = inputs.to(DEVICE), target.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for c in np.unique(target.cpu()):
                class_correct[c] = class_correct.get(c, 0) + (predicted[target == c] == c).sum().item()
                class_total[c] = class_total.get(c, 0) + (target == c).sum().item()

    class_accuracies = {}
    for c, correct in class_correct.items():
        class_accuracies[c] = correct / class_total[c]

    # print("Per-class accuracies:")
    # for c, accuracy in class_accuracies.items():
    #     print(f"Class {c}: {accuracy:.4f}")

    return class_accuracies


def log_labels(DEVICE, model, data_set):
    model.eval()
    labels = []
    with torch.no_grad():
        for data in data_set:
            _, target, _, _ = data
            labels.extend(target)
    return labels



