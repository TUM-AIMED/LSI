import warnings
import os
import torch
import pickle
import numpy as np
import pandas as pd
import wandb
import time
from opacus.utils.batch_memory_manager import BatchMemoryManager
from laplace import Laplace
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def log_data_final(
    params,
    train_loader,
    model: torch.nn.Module
):
    laplace_approx_mean = None
    results_dict = {}
    if "model_laplace" in params["logging"]["final"]:
        laplace_approx_mean, laplace_approx_precision = _create_laplace_approx(model, train_loader)
        results_dict["laplace_approx_mean"] = laplace_approx_mean
        results_dict["laplace_approx_precision"] = laplace_approx_precision
    if "compare_laplace" in params["logging"]["final"]:
        if laplace_approx_mean is not None:
            mean_KL_divergence_model_vs_compare, mean_KL_divergence_compare_vs_model = _compare_laplace(params, mean=laplace_approx_mean, precision=laplace_approx_precision)
        else:
            mean_KL_divergence_model_vs_compare, mean_KL_divergence_compare_vs_model = _compare_laplace(params, model=model, train_loader=train_loader)
        results_dict["mean_KL_divergence_model_vs_compare"] = mean_KL_divergence_model_vs_compare
        results_dict["mean_KL_divergence_compare_vs_model"] = mean_KL_divergence_compare_vs_model

    return results_dict


def log_data_epoch(
    params,
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

    return results_dict


def save_data_to_pickle(params, results_dict: dict):
    path = params["Paths"]["gradient_save_path"]
    with open(os.path.join(path, str(params["model"]["name"]) + ".pkl"), 'wb') as file:
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

    # Train loader returns also indices (vector idx)
    if params["model"]["private"]:
        with BatchMemoryManager(data_loader=train_loader_0, max_physical_batch_size=5000, optimizer=optimizer) as train_loader_0_new:
            label_wise_gradients = {}
            label_wise_item_count = {}
            for _, (data, target, idx, _) in enumerate(train_loader_0_new):
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

def _create_laplace_approx(model, train_loader):
    print("Replacing ReLus inplace variable")
    model.ReLU_inplace_to_False(model.features)
    model = model.features
    start_time = time.time()
    print("create laplace")
    la = Laplace(model, 'classification',
                 subset_of_weights='all',
                 hessian_structure='diag')
    print(f"fit laplace - laplace creation took {start_time - time.time()}s")
    start_time = time.time()
    la.fit(train_loader)
    print(f"computed laplace - laplace fitting took {start_time - time.time()}s")
    return la.mean, la.posterior_precision


def _compare_laplace(params, model=None, train_loader=None, mean=None, precision=None):
    if mean is None:
        mean, precision = _create_laplace_approx(model, train_loader)
    
    compare_path = params["Paths"]["compare_model_path"]
    with open(compare_path, 'rb') as f:
        data = pickle.load(f) 
        data = data[-1]
        mean_compare = data["laplace_approx_mean"]
        precision_compare = data["laplace_approx_precision"]
    
    # KL(model||model_compare)
    mean_difference = mean_compare - mean
    weight_wise_KL_divergences_model_vs_compare = 0.5 * (precision_compare/precision + mean_difference * precision_compare * mean_difference - 1 + np.log(precision/precision_compare))
    mean_KL_divergence_model_vs_compare = np.mean(weight_wise_KL_divergences_model_vs_compare)
    mean_difference = mean - mean_compare
    weight_wise_KL_divergences_compare_vs_model = 0.5 * (precision/precision_compare + mean_difference * precision * mean_difference - 1 + np.log(precision_compare/precision))
    mean_KL_divergence_compare_vs_model = np.mean(weight_wise_KL_divergences_compare_vs_model)

    return mean_KL_divergence_model_vs_compare, mean_KL_divergence_compare_vs_model

