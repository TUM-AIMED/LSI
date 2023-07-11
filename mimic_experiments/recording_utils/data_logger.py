import warnings
import os
import torch
import pickle
import numpy as np
import pandas as pd
from Datasets.dataset_mnist_unbalanced import MNISTDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log_data(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader_active: torch.utils.data.DataLoader,
    train_loader_complete: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    iteration: int,
    max_grad_norm: float,
    test_every: int,
):
    results_dict = {
        "iteration": iteration
    }
    gradient_df_active = _create_df_of_gradients(
        model,
        optimizer,
        train_loader_active,
        criterion,
        max_grad_norm=max_grad_norm,
    )
    gradient_df_complete = _create_df_of_gradients(
        model,
        optimizer,
        train_loader_complete,
        criterion,
        max_grad_norm=max_grad_norm,
    )

    losses_df_active = _create_df_of_losses(
        model,
        train_loader_active,
    )

    losses_df_complete = _create_df_of_losses(
        model,
        train_loader_complete,
    )

    if iteration % test_every == 0:
        model.eval()
        losses_df_test = _create_df_of_losses(
            model,
            test_loader,    
        )
        model.train()
        results_dict["test_loss"] = losses_df_test
        
    results_dict["gradient_df_active"] = gradient_df_active
    results_dict["gradient_df_complete"] = gradient_df_complete
    results_dict["losses_df_active"] = losses_df_active
    results_dict["losses_df_complete"] = losses_df_complete


    return results_dict


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
    for _, (data, target, id) in enumerate(data_loader):

        # ignore pytorch warnings about hooks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # compute gradients for current sample
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

        # concatenate gradients to compute l2-norm per sample
        total_gradient_norm = 0
        batch_grad_norms = 0
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
    for _, (data, target, id) in enumerate(data_loader):

        # ignore pytorch warnings about hooks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                y_pred = model(torch.atleast_2d(data))
                losses_list.extend(loss(torch.atleast_2d(y_pred), torch.atleast_1d(target)))
                id_list.extend(id.tolist())
                sample_ids_list.extend(sample_ids[id])
    losses_list = [tensor.item() for tensor in losses_list]
    df = pd.DataFrame({"ids" : torch.tensor(id_list),
                    "classes": torch.tensor(sample_ids_list), 
                    "losses": torch.tensor(losses_list).cpu(),
                    "idx": torch.tensor(dataset_idx)})
    df.avg_loss = torch.mean(torch.tensor(losses_list))

    return df



def save_gradients_to_pickle(params, results_dict: dict, path: str):
    with open(os.path.join(path, str(params["model"]["name"]) + ".pkl"), 'wb') as file:
        pickle.dump(results_dict, file)
    print("Save results to", path)
