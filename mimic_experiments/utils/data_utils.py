from opacus.utils.batch_memory_manager import BatchMemoryManager
from copy import deepcopy
import torch

def determine_active_set(params,
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
    with BatchMemoryManager(data_loader=train_loader_0, max_physical_batch_size=5000, optimizer=optimizer) as train_loader_0_new:
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