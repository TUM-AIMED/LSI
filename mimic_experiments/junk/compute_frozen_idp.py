import sys
import os

from models.model_helper import get_model
from Datasets.dataset_helper import get_dataset
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.data_loader import DPDataLoader
from opacus.validators.module_validator import ModuleValidator
from utils.kl_div import load_sorted_idx
from opacus import PrivacyEngine
import torch
import numpy as np
import warnings
import json
import argparse
import time
import pickle
from tqdm import tqdm
from collections import defaultdict

def log_realized_gradients(params,
                         model,
                         DEVICE,
                         optimizer,
                         criterion,
                         train_loader_0,
                         grad_norms,
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
    with BatchMemoryManager(data_loader=train_loader_0, max_physical_batch_size=500, optimizer=optimizer) as train_loader_0_new:
        for _, (data, target, idx, _) in enumerate(train_loader_0_new):
            # print(idx[0])
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward(retain_graph=True)

            batch_grad_norms = torch.zeros(len(target)).to(DEVICE)
            # Clip each parameter's per-sample gradient
            for (ii, p) in enumerate(model.parameters()):
                if not p.requires_grad:
                    continue
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
            for index, batch_grad_norm in zip(idx, batch_grad_norms):
                grad_norms[index.item()].append(((
                    torch.sqrt(batch_grad_norm).clamp(max=params["DP"]["max_per_sample_grad_norm"])
                ) ** 2).item())

    del batch_grad_norms
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    return 



def normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               train_loader_active):
   
    # Run full batch, sum up the gradients. Then we filter and replace the train_loader
    with BatchMemoryManager(data_loader=train_loader_active, max_physical_batch_size=500, optimizer=optimizer) as train_loader_new:
        correct = 0
        total = 0
        loss_list = []
        for _, (data, target, idx, _) in enumerate(train_loader_new):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
    accuracy = correct/total
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    print(f"accuracy was {accuracy}")
    return np.mean(np.array(loss_list)), accuracy


def normal_val_step(params,
            model, 
            optimizer, 
            DEVICE, 
            criterion, 
            val_loader_active):
    start_time = time.time()
    total = 0
    correct = 0
    loss_list = []
    for _, (data, target, idx, _) in enumerate(val_loader_active):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        loss_list.append(loss.item())
    accuracy = correct/total
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    return np.mean(np.array(loss_list)), accuracy



def train(
    params,
    model,
    DEVICE,
    train_loader_0,
    test_loader,
    optimizer,
    criterion,
    N,
    privacy_engine=None
):

    # Compute all the individual norms (actually the squared norms squares are saved here)
    grad_norms = defaultdict(list)
    recorded_gradients = []
    for epoch in range(1, params["training"]["num_epochs_init"]):
        model.train()
        print(epoch, flush=True)
        log_realized_gradients(params,
                         model,
                         DEVICE,
                         optimizer,
                         criterion,
                         train_loader_0,
                         grad_norms,
                         N)
        
        # print(grad_norms)
        train_loss, train_accuracy = normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               train_loader_0)
        model.eval()
        if epoch % params["testing"]["test_every"] == 0 or epoch == params["training"]["num_epochs_init"]-1:
            val_loss, val_accuracy = normal_val_step(params,
                model, 
                optimizer, 
                DEVICE, 
                criterion, 
                test_loader)
    print(f"train accuracy {train_accuracy:.4f}")
    print(f"val accuracy {val_accuracy:.4f}")
    return model, epoch, grad_norms, train_accuracy, val_accuracy


def train_with_params(
    params : dict,
    train_loader,
    test_loader
):
    """
    train_with_params initializes the main training parts (model, criterion, optimizer and makes private)

    :param params: dict with all parameters
    """ 
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(params["model"]["seed"])
    torch.cuda.manual_seed_all(params["model"]["seed"])
    np.random.seed(params["model"]["seed"])
    warnings.filterwarnings("ignore", message=r".*Using a non-full backward hook.*")



    train_X_example, _, _, _ = train_loader.dataset[0]
    N = len(train_loader.dataset)
    if N < params["training"]["batch_size"]:
        raise Exception(f"Batchsize of {params['training']['batch_size']} is larger than the size of the dataset of {N}")
    N_CLASSES =  len(np.unique(train_loader.dataset.labels))
    model_class = get_model(params["model"]["model"])
    if params["model"]["model"] == "mlp" or params["model"]["model"] == "small_mlp" or params["model"]["model"] == "logreg":
        model = ModuleValidator.fix_and_validate(model_class(len(torch.flatten(train_X_example)), N_CLASSES).to(DEVICE))
    else:
        model = ModuleValidator.fix_and_validate(model_class(len(train_X_example), N_CLASSES))
        model = model.to(DEVICE)

    if params["freeze"] == True:
        model.freeze_all_but_last() 
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=params["training"]["learning_rate"],
    #     weight_decay=params["training"]["l2_regularizer"],
    # )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params["training"]["learning_rate"],
        weight_decay=params["training"]["l2_regularizer"], momentum=0.9, nesterov=True)

    privacy_engine = PrivacyEngine(secure_mode=False)

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=params["DP"]["sigma_tilde"],
        max_grad_norm=params["DP"]["max_per_sample_grad_norm"],
        poisson_sampling=False,
    )



    return train(
        params,
        model,
        DEVICE,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        N,
        privacy_engine=privacy_engine
        )


if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_seeds", type=int, default=2, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default=None, help="Value for lerining_rate (optional)")
    parser.add_argument("--name_ext", type=str, default="")
    parser.add_argument("--freeze", type=bool, default=False, help="Value for lerining_rate (optional)")
    parser.add_argument("--epochs", type=int, default=500, help="Value for lerining_rate (optional)")
    parser.add_argument("--model", type=str, default="logreg4", help="Value for lerining_rate (optional)")
    parser.add_argument("--dataset", type=str, default="cifar100compressed", help="Value for lerining_rate (optional)")
    parser.add_argument("--corrupt", type=str, default="", help="Value for lerining_rate (optional)")
    parser.add_argument("--clip", type=float, default=1000, help="Value for lerining_rate (optional)")
    parser.add_argument("--subset", type=float, default=0.1, help="Value for lerining_rate (optional)")
    parser.add_argument("--dp_noise", type=float, default=1e-12, help="Value for lerining_rate (optional)")



    args = parser.parse_args()

    print("--------------------------", flush=True)
    print("--------------------------", flush=True)
    N_SEEDS = args.n_seeds
    try_num = 0
    
    params = {}
    params["save"] = True
    params["model"] = {}
    params["model"]["seed"] = 472168
    params["model"]["model"] = args.model
    params["model"]["dataset_name"] = args.dataset
    params["training"] = {}
    params["training"]["batch_size"] = 256
    params["training"]["learning_rate"] = 3e-01# 0.002 mlp #3e-03 # -3 for mnist
    params["training"]["l2_regularizer"] = 1e-08
    params["training"]["num_epochs_init"] = args.epochs
    params["testing"] = {}
    params["testing"]["test_every"] = params["training"]["num_epochs_init"] + 1
    params["Paths"] = {}

    params["freeze"] = True # args.freeze
    params["DP"] = {}
    params["DP"]["sigma_tilde"] = args.dp_noise
    params["DP"]["max_per_sample_grad_norm"] = args.clip
    params["idx_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_idp/results_cifar100compressed_logreg4_1_100.0_1e-09__1000_50000/results.pkl"

    print("--------------------------")
    print("Load data")
    print("--------------------------", flush=True)

    data_set_class, data_path = get_dataset(params["model"]["dataset_name"])
 
    data_set = data_set_class(data_path, train=True)
    data_set_test = data_set_class(data_path, train=False) 
    if args.subset > 1:
        args.subset = int(args.subset)
        keep_indices = [*range(args.subset)]
    else:
        keep_indices = load_sorted_idx(params["idx_path"])
        keep_indices = keep_indices[-int(args.subset * 50000):]
        # params["training"]["learning_rate"] = params["training"]["learning_rate"]*args.subset
    data_set.reduce_to_active(keep_indices)

    train_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=len(data_set), # params["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = torch.utils.data.DataLoader(
        data_set_test,
        batch_size=len(data_set), # params["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    grad_norms_all = {}
    for i in tqdm(range(N_SEEDS)):
        params["model"]["seed"] = i
        _, _, grad_norms, train_accuracy, val_accuracy = train_with_params(
            params,
            train_loader,
            test_loader
        )
        grad_norms_all[i] = grad_norms
    
    if args.name != None:
        params["Paths"]["final_save_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_idp/results_" + str(args.name)
    else:
        name_str = f"{params['model']['dataset_name']}_{params['model']['model']}_{N_SEEDS}_{args.clip}_{args.dp_noise}_{args.corrupt}_{args.epochs}_{args.subset}_{train_accuracy:.4f}_{val_accuracy:.4f}{args.name_ext}"
        params["Paths"]["final_save_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_script/results_" + name_str

    if not os.path.exists(params["Paths"]["final_save_path"]):
        os.makedirs(params["Paths"]["final_save_path"])
    filename = params["Paths"]["final_save_path"] + "/results.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(grad_norms_all, file)
    print(f"Saving file under {filename}")
