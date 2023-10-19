import sys
import os
import pickle

from models.model_helper import get_model
from Datasets.dataset_helper import get_dataset, CustomDataLoader
from opacus.validators.module_validator import ModuleValidator
from utils.idp_tracker import PrivacyLossTracker
from utils.data_utils import pretty_print_dict
from utils.data_logger import (
    log_data_epoch, log_data_final, save_data_to_pickle, computeKL)


from opacus import PrivacyEngine
from copy import deepcopy
from tqdm import tqdm
import torch
import numpy as np
import warnings
import json
import argparse
import wandb
import time

from backpack.extensions import BatchGrad
from backpack import backpack, extend
from dp_accounting.rdp.rdp_privacy_accountant import RdpAccountant
from dp_accounting.dp_event import SampledWithoutReplacementDpEvent, GaussianDpEvent
from dp_accounting.privacy_accountant import NeighboringRelation

steps = 0

class individual_Grad_Accountant():
    def __init__(self, params):
        self.indiv_Accountants = [list() for i in range(params["training"]["subset_of_data"])]
    
    def update_norms(self, params, DEVICE, epoch, model, train_loader, optimizer, criterion):
        return
    

    def rdp_step(self, idx, grad_norm_list):
        for (index, norm) in zip(idx, grad_norm_list):
            self.indiv_Accountants[index].append(norm)
        return

    
    def return_epsilon(self, delta):
        epsilons = [sum(Accountant) for Accountant in self.indiv_Accountants]
        print(epsilons)
        return epsilons
    


class individual_RDP_Accountant():
    def __init__(self, params):
        self.possible_grad_norms, self.possible_SWOR_DP_events = self.generate_DP_events(params)
        self.indiv_Accountants = self.generate_Accountants(params)


    def generate_DP_events(self, params):
        possible_grad_norms = [0 + i * params["DP"]["rounding"] for i in range(1, int(params["DP"]["clipping_norm"] / params["DP"]["rounding"]) + 1)]
        possible_true_sigmas = [params["DP"]["sigma"] * params["DP"]["clipping_norm"] / poss_grad_norm for poss_grad_norm in possible_grad_norms]
        possible_DP_events = [GaussianDpEvent(true_sigma) for true_sigma in possible_true_sigmas]
        possible_SWOR_DP_events = [SampledWithoutReplacementDpEvent(params["training"]["subset_of_data"], params["training"]["batch_size"], poss_DP_event) for poss_DP_event in possible_DP_events]
        return possible_grad_norms, possible_SWOR_DP_events


    def generate_Accountants(self, params):
        orders = [*range(2, 1024, 1)]
        indiv_Accountants = [RdpAccountant(orders, NeighboringRelation.REPLACE_ONE) for i in range(params["training"]["subset_of_data"])]
        return indiv_Accountants
    

    def get_grad_batch_norms(self, model, DEVICE):
        model_params = list(model.named_parameters())
        n = model_params[0][1].grad_batch.shape[0]

        grad_norm_list = torch.zeros(n).to(DEVICE)
        for name, p in model_params: 
            if('bn' not in name):
            # print(name)
                flat_g = p.grad_batch.reshape(n, -1)
                current_norm_list = torch.norm(flat_g, dim=1)
                grad_norm_list += torch.square(current_norm_list)
        grad_norm_list = torch.sqrt(grad_norm_list)

        return grad_norm_list


    def update_norms(self, params, DEVICE, epoch, model, train_loader, optimizer, criterion):
        # print('updating norms at step %d'%(epoch))
        model.eval()
        norms_list = []
        idx_list = []
        for _, (data, target, idx,  _) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            minibatch_idx = idx

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            print("here1")
            with backpack(BatchGrad()):
                loss.backward()
                norms = self.get_grad_batch_norms(model, DEVICE)
            idx_list.append(minibatch_idx)
            norms_list.append(norms)
        idx = torch.cat(idx_list)
        norms = torch.cat(norms_list)
        model.train()
        self.norms = norms
        self.idx = idx

        self.rounded_norms = [self.possible_grad_norms[np.array(abs(norm.item() - np.array(self.possible_grad_norms))).argmin()] for norm in self.norms]


    def rdp_step(self):
        DP_events = [self.possible_SWOR_DP_events[self.possible_grad_norms.index(rounded_norm)] for rounded_norm in self.rounded_norms]
        for DP_event, Accountant in zip(DP_events, self.indiv_Accountants):
            Accountant.compose(DP_event)

    def return_epsilon(self, delta):
        epsilons = [Accountant.get_epsilon(delta) for Accountant in self.indiv_Accountants]
        print(epsilons)
        return epsilons


def clip_and_noise_gradients(params, DEVICE, model):
    model_params = list(model.named_parameters())
    max_clip = params["DP"]["clipping_norm"]
    noise_multiplier = params["DP"]["sigma"]
    batchsize = params["training"]["batch_size"]
    n = model_params[0][1].grad_batch.shape[0]
    grad_norm_list = torch.zeros(n).to(DEVICE)
    for name, p in model_params: 
        flat_g = p.grad_batch.reshape(n, -1)
        current_norm_list = torch.norm(flat_g, dim=1)
        grad_norm_list += torch.square(current_norm_list)
    grad_norm_list = torch.sqrt(grad_norm_list)
    scaling = max_clip/grad_norm_list
    scaling[scaling>1] = 1

    for name, p in model_params:
        p_dim = len(p.shape)
        scaling = scaling.view([n] + [1]*p_dim)
        p.grad_batch *= scaling
        p.grad = torch.sum(p.grad_batch, dim=0)
        p.grad_batch.mul_(0.)

    for name, p in model_params:
        if('bn' not in name):
            grad_noise = torch.normal(0, noise_multiplier*max_clip/batchsize, size=p.grad.shape, device=p.grad.device)
            p.grad.data += grad_noise
        else:
            p.grad = None
    return [min(grad_norm_list_elem.item(), max_clip) for grad_norm_list_elem in grad_norm_list]

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
               train_loader_active,
               indiv_Accountant):
    # Run full batch, sum up the gradients. Then we filter and replace the train_loader
    start_time = time.time()
    total = 0
    correct = 0
    global steps
    if False: #params["Inform"]["remove"]:
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
                steps += 1
                if steps % params["DP"]["update_norms_every"] == 0 and indiv_Accountant:
                    indiv_Accountant.update_norms(params, DEVICE, steps, model, train_loader_active, optimizer, criterion)
                data, target = data.to(DEVICE), target.to(DEVICE)
                with backpack(BatchGrad()):
                    optimizer.zero_grad()

                    output = model(data)
                    loss = criterion(output, target)
                    loss_list.append(loss)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    loss.backward() # This takes the most time in the private setting
                    if params["DP"]["private"]:
                        clip_and_noise_gradients(params, DEVICE, model)
                        indiv_Accountant.rdp_step()
                    optimizer.step()
                    if model.clip_weights != None:
                        model.clip_weights(params["training"]["clip_weight_value"])
                    

    else:
        for _, (data, target, idx, _) in enumerate(train_loader_active):
            steps += 1
            if steps % params["DP"]["update_norms_every"] == 0 and indiv_Accountant:
                indiv_Accountant.update_norms(params, DEVICE, steps, model, train_loader_active, optimizer, criterion)
            data, target = data.to(DEVICE), target.to(DEVICE)
            with backpack(BatchGrad()):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                loss.backward()
                if params["DP"]["private"]:
                    grad_norm_list = clip_and_noise_gradients(params, DEVICE, model)
                    indiv_Accountant.rdp_step(idx, grad_norm_list)
                optimizer.step()
                if model.clip_weights != None:
                    model.clip_weights(params["training"]["clip_weight_value"])
    accuracy = correct/total
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    print(f"training accuracy {accuracy:.4f}, epoch took {time.time() - start_time:.4f}")


def train(
    params,
    model,
    DEVICE,
    train_loader_0,
    test_loader,
    optimizer,
    criterion,
    indiv_Accountant = None
):

    # Compute all the individual norms (actually the squared norms squares are saved here)
    print('Hello world', file=sys.stderr, flush=True)
    print(f'File name: {params["model"]["name"]}')
    print(f'lr: {params["training"]["learning_rate"]}', flush=True)
    print(f'l2: {params["training"]["l2_regularizer"]}', flush=True)
    print(f'epochs: {params["training"]["num_epochs"]}', flush=True)
    print(f'batchsize: {params["training"]["batch_size"]}', flush=True)
    recorded_data = []


    if indiv_Accountant:
        indiv_Accountant.update_norms(params, DEVICE, 0, model, train_loader_0, optimizer, criterion)
    global steps
    steps = 0
    for epoch in tqdm(range(params["training"]["num_epochs"])):
        model.train()
        # print(epoch, flush=True)
        normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               train_loader_0,
               indiv_Accountant)

        if epoch % params["testing"]["test_every"] == 0 and epoch != 0:
            test(DEVICE, model, test_loader)
        
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
        # print(f"Epoch took {end_epoch-start_epoch}")


    recorded_data.append(log_data_final(params, 
                                        DEVICE,
                                        train_loader_0, 
                                        model, 
                                        indiv_Accountant))
    

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
        model = model_class(len(train_X_example), N_CLASSES)
        # model.ReLU_inplace_to_False(model.features)
        model = model.to(DEVICE)
        model = ModuleValidator.fix_and_validate(model)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    model.features = extend(model.features)
    criterion = extend(criterion)
    indiv_Accountant = None
    if params["DP"]["private"]:
        indiv_Accountant = individual_Grad_Accountant(params)

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
        indiv_Accountant
    )

def pretty_print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            pretty_print_dict(value, indent + 4)
        else:
            print(" " * indent + f"{key}: {value}")


if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_rem", type=int, default=200, help="Value for lerining_rate (optional)")
    parser.add_argument("--idx", type=int, default=0, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default="", help="Value for lerining_rate (optional)")
    args = parser.parse_args()

    print("--------------------------")
    print("--------------------------")
    N_REMOVE = args.n_rem
    try_num = 0

    params = {}
    params["save"] = True
    params["model"] = {}
    params["model"]["seed"] = 472168
    params["model"]["model"] = "med_cnn"
    params["model"]["dataset_name"] = "cifar10"
    params["model"]["name_base"] = "laplace_"
    params["model"]["name"] = "laplace_"
    params["model"]["name"] = params["model"]["name_base"] + str(try_num)
    params["training"] = {}
    params["training"]["clip_weight_value"] = 1 #0.05
    params["training"]["batch_size"] = 2048
    params["training"]["learning_rate"] = 3e-03 # -3 for mnist
    params["training"]["l2_regularizer"] = 1e-02
    params["training"]["num_epochs"] = 38
    params["training"]["subset_of_data"] = 50000
    params["testing"] = {}
    params["testing"]["test_every"] = 24
    params["Inform"] = {}
    params["Inform"]["remove"] = True
    params["Inform"]["idx"] = args.idx
    params["Inform"]["approximation"] = "AsdlGGN"
    params["Inform"]["representation"] = "diag"
    params["Paths"] = {}
    params["Paths"]["final_save_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_" + str(args.name) + str(N_REMOVE) + "_idx_" + str(params["Inform"]["idx"])
    params["Paths"]["gradient_save_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_" + str(args.name) + str(N_REMOVE) + "_idx_" + str(params["Inform"]["idx"])
    params["logging"] = {}
    params["logging"]["every_epoch"] = ["per_class_accuracies"]
    params["logging"]["final"] = ["model_laplace", "idp_accountant"]
    params["DP"] = {}
    params["DP"]["private"] = True
    params["DP"]["clipping_norm"] = 3
    params["DP"]["sigma"] = 0.2
    params["DP"]["delta"] = 1e-5
    params["DP"]["rounding"] = 0.1
    params["DP"]["update_norms_every"] = 10
    params["DP"]["how_many"] = 3

    pretty_print_dict(params)
    
    print("--------------------------")
    print("Load data")
    print("--------------------------")
    dataset_class, data_path = get_dataset(params["model"]["dataset_name"])
    n_length = params["training"]["subset_of_data"]
    print(f"Limit to first {n_length} of data")
    data_set = dataset_class(data_path, train=True)
    print("here0", flush=True)

    active_indices = [*range(params["training"]["subset_of_data"])]
    data_set.reduce_to_active(active_indices)

    data_set_test = dataset_class(data_path, train=False)
    # data_set_test._set_classes([0, 5])
    test_loader = torch.utils.data.DataLoader(
        data_set_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


    params["training"]["batch_size"] = params["training"]["batch_size"] - 1
    params["training"]["subset_of_data"] = params["training"]["subset_of_data"] - 1
    for i in range(N_REMOVE):
        print("--------------------------")
        print(f"Compute Remove Model {i}", flush=True)
        print("--------------------------")
        data_set_rm = deepcopy(data_set)
        data_set_rm.remove_curr_index_from_data(params["Inform"]["idx"])
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



    files =  os.listdir(params["Paths"]["gradient_save_path"])

    kl1_all = {}
    kl2_all = {}
    kl1_all_max = {}
    kl2_all_max = {}
    for file_invest in tqdm(files):
        kl_1_indiv = []
        kl_2_indiv = []
        with open(params["Paths"]["gradient_save_path"] + "/" + file_invest, 'rb') as file_data_invest:
            data_invest = pickle.load(file_data_invest)
            data_invest = data_invest[-1]
            data_invest_mean = data_invest["laplace_approx_mean"]
            data_invest_prec = data_invest["laplace_approx_precision"]
            removed_idx = data_invest["removed_idx"]

        for file_compare in files:
            if file_compare == file_invest:
                continue
            with open(params["Paths"]["gradient_save_path"] + "/" + file_compare, 'rb') as file_data_compare:
                data_compare = pickle.load(file_data_compare)
                data_compare = data_compare[-1]
                data_compare_mean = data_compare["laplace_approx_mean"]
                data_compare_prec = data_compare["laplace_approx_precision"]
            kl1 = computeKL(data_invest_mean, data_compare_mean, data_invest_prec, data_compare_prec)
            kl2 = computeKL(data_compare_mean, data_invest_mean, data_compare_prec, data_invest_prec)
            kl_1_indiv.append(kl1)
            kl_2_indiv.append(kl1)
        kl1_all[removed_idx] = np.mean(kl_1_indiv)
        kl2_all[removed_idx] = np.mean(kl_2_indiv)
        kl1_all_max[removed_idx] = np.max(kl_1_indiv)
        kl2_all_max[removed_idx] = np.max(kl_2_indiv)
    results = {}
    results["kl1_mean"] = kl1_all
    results["kl2_mean"] = kl2_all
    results["kl1_max"] = kl1_all_max
    results["kl2_max"] = kl2_all_max
    if not os.path.exists(params["Paths"]["final_save_path"]):
        os.makedirs(params["Paths"]["final_save_path"])
    with open(params["Paths"]["final_save_path"] + "/" + "final.pkl", 'wb') as file:
        pickle.dump(results, file)
    print(kl1_all)
