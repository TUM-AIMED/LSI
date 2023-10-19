import sys
import os
import pickle

from models.model_helper import get_model
from Datasets.dataset_helper import get_dataset
from opacus.validators.module_validator import ModuleValidator
from copy import deepcopy
from tqdm import tqdm
import torch
import numpy as np
import warnings
import argparse
import time
import random
from laplace.curvature import AsdlGGN, BackPackGGN, AsdlHessian, AsdlEF, BackPackEF
from laplace import Laplace
from scipy.linalg import logm, expm, norm
from backpack import backpack, extend
from backpack.extensions import (
    KFAC
)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def _compute_log_det(diag_array):
    det = np.sum(np.log(diag_array))
    return det


def _computeKL(mean1, mean2, precision1, precision2):
    inv_precision1 = 1/precision1
    inv_precision2 = 1/precision2
    mean_difference = mean2 - mean1
    test1 = np.sum(np.multiply(precision2, inv_precision1))
    test2 = np.sum(np.multiply(mean_difference, np.multiply(precision2, mean_difference))) 
    test3 = len(mean1) 
    test4 = _compute_log_det(inv_precision2) - _compute_log_det(inv_precision1)
    kl = 0.5*(np.sum(np.multiply(precision2, inv_precision1)) 
              + np.sum(np.multiply(mean_difference, np.multiply(precision2, mean_difference))) 
              - len(mean1) 
              + _compute_log_det(inv_precision2) - _compute_log_det(inv_precision1))
    return kl   

def _create_laplace_approx(backend_class, representation, model, train_loader):
    la = Laplace(model, 'classification',
                 subset_of_weights='all',
                 hessian_structure=representation,
                 backend=backend_class)
    la.fit(train_loader)
    mean = la.mean.cpu().numpy()
    if representation == "kron":
        post_prec = post_prec = la.posterior_precision
    elif representation == "diag":
        post_prec = la.posterior_precision.cpu().numpy()
    return mean, post_prec    

def _computeblockKL(mean0, mean1, blocks0, blocks1):
    def inverse_block_diag(blocks):
        inv = []
        for block in blocks:
            if len(block) == 2: #Kron decomposed
                inv_block = [torch.inverse(block[0]), torch.inverse(block[1])]
            elif len(block) == 1: # Full matrix
                inv_block = [torch.inverse(block[0])]
            else:
                raise Exception("Not defined")
            inv.append(inv_block)
        return inv
    def mult_block_diags(blocks0, blocks1):
        prod = []
        for block0, block1 in zip(blocks0, blocks1):
            if len(block0) == 2:
                prod_block = [torch.matmul(block0[0], block1[0]), torch.matmul(block0[1], block1[1])]
            elif len(block0) == 1:
                prod_block = [torch.matmul(block0[0], block1[0])]
            else:
                raise Exception("Not defined")
            prod.append(prod_block)
        return prod
    def trace_block_diag(blocks):
        trace = 0
        for block in blocks:
            if len(block) == 2: #Kron decomposed
                trace += torch.trace(block[0]) * torch.trace(block[1])
            elif len(block) == 1: # Full matrix
                trace += torch.trace(block[0])
            else:
                raise Exception("Not defined")
        return trace
    def logdet_block_diag(blocks):
        det = 0
        for block in blocks:
            if len(block) == 2: #Kron decomposed
                test1 = torch.logdet(block[0])
                test2 = torch.logdet(block[1])
                test3 = torch.det(block[1])
                test4 = torch.log(test3)
                det += torch.logdet(block[0])*block[1].shape[0] + torch.logdet(block[1])*block[0].shape[0]
            elif len(block) == 1: # Full matrix
                det += torch.logdet(block[0])
            else:
                raise Exception("Not defined")
        return det
    
    def logdetquot_block_diag(blocks0, blocks1):
        # blocks0/blocks1
        def logdet(block0, block1):
            np_block0 = block0.numpy().astype('float64') 
            np_block1 = block1.numpy().astype('float64') 
            logm_block0 = logm(np_block0)
            logm_block1 = logm(np_block1)
            test0 = expm(logm_block0)
            test1 = expm(logm_block1)
            try:
                closeness0 = norm(block0.numpy() - test0)
                closeness1 = norm(block1.numpy() - test1)
            except:
                print("closeness could not be computed")
            logdet0 = np.trace(logm_block0)
            logdet1 = np.trace(logm_block1)
            return np.real(logdet0 - logdet1)

        det = 0
        for block0, block1 in zip(blocks0,blocks1):
            if len(block0) == 2: #Kron decomposed
                block0_size = block0[0].shape[0]
                block1_size = block0[1].shape[0]

                logdet_res = logdet(block0[0], block1[0])
                det += block1_size * logdet_res

                logdet_res = logdet(block0[1], block1[1])
                det += block0_size * logdet_res

            elif len(block0) == 1:
                det += logdet(block0[0], block1[0])
            else:
                raise Exception("Not defined")
        return det
            

    def left_right_prod(blocks, l_vector, r_vector):
        # Right multiply
        res = 0
        for block, l_vec, r_vec in zip(blocks, l_vector, r_vector):
            if len(block) == 2: #Kron decomposed
                a1 = block[0]
                a2 = block[1]
                vec_mat = torch.transpose(torch.reshape(torch.Tensor(r_vec), (a1.shape[0], a2.shape[0])), 1, 0)
                inter = torch.matmul(vec_mat, torch.transpose(a1, 1, 0))
                inter2 = torch.matmul(a2, inter)
                inter3 = torch.flatten(torch.transpose(inter2, 1, 0)) # [20, 12, 13, 8]
                result1 = torch.dot(l_vec, inter3)

                # full = torch.kron(a1, a2)
                # inter_full = torch.matmul(full, r_vec) # [20, 14, 14, 10]
                # res_full = torch.dot(l_vec, torch.matmul(full, r_vec)) # 130

                res += result1
            elif len(block) == 1: # Full matrix
                res += torch.dot(l_vec, torch.matmul(block[0], torch.Tensor(r_vec)))
            else:
                raise Exception("Not defined")
        return res

    def compute_whole_kl(mean0, mean1, blocks0, blocks1):
        # inv_blocks = variance
        # blocks = precision
        inv_blocks1 = inverse_block_diag(blocks1)
        inv_blocks0 = inverse_block_diag(blocks0)
        mean_diff = [mean1_val - mean0_val for mean0_val, mean1_val in zip(mean0, mean1)]

        test1 = trace_block_diag(mult_block_diags(blocks1, inv_blocks0))
        test3 = len(torch.cat(mean_diff))
        test4 = left_right_prod(blocks1, mean_diff, mean_diff)
        test5 = logdetquot_block_diag(blocks0, blocks1)
        kl = 0.5*(
            trace_block_diag(mult_block_diags(blocks1, inv_blocks0)) - 
            len(torch.cat(mean_diff)) + 
            left_right_prod(blocks1, mean_diff, mean_diff) + 
            logdetquot_block_diag(blocks0, blocks1) # instead of variance, inverse relation of precisions
        )
        return kl

    kl1 = compute_whole_kl(mean0, mean1, blocks0, blocks1)
    kl2 = compute_whole_kl(mean1, mean0, blocks1, blocks0)
    return kl1, kl2


def computeKL(backend_class, representation, model1, model2, train_loader, weight_reg):
    mean1, prec1 = _create_laplace_approx(backend_class, representation, model1, train_loader)
    mean2, prec2 = _create_laplace_approx(backend_class, representation, model2, train_loader)
    if representation == "kron":
        kl1, kl2 = _computeKL_from_kron(mean1, mean2, prec1, prec2, weight_reg)
    elif representation == "diag":
        kl1 = _computeKL(mean1, mean2, prec1, prec2)
        kl2 = _computeKL(mean2, mean1, prec2, prec1)
    elif representation == "full":
        kl1 = _computeKL_from_full(mean1, mean2, prec1, prec2)
    mean_diff_sum = np.sum(mean1 - mean2)
    mean_diff_mean = np.mean(mean1 - mean2)
    print(f"kl1 {kl1} kl2 {kl2} ")
    return kl1, kl2, mean_diff_sum, mean_diff_mean

def _compute_block_diag(DEVICE, model, model2, train_loader, criterion):
    model = extend(model.features, use_converter=True)
    model2 = extend(model2.features, use_converter=True)
    criterion = extend(criterion)
    data_all = []
    target_all = []
    for _, (data, target, idx, _) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        data_all.extend(data)
        target_all.extend(target)
    data_all = torch.stack(data_all)
    target_all = torch.tensor(target_all)

    output = model(data_all)
    loss = criterion(output, target_all)
    with backpack(KFAC()):
        loss.backward()
    
    output2 = model2(data_all)
    loss2 = criterion(output2, target_all)
    with backpack(KFAC()):
        loss2.backward()

    blocks = []
    means = []
    for name, param in model.named_parameters():
        blocks.append(param.kfac)
        means.append(torch.flatten(param.data))

    blocks2 = []
    means2 = []
    for name, param in model2.named_parameters():
        blocks2.append(param.kfac)
        means2.append(torch.flatten(param.data))
    z_test = torch.equal(torch.cat(means), torch.cat(means2))
    return blocks, means, blocks2, means2

def _add_prior_to_block(weight_reg, blocks):
    # https://www.cs.toronto.edu/~rgrosse/icml2016-kfc.pdf
    # https://arxiv.org/pdf/2106.14806.pdf
    prior_value = 1/(weight_reg**2)
    res_blocks = []
    for block in blocks:
        if len(block) == 2: #Kron decomposed
            pi_l = torch.sqrt(
                (torch.trace(block[0])/(block[1].shape[0] + 1))/
                (torch.trace(block[1])/(block[0].shape[0]))
                ) # https://arxiv.org/pdf/1503.05671.pdf
            res_blocks.append([block[0] - pi_l * np.sqrt(prior_value) * torch.eye(block[0].shape[0]),
                               block[1] - pi_l * np.sqrt(prior_value) * torch.eye(block[1].shape[0])])
        elif len(block) == 1: # Full matrix
            res_blocks.append([block[0] - prior_value * torch.eye(block[0].shape[0])])
        else:
            raise Exception("Not defined")
    return res_blocks


def _computeKL_from_kron(mean0, mean1, precision0, precision1, weight_reg):
    def split_values_by_lengths(values, lengths):
        split_tensors = []
        start_idx = 0
        for length in lengths:
            split_portion = values[start_idx:start_idx + length]
            split_tensor = torch.tensor(split_portion)
            split_tensors.append(split_tensor)
            start_idx += length
        return split_tensors

    # block0 = precision0.to_matrix() --> Own implementation is way faster
    block_sizes0 = []
    block0 = []
    for block_eigval, block_eigvec in zip(precision0.eigenvalues, precision0.eigenvectors):
        if len(block_eigvec) == 2:
            q0 = block_eigvec[0]
            q0_inv = torch.inverse(q0)
            delta0 = torch.diag(block_eigval[0])

            q1 = block_eigvec[1]
            q1_inv = torch.inverse(q1)
            delta1 = torch.diag(block_eigval[1])

            block0.append([torch.matmul(q0, torch.matmul(delta0, q0_inv)), torch.matmul(q1, torch.matmul(delta1, q1_inv))])
            block_sizes0.append(q0.shape[0] * q1.shape[0])
        elif len(block_eigvec) == 1:
            q0 = block_eigvec[0]
            q0_inv = torch.inverse(q0)
            delta0 = torch.diag(block_eigval[0])
            block0.append([torch.matmul(q0, torch.matmul(delta0, q0_inv))]) 
            block_sizes0.append(q0.shape[0])
        else:
            raise Exception("Error")
        
    block_sizes1 = []
    block1 = []
    for block_eigval, block_eigvec in zip(precision1.eigenvalues, precision1.eigenvectors):
        if len(block_eigvec) == 2:
            q0 = block_eigvec[0]
            q0_inv = torch.inverse(q0)
            delta0 = torch.diag(block_eigval[0])

            q1 = block_eigvec[1]
            q1_inv = torch.inverse(q1)
            delta1 = torch.diag(block_eigval[1])

            block1.append([torch.matmul(q0, torch.matmul(delta0, q0_inv)), torch.matmul(q1, torch.matmul(delta1, q1_inv))])
            block_sizes1.append(q0.shape[0] * q1.shape[0])
        elif len(block_eigvec) == 1:
            q0 = block_eigvec[0]
            q0_inv = torch.inverse(q0)
            delta0 = torch.diag(block_eigval[0])
            block1.append([torch.matmul(q0, torch.matmul(delta0, q0_inv))]) 
            block_sizes1.append(q0.shape[0])
        else:
            raise Exception("Error")
    mean0 = split_values_by_lengths(mean0, block_sizes0)
    mean1 = split_values_by_lengths(mean1, block_sizes1)
    block0 = _add_prior_to_block(weight_reg, block0)
    block1 = _add_prior_to_block(weight_reg, block1)
    kl1, kl2 = _computeblockKL(mean0, mean1, block0, block1)
    print(f"kl1 {kl1} kl2 {kl2}")
    return kl1, kl2           


def normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               train_loader_active):
    # Run full batch, sum up the gradients. Then we filter and replace the train_loader
    start_time = time.time()
    total = 0
    correct = 0
    loss_list = []
    for _, (data, target, idx, _) in enumerate(train_loader_active):
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
    print(f"training accuracy {accuracy:.4f}, epoch took {time.time() - start_time:.4f}")
    return np.mean(np.array(loss_list))


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
    print(f"val accuracy {accuracy:.4f}, epoch took {time.time() - start_time:.4f}")
    return np.mean(np.array(loss_list))


def train(
    params,
    model,
    DEVICE,
    train_loader_0,
    test_loader,
    optimizer,
    criterion,
    indiv_Accountant = None,
    rem = False
):

    # Compute all the individual norms (actually the squared norms squares are saved here)
    print(f'File name: {params["model"]["name"]}')
    print(f'lr: {params["training"]["learning_rate"]}', flush=True)
    print(f'l2: {params["training"]["l2_regularizer"]}', flush=True)
    print(f'epochs: {params["training"]["num_epochs"]}', flush=True)
    print(f'batchsize: {params["training"]["batch_size"]}', flush=True)

    early_stopper = EarlyStopper(patience=5, min_delta=0.2)    

    for epoch in range(params["training"]["num_epochs"]):
        model.train()
        train_loss = normal_train_step(params,
               model, 
               optimizer, 
               DEVICE, 
               criterion, 
               train_loader_0)
        model.eval()
        val_loss = normal_val_step(params,
            model, 
            optimizer, 
            DEVICE, 
            criterion, 
            test_loader)
        if not rem:
            if early_stopper.early_stop(val_loss): 
                print(f"Stopping early at epoch {epoch}")            
                break
        print(f"Epoch {epoch} with training_loss {train_loss} and val_loss {val_loss}")
    return model, epoch


def train_with_params(
    params,
    train_loader_0,
    test_loader,
    rem = False
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

    model_class = get_model(params["model"]["model"])
    

    train_X_example, _, _, _ = train_loader_0.dataset[0]
    N = len(train_loader_0.dataset)
    print(f"Length = {N}")

    N_CLASSES =  len(np.unique(train_loader_0.dataset.labels))

    if N < params["training"]["batch_size"]:
        raise Exception(f"Batchsize of {params['training']['batch_size']} is larger than the size of the dataset of {N}")
    
    if params["model"]["model"] == "mlp" or params["model"]["model"] == "small_mlp":
        model = ModuleValidator.fix_and_validate(model_class(len(torch.flatten(train_X_example)), N_CLASSES).to(DEVICE))
    else:
        model = model_class(len(train_X_example), N_CLASSES)
        model = model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["training"]["learning_rate"],
        weight_decay=params["training"]["l2_regularizer"],
    )



    return criterion, train(
                        params,
                        model,
                        DEVICE,
                        train_loader_0,
                        test_loader,
                        optimizer,
                        criterion,
                        rem=rem
                    )




if __name__ == "__main__":
    """
    __main__ Main starting point for running a dpsgt algorithm

    - edit: json_file_path for config file
    """ 
    parser = argparse.ArgumentParser(description="Process optional float inputs.")
    parser.add_argument("--n_rem", type=int, default=0, help="Value for lerining_rate (optional)")
    parser.add_argument("--n_seeds", type=int, default=100, help="Value for lerining_rate (optional)")
    parser.add_argument("--name", type=str, default="", help="Value for lerining_rate (optional)")
    parser.add_argument("--repr", type=str, default="kron", help="Value for lerining_rate (optional)")
    parser.add_argument("--lap_type", type=str, default="asdlgnn", help="Value for lerining_rate (optional)")
    args = parser.parse_args()

    print("--------------------------")
    print("--------------------------")
    N_REMOVE = args.n_rem
    N_SEEDS = args.n_seeds
    try_num = 0

    if args.lap_type == "asdlgnn":
        backend_class = AsdlGGN
    elif args.lap_type == "asdlhessian":
        backend_class = AsdlHessian
    elif args.lap_type == "asdlef":
        backend_class = AsdlEF
    elif args.lap_type == "backpackgnn":
        backend_class = BackPackGGN
    elif args.lap_type == "backpackef":
        backend_class = BackPackEF
    else:
        raise Exception("Not implemented")
    
    representation = args.repr

    params = {}
    params["save"] = True
    params["model"] = {}
    params["model"]["seed"] = 472168
    params["model"]["model"] = "small_mlp"
    params["model"]["dataset_name"] = "mnist"
    params["model"]["name_base"] = "laplace_Mnist_"
    params["model"]["name"] = "laplace_Mnist_"
    params["model"]["name"] = params["model"]["name_base"] + str(try_num)
    params["training"] = {}
    params["training"]["batch_size"] = 256
    params["training"]["learning_rate"] = 0.00008 #3e-03 # -3 for mnist
    params["training"]["l2_regularizer"] = 1e-02
    params["training"]["num_epochs_init"] = 10
    params["testing"] = {}
    params["testing"]["test_every"] = 24
    params["Paths"] = {}
    params["Paths"]["final_save_path"] = "/vol/aimspace/users/kaiserj/Individual_Privacy_Accounting/results_kl_indiv_script/results_" + str(args.name)

    keep_indices = [*range(1500)]

    print("--------------------------")
    print("Load data")
    print("--------------------------")

    data_set_class, data_path = get_dataset(params["model"]["dataset_name"])

    data_set = data_set_class(data_path, train=True)
    data_set.reduce_to_active(keep_indices)
    print(np.unique(data_set.labels))
    if N_REMOVE == 0:
        N_REMOVE = len(data_set.labels)


    data_set_test = data_set_class(data_path, train=False)
    test_loader = torch.utils.data.DataLoader(
        data_set_test,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )



    resultskl1 = {}
    resultskl2 = {}
    resultstarget = {}
    mean_diff_sum = {}
    mean_diff_mean = {}
    for seed in tqdm(range(N_SEEDS)):
        params["training"]["num_epochs"] = params["training"]["num_epochs_init"]

        resultskl1[seed] = {}
        resultskl2[seed] = {}
        resultstarget[seed] = {}
        mean_diff_sum[seed] = {}
        mean_diff_mean[seed] = {}

        params["model"]["seed"] = seed

        train_loader_0 = torch.utils.data.DataLoader(
                data_set,
                batch_size=params["training"]["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            
        criterion, (model_all, epochs) = train_with_params(
            params,
            train_loader_0,
            test_loader
        )
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # compute_block_diag(DEVICE, model_all, train_loader_0, criterion)
        params["training"]["num_epochs"] = epochs

    
        for rem_idx in range(N_REMOVE):
            print("--------------------------")
            print(f"Compute Remove Model {rem_idx}", flush=True)
            print("--------------------------")
            data_set_rm = deepcopy(data_set)
            data_set_rm.remove_curr_index_from_data(rem_idx)
            
            train_loader_rm = torch.utils.data.DataLoader(
                data_set_rm,
                batch_size=params["training"]["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            
            _, (model_rem, _) = train_with_params(
                params,
                train_loader_rm,
                test_loader,
                rem = True
            )
           # kl1, kl2 = computeKL_from_blocks(DEVICE, model_rem, model_all, train_loader_0, criterion, params["training"]["l2_regularizer"])
            kl1, kl2, mean_diff_s, mean_diff_m = computeKL(backend_class, representation, model_rem, model_all, train_loader_0, params["training"]["l2_regularizer"]) 
            kl11, kl22, mean_diff_s, mean_diff_m = computeKL(backend_class, "diag", model_rem, model_all, train_loader_0, params["training"]["l2_regularizer"]) 
            kl11, kl22, mean_diff_s, mean_diff_m = computeKL(backend_class, "full", model_rem, model_all, train_loader_0, params["training"]["l2_regularizer"]) 

            # print(f"kl1 {kl1}, kl2 {kl2}")
            true_rm_idx = data_set.active_indices[rem_idx]
            true_rm_idx2 = keep_indices[rem_idx]
            if true_rm_idx != true_rm_idx2:
                raise Exception("Something went wrong here")
            resultskl1[seed][true_rm_idx] = kl1
            resultskl2[seed][true_rm_idx] = kl2
            mean_diff_sum[seed][true_rm_idx] = mean_diff_s
            mean_diff_mean[seed][true_rm_idx] = mean_diff_m
            resultstarget[seed][true_rm_idx] = data_set.labels[rem_idx]

    if not os.path.exists(params["Paths"]["final_save_path"]):
        os.makedirs(params["Paths"]["final_save_path"])
    results_all = {
        "kl1": resultskl1,
        "kl2": resultskl2,
        "labels": resultstarget,
        "mean_diff_sum": mean_diff_sum,
        "mean_diff_mean": mean_diff_mean
    }
    with open(params["Paths"]["final_save_path"] + "/results_all.pkl", 'wb') as file:
        pickle.dump(results_all, file)
    print(f'Saving at {params["Paths"]["final_save_path"] + "/results_all.pkl"}')

