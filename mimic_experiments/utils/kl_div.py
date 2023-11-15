import numpy as np
import torch
from laplace import Laplace
from scipy.linalg import logm, expm, norm
from backpack import backpack, extend
from backpack.extensions import (
    KFAC
)
import time



def _computeKL(mean1, mean2, precision1, precision2):
    def _compute_log_det(diag_array):
        det = np.sum(np.log(diag_array))
        return det
    mean_difference = mean2 - mean1
    addit = "300"

    inv_precision1 = 1/precision1
    inv_precision2 = 1/precision2
    # plot_and_save_histogram(precision1, "z"+ addit + "_prec_1", bins=100)
    # plot_and_save_histogram(precision2, "z"+ addit + "_prec_2", bins=100)
    # plot_and_save_histogram(inv_precision1, "z"+ addit + "_inv_1", bins=100)
    # plot_and_save_histogram(inv_precision2, "z"+ addit + "_inv_2", bins=100)
    # plot_and_save_lineplot([*range(len(inv_precision1))], inv_precision1, "z"+ addit + "_inv_line1")
    # plot_and_save_lineplot([*range(len(inv_precision2))], inv_precision2, "z"+ addit + "_inv_line2")
    # plot_and_save_lineplot([*range(len(inv_precision2))], inv_precision1 - inv_precision2, "z"+ addit + "_inv_line_diff")
    # plot_and_save_lineplot([*range(len(inv_precision2))], np.multiply(mean_difference, np.multiply(precision2, mean_difference)), "z"+ addit + "_quad_term")
    # plot_and_save_lineplot_with_running_sum([*range(len(inv_precision2))], np.multiply(precision2, inv_precision1) - 1, "z"+ addit + "_trace_term")
    # plot_and_save_lineplot_with_running_sum([*range(len(inv_precision2))], np.log(inv_precision2) - np.log(inv_precision1), "z"+ addit + "_det_term")

    kl = 0.5*(np.sum(np.multiply(precision2, inv_precision1)) 
              + np.sum(np.multiply(mean_difference, np.multiply(precision2, mean_difference))) 
              - len(mean1) 
              + _compute_log_det(inv_precision2) - _compute_log_det(inv_precision1))
    return kl   

def _create_laplace_approx(backend_class, representation, model, train_loader, subset_of_weights="all"):
    la = Laplace(model, 'classification',
                 subset_of_weights=subset_of_weights,
                 hessian_structure=representation,
                 backend=backend_class)
    start_time = time.time()
    la.fit(train_loader)
    mean = la.mean.cpu().numpy()
    start_time = time.time()
    if representation == "kron":
        post_prec = post_prec = la.posterior_precision
    elif representation == "diag":
        post_prec = la.posterior_precision.cpu().numpy()
    elif representation == "full":
        post_prec = la.posterior_precision.cpu().numpy()
    return mean, post_prec    

def _computeblockKL(DEVICE, mean0, mean1, blocks0, blocks1):
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
            np_block0 = block0.cpu().numpy().astype('float64') 
            np_block1 = block1.cpu().numpy().astype('float64') 
            logm_block0 = logm(np_block0)
            logm_block1 = logm(np_block1)
            test0 = expm(logm_block0)
            test1 = expm(logm_block1)
            try:
                closeness0 = norm(block0.cpu().numpy() - test0)
                closeness1 = norm(block1.cpu().numpy() - test1)
            except:
                print("")# print("closeness could not be computed")
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
            l_vec = l_vec.to(DEVICE)
            r_vec = l_vec.to(DEVICE)
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
        inv_blocks0 = inverse_block_diag(blocks0)
        mean_diff = [mean1_val - mean0_val for mean0_val, mean1_val in zip(mean0, mean1)]

        trace_part = trace_block_diag(mult_block_diags(blocks1, inv_blocks0))
        len_part = len(torch.cat(mean_diff))
        prod_part = left_right_prod(blocks1, mean_diff, mean_diff)
        logdet_part = logdetquot_block_diag(blocks0, blocks1)
        kl = 0.5*(
            trace_part - 
            len_part + 
            prod_part + 
            logdet_part # instead of variance, inverse relation of precisions
        )
        return kl

    kl1 = compute_whole_kl(mean0, mean1, blocks0, blocks1)
    kl2 = compute_whole_kl(mean1, mean0, blocks1, blocks0)
    return kl1, kl2
    

def _computeKL_from_full(mean1, mean2, prec1, prec2):
    def kl_mvn(m0, S0, m1, S1, iS1):
        """
        Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
        Also computes KL divergence from a single Gaussian pm,pv to a set
        of Gaussians qm,qv.
        """
        # store inv diag covariance of S1 and diff between means
        N = m0.shape[0]
        # iS1 = np.linalg.inv(S1)
        diff = m1 - m0
        # kl is made of three terms
        start_time = time.time()
        tr_term   = np.trace(iS1 @ S0)
        print(f"trace took {time.time() - start_time}")
        start_time = time.time()
        # det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
        (sign0, logdet0) = np.linalg.slogdet(S0)
        (sign1, logdet1) = np.linalg.slogdet(S1)
        det_term = sign1 * logdet1 - sign0 * logdet0
        print(f"det_term took {time.time() - start_time}")
        start_time = time.time()
        quad_term = diff.T @ iS1 @ diff
        print(f"quad_term  took {time.time() - start_time}")
        #print(tr_term,det_term,quad_term)
        return .5 * (tr_term + det_term + quad_term - N)
    cov1 = np.linalg.inv(prec1)
    cov2 = np.linalg.inv(prec2)
    kl1 = kl_mvn(mean1, cov1, mean2, cov2, prec2)
    kl2 = kl_mvn(mean2, cov2, mean1, cov1, prec1)
    return kl1, kl2

def computeKL(DEVICE, backend_class, representation, model_rm, model_all, train_loader, train_loader_rm, weight_reg, subset_of_weights="all"):
    mean1, prec1 = _create_laplace_approx(backend_class, representation, model_all, train_loader, subset_of_weights=subset_of_weights)
    mean2, prec2 = _create_laplace_approx(backend_class, representation, model_rm,  train_loader_rm, subset_of_weights=subset_of_weights)
    kl1_elem, kl2_elem = elem_wise_KL(DEVICE, mean1, mean2, prec1, prec2)
    start_time = time.time()
    if representation == "kron":
        kl1, kl2 = _computeKL_from_kron(DEVICE, mean1, mean2, prec1, prec2, weight_reg)
    elif representation == "diag":
        kl1 = _computeKL(mean1, mean2, prec1, prec2)
        kl2 = _computeKL(mean2, mean1, prec2, prec1)
    elif representation == "full":
        kl1, kl2 = _computeKL_from_full(mean1, mean2, prec1, prec2)
    mean_diff_sum = np.sum(abs(mean1 - mean2))
    mean_diff_mean = np.mean(mean1 - mean2)
    print(f"kl1 {kl1} kl2 {kl2} ")
    print(f"computation took {time.time() - start_time}")
    return kl1, kl2, mean_diff_sum, mean_diff_mean, kl1_elem, kl2_elem

# Manual computation of block diag
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

def _add_prior_to_block(DEVICE, weight_reg, blocks):
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
            res_blocks.append([block[0] - pi_l * np.sqrt(prior_value) * torch.eye(block[0].shape[0]).to(DEVICE),
                               block[1] - pi_l * np.sqrt(prior_value) * torch.eye(block[1].shape[0]).to(DEVICE)])
        elif len(block) == 1: # Full matrix
            res_blocks.append([block[0] - prior_value * torch.eye(block[0].shape[0]).to(DEVICE)])
        else:
            raise Exception("Not defined")
    return res_blocks

def _computeKL_from_kron(DEVICE, mean0, mean1, precision0, precision1, weight_reg):
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
    block0 = _add_prior_to_block(DEVICE, weight_reg, block0)
    block1 = _add_prior_to_block(DEVICE, weight_reg, block1)
    kl1, kl2 = _computeblockKL(DEVICE, mean0, mean1, block0, block1)
    kl1 = kl1.item()
    kl2 = kl2.item()
    return kl1, kl2    

def elem_wise_KL(DEVICE, mean0, mean1, precision0, precision1):
    def one_dir_kl(mean0, mean1, precision0, precision1):
        mean_diff = mean1-mean0
        spur_term = precision1/precision0
        logdet_term = np.log(precision0/precision1)
        square_term = mean_diff**2 * precision1

        kl = 0.5*(spur_term + square_term - 1 + logdet_term)
        return np.mean(kl)
    kl1 = one_dir_kl(mean0, mean1, precision0, precision1)
    kl2 = one_dir_kl(mean1, mean0, precision1, precision0)
    return kl1, kl2