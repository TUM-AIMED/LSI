import numpy as np
import torch
from laplace import Laplace
from scipy.linalg import logm, expm, norm
from backpack import backpack, extend
from backpack.extensions import (
    KFAC
)
import time
import pickle
from collections import defaultdict
from Datasets.dataset_helper import get_dataset

from laplace.utils.matrix import Kron


import numpy as np

def kl_divergence_kron_ala_gpt(mu1, mu2, sigma1_blocks, sigma2_blocks):
    """ 
    Compute the KL divergence between two multivariate normal distributions 
    with block diagonal covariance matrices represented in the form of 
    Kronecker products or full blocks.
    """

    def kronecker_product(A, B):
        """ Compute the Kronecker product of two matrices A and B """
        return np.kron(A, B)

    def block_matrix_inverse(blocks):
        """ Compute the inverse of a block diagonal matrix represented by blocks """
        inverse_blocks = []
        for block in blocks:
            if len(block) == 2:
                # If block is represented in Kronecker form
                inv_block1 = np.linalg.inv(block[0])
                inv_block2 = np.linalg.inv(block[1])
                inverse_blocks.append([inv_block1, inv_block2])
            else:
                # If block is a full block
                inverse_blocks.append([np.linalg.inv(block[0])])
        return inverse_blocks

    def block_matrix_det(blocks):
        """ Compute the determinant of a block diagonal matrix represented by blocks """
        det = 1.0
        for block in blocks:
            if len(block) == 2:
                # If block is represented in Kronecker form
                # t1 = np.linalg.det(block[0])
                # t2 = np.linalg.det(block[1])
                # t3 = block[1].shape[1]
                # t4 = block[0].shape[0]

                # sign, logdet1 = np.linalg.slogdet(block[0])
                # determinant1 = sign * np.exp(logdet1)

                # sign, logdet2 = np.linalg.slogdet(block[1])
                # determinant2 = sign * np.exp(logdet2)

                # det *= np.linalg.det(block[0]) ** block[1].shape[1] * np.linalg.det(block[1]) ** block[0].shape[0]

                # sign, logdet3 = np.linalg.slogdet(np.kron(block[0], block[1]))
                # determinant3 = sign * np.exp(logdet3)
                # det *= determinant3
                det *= np.linalg.det(np.kron(block[0], block[1]))
                print("")

            else:
                # If block is a full block
                det *= np.linalg.det(block[0])
        return det

    def block_matrix_trace(A_blocks, B_blocks):
        """ Compute the trace of the product of two block diagonal matrices A and B """
        trace = 0.0
        for A_block, B_block in zip(A_blocks, B_blocks):
            if len(A_block) == 2:
                A_full = np.kron(A_block[0], A_block[1])
            else:
                A_full = A_block[0]
            
            if len(B_block) == 2:
                B_full = np.kron(B_block[0], B_block[1])
            else:
                B_full = B_block[0]
            mul = np.dot(A_full, B_full)
            trace += np.trace(mul)
        return trace

    def compute_quadratic_term(mu_diff, sigma2_inv_blocks):
        """ Compute the quadratic term (mu_diff.T @ sigma2_inv @ mu_diff) """
        quad_term = 0.0
        start_index = 0
        for sigma2_inv_block in sigma2_inv_blocks:
            if len(sigma2_inv_block) == 2:
                sigma2_inv_full = np.kron(sigma2_inv_block[0], sigma2_inv_block[1])
                block_size = sigma2_inv_block[0].shape[0] * sigma2_inv_block[1].shape[0]
            else:
                sigma2_inv_full = sigma2_inv_block[0]
                block_size = sigma2_inv_block[0].shape[0]

            mu_diff_block = mu_diff[start_index:start_index + block_size]
            quad_term += mu_diff_block.T @ sigma2_inv_full @ mu_diff_block
            start_index += block_size
        
        return quad_term

    k = mu1.shape[0]

    # Compute the inverses of the block diagonal covariance matrices
    sigma1_blocks = block_matrix_inverse(sigma1_blocks)

    sigma2_inv_blocks = sigma2_blocks
    sigma2_blocks = block_matrix_inverse(sigma2_blocks)
    
    # Compute the trace term
    trace_term = block_matrix_trace(sigma2_inv_blocks, sigma1_blocks)
    
    # Compute the quadratic term
    mu_diff = mu2 - mu1
    mu_diff = torch.tensor(mu_diff).to(torch.float64)
    quad_term = compute_quadratic_term(mu_diff, sigma2_inv_blocks)

    # Compute the log determinant ratio
    # det_sigma1 = block_matrix_det(sigma1_blocks) + 0.0000000001
    # det_sigma2 = block_matrix_det(sigma2_blocks) + 0.0000000001
    # log_det_ratio = np.log(det_sigma2 / det_sigma1)
    log_det_ratio = 0

    # Combine all terms to compute the KL divergence
    kl_div = 0.5 * (trace_term + quad_term - k + log_det_ratio)
    
    return kl_div


def _computeKL(mean1, mean2, precision1, precision2):
    mean1 = np.longdouble(mean1)
    mean2 = np.longdouble(mean2)
    precision1 = np.longdouble(precision1)
    precision2 = np.longdouble(precision2)
    zer0 = np.count_nonzero(precision1 == 0)
    zer1 = np.count_nonzero(precision2 == 0)
    def _compute_log_det(diag_array):
        det = np.sum(np.log(diag_array))
        return det
    mean_difference = mean2 - mean1
    addit = "300"

    inv_precision1 = 1/precision1
    inv_precision2 = 1/precision2
    part0 = np.multiply(precision2, inv_precision1)
    part00 = np.sum(part0)
    part1 = np.sum(np.multiply(precision2, inv_precision1))
    part2 = np.sum(np.multiply(mean_difference, np.multiply(precision2, mean_difference))) 
    part3 = len(mean1) 
    part4 = _compute_log_det(inv_precision2) - _compute_log_det(inv_precision1)
    kl = 0.5*(np.sum(np.multiply(precision2, inv_precision1)) 
              + np.sum(np.multiply(mean_difference, np.multiply(precision2, mean_difference))) 
              - len(mean1) 
              + _compute_log_det(inv_precision2) - _compute_log_det(inv_precision1))
    if kl < -0.00:
        print("shit")
    return kl, np.sum(mean_difference**2)   

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
    def trace_block_diag(blocks0, blocks1):
        trace = 0
        for block0, block1 in zip(blocks0, blocks1):
            if len(block0) == 2: #Kron decomposed
                trace += torch.trace(block0[0] @ block1[0]) * torch.trace(block0[1] @ block1[1])
            elif len(block0) == 1: # Full matrix
                trace += torch.trace(block0[0] @ block1[0]) 
            else:
                raise Exception("Not defined")
        return trace
    # def logdet_block_diag(blocks):
    #     det = 0
    #     for block in blocks:
    #         if len(block) == 2: #Kron decomposed
    #             test1 = torch.logdet(block[0])
    #             test2 = torch.logdet(block[1])
    #             test3 = torch.det(block[1])
    #             test4 = torch.log(test3)
    #             det += torch.logdet(block[0])*block[1].shape[0] + torch.logdet(block[1])*block[0].shape[0]
    #         elif len(block) == 1: # Full matrix
    #             det += torch.logdet(block[0])
    #         else:
    #             raise Exception("Not defined")
    #     return det
    
    def logdetquot_block_diag(blocks0, blocks1):
        # blocks0/blocks1
        # def logdet():
   
        #     return np.real(logdet0 - logdet1)

        det = 1
        for block0, block1 in zip(blocks0,blocks1):
            if len(block0) == 2: #Kron decomposed
                det_block0 = block0[1].shape[0] * np.linalg.det(block0[0]) + block0[0].shape[0] * np.linalg.det(block0[1])

                det_block1 = block1[1].shape[0] * np.linalg.det(block1[0]) + block1[0].shape[0] * np.linalg.det(block1[1])
                det *= det_block0/det_block1

            elif len(block0) == 1:
                det *= np.linalg.det(block0[0])/np.linalg.det(block1[0])
            else:
                raise Exception("Not defined")
            
        retval = np.log(det)
        return retval
            

    def left_right_prod(blocks, l_vector, r_vector):
        # Right multiply
        res = 0
        for block, l_vec, r_vec in zip(blocks, l_vector, r_vector):
            # l_vec = l_vec.to(DEVICE)
            # r_vec = l_vec.to(DEVICE)
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

        blocks0 = [[ten.to(torch.float64) for ten in kron] for kron in blocks0 ]
        blocks1 = [[ten.to(torch.float64) for ten in kron] for kron in blocks1 ]
        mean0 = [ten.to(torch.float64) for ten in mean0] 
        mean1 = [ten.to(torch.float64) for ten in mean1] 
        inv_blocks0 = inverse_block_diag(blocks0)

        test = blocks0[0][0] @ inv_blocks0[0][0]
        mean_diff = [mean1_val - mean0_val for mean0_val, mean1_val in zip(mean0, mean1)]

        trace_part = trace_block_diag(blocks1, inv_blocks0)
        len_part = len(torch.cat(mean_diff))
        prod_part = left_right_prod(blocks1, mean_diff, mean_diff)
        logdet_part = logdetquot_block_diag(blocks0, blocks1)
        kl = 0.5*(
            0 * trace_part - 
            0 * len_part + 
            prod_part + 
            logdet_part # instead of variance, inverse relation of precisions
        )
        return kl
    
    def compute_whole_kl_via_full(mean0, mean1, blocks0, blocks1):
        def kronecker_product(tensors):
            """
            Compute the Kronecker product of a list of tensors.
            """
            if len(tensors) == 0:
                raise ValueError("List of tensors must not be empty.")
            
            result = tensors[0]
            for tensor in tensors[1:]:
                result = torch.kron(result, tensor)
            return result

        def construct_full_matrix(blocks):
            """
            Construct a full matrix from a nested list of block matrices.
            """
            block_matrices = []
            
            for block in blocks:
                if isinstance(block, list):
                    # Compute Kronecker product if block is a list of tensors
                    block_matrix = kronecker_product(block)
                elif isinstance(block, torch.Tensor):
                    # Use the tensor directly if it is a single tensor
                    block_matrix = block
                else:
                    raise ValueError("Each block must be either a list of tensors or a single tensor.")
                
                block_matrices.append(block_matrix)
            
            # Construct block diagonal matrix
            full_matrix = torch.block_diag(*block_matrices)
            
            return full_matrix
        
        full_prec0 = construct_full_matrix(blocks0).numpy()
        full_prec1 = construct_full_matrix(blocks1).numpy()
        kl1 = _computeKL_from_full(mean0, mean1, full_prec0, full_prec1)
        return kl1

    kl1 = compute_whole_kl(mean0, mean1, blocks0, blocks1)
    # kl2 = compute_whole_kl(mean1, mean0, blocks1, blocks0)

    # kl1, diff = compute_whole_kl_via_full(mean0, mean1, blocks0, blocks1)
    return kl1
    

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
        return .5 * (tr_term + det_term + quad_term - N), diff
    print("comp from full")

    def compute_inverse(matrix):
        # Ensure the matrix is a square matrix
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("The matrix must be square to compute its inverse.")
        
        # Create the identity matrix of the same size
        identity = np.eye(matrix.shape[0])
        
        # Solve A * X = I for X
        inverse = np.linalg.solve(matrix, identity)
        
        return inverse
    
    prec1 = prec1.astype(np.float64)
    prec2 = prec2.astype(np.float64)
    mean1 = mean1.astype(np.float128)
    mean2 = mean2.astype(np.float128)

    cov1 = compute_inverse(prec1)
    cov2 = compute_inverse(prec2)

    kl1, diff = kl_mvn(mean1, cov1, mean2, cov2, prec2)
    # kl2 = kl_mvn(mean2, cov2, mean1, cov1, prec1)
    return kl1, np.sum(diff**2)# , kl2

def computeKL(DEVICE, backend_class, representation, model_rm, model_all, train_loader, train_loader_rm, weight_reg, subset_of_weights="all"):
    mean1, prec1 = _create_laplace_approx(backend_class, representation, model_all, train_loader, subset_of_weights=subset_of_weights)
    mean2, prec2 = _create_laplace_approx(backend_class, representation, model_rm,  train_loader_rm, subset_of_weights=subset_of_weights)
    # kl1_elem, kl2_elem = elem_wise_KL(DEVICE, mean1, mean2, prec1, prec2)
    kl1_elem = 0
    kl2_elem = 0
    start_time = time.time()
    if representation == "kron":
        kl1, kl2 = _computeKL_from_kron(DEVICE, mean1, mean2, prec1, prec2, weight_reg)
    elif representation == "diag":
        kl1 = _computeKL(mean1, mean2, prec1, prec2)
        # kl2 = _computeKL(mean2, mean1, prec2, prec1)
        kl2 = 0
    elif representation == "full":
        kl1, kl2 = _computeKL_from_full(mean1, mean2, prec1, prec2)
    mean_diff_sum = np.sum(abs(mean1 - mean2))
    mean_diff_mean = np.mean(mean1 - mean2)
    # print(f"kl1 {kl1} kl2 {kl2} ")
    # print(f"computation took {time.time() - start_time}")

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

def _computeKL_from_kron(mean0, mean1, precision0, precision1):
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
            q0_t = q0.T
            delta0 = torch.diag(block_eigval[0])

            q1 = block_eigvec[1]
            q1_t = q1.T
            delta1 = torch.diag(block_eigval[1])

            block0.append([q0 @ delta0 @ q0_t, q1 @ delta1 @ q1_t])
            block_sizes0.append(q0.shape[0] * q1.shape[0])
        elif len(block_eigvec) == 1:
            q0 = block_eigvec[0]
            q0_t = q0.transpose(0, 1)
            delta0 = torch.diag(block_eigval[0])
            block0.append([q0 @ delta0 @ q0_t]) 
            block_sizes0.append(q0.shape[0])
        else:
            raise Exception("Error")
        
    block_sizes1 = []
    block1 = []
    for block_eigval, block_eigvec in zip(precision1.eigenvalues, precision1.eigenvectors):
        if len(block_eigvec) == 2:
            q0 = block_eigvec[0]
            q0_t = q0.transpose(0, 1)
            delta0 = torch.diag(block_eigval[0])

            q1 = block_eigvec[1]
            q1_t = q1.transpose(0, 1)
            delta1 = torch.diag(block_eigval[1])

            block1.append([q0 @ delta0 @ q0_t, q1 @ delta1 @ q1_t])
            block_sizes1.append(q0.shape[0] * q1.shape[0])
        elif len(block_eigvec) == 1:
            q0 = block_eigvec[0]
            q0_t = q0.transpose(0, 1)
            delta0 = torch.diag(block_eigval[0])
            block1.append([q0 @ delta0 @ q0_t]) 
            block_sizes1.append(q0.shape[0])
        else:
            raise Exception("Error")
    # mean0 = split_values_by_lengths(mean0, block_sizes0)
    # mean1 = split_values_by_lengths(mean1, block_sizes1)

    block0 = [[tensor.to(torch.float64) for tensor in prec] for prec in block0]
    block1 = [[tensor.to(torch.float64) for tensor in prec] for prec in block1]
    # mean0 = [tensor.to(torch.float64) for tensor in mean0]
    # mean1 = [tensor.to(torch.float64) for tensor in mean1]
 
    kl1 = kl_divergence_kron_ala_gpt(mean0, mean1, block0, block1) # _computeblockKL
    kl1 = kl1.item()
    return kl1, 0    

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


def load_sorted_idx(path):
    with open(path, 'rb') as file:
        grad_data = pickle.load(file)
    idxs = []
    result = []
    for seed, seed_data in grad_data.items():
        seed_wise_results = []
        idx = []
        for index, idx_data in seed_data.items():
            idx.append(index)
            seed_wise_results.append(np.sqrt(np.sum(idx_data)))
        result.append(seed_wise_results)
        idxs.append(idx)
    if not all(inner_list == idxs[0] for inner_list in idxs):
        raise Exception("some mix up")

    idx = idxs[0]
    result = np.mean(result, axis=0)
    combined_data = list(zip(result, idx))
    sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
    kl1_diag, idx = zip(*sorted_data)
    return list(idx)


def get_images_form_idx(idxs):
    dataset_class, data_path = get_dataset("cifar100")
    data_set = dataset_class(data_path, train=True)
    # data_set._set_classes([0, 5])
    # data_set._set_classes([4, 9]) # mnist
    images = []
    labels =[]
    labels = data_set.labels[idxs]
    images = data_set.data[idxs]
    images = [im.transpose(1, 2, 0) for im in images]
    return images, labels

def load_sorted_classes(path):
    with open(path, 'rb') as file:
        grad_data = pickle.load(file)
    idxs = []
    result = []
    for seed, seed_data in grad_data.items():
        seed_wise_results = []
        idx = []
        for index, idx_data in seed_data.items():
            idx.append(index)
            seed_wise_results.append(np.sqrt(np.sum(idx_data)))
        result.append(seed_wise_results)
        idxs.append(idx)
    if not all(inner_list == idxs[0] for inner_list in idxs):
        raise Exception("some mix up")

    idx = idxs[0]
    result = np.mean(result, axis=0)
    combined_data = list(zip(result, idx))
    sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
    kl1_diag, idx = zip(*sorted_data)
    images, label = get_images_form_idx(idx)

    all_dict = defaultdict(list)
    for lab, data in zip(label, kl1_diag):
        all_dict[lab].append(data)

    all_list = []
    keys = []
    for key, value in all_dict.items():
        all_list.append(value)
        keys.append(key)
        
    combined_data = list(zip(all_list, keys))
    sorted_data = sorted(combined_data, key=lambda x: np.median(x[0]))
    all_list, keys = zip(*sorted_data)
    return list(keys)

def load_label_sorted_idx(label, dataset_name):
    dataset_class, data_path = get_dataset(dataset_name)
    data_set = dataset_class(data_path, train=True)
    sorted_idx = []
    idx_w_lab = zip(data_set.active_indices, data_set.labels)
    for lab_l in label:
        label_list = [idx for idx, lab in idx_w_lab if lab == lab_l]
        sorted_idx.append(label_list)
    return sorted_idx

