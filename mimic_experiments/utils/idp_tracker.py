import torch
import torch.nn as nn

import numpy as np

import time

import sys

import threading
from multiprocessing import Pool
import os
from utils.rdp_accountant import compute_rdp, get_privacy_spent
from utils.idp_tracker_utils import get_grad_batch_norms2

def update_norms(DEVICE,
                 epoch,
                 model,
                 optimizer,
                 criterion,
                 train_loader,
                 idp_accountant):
    print('updating norms at epoch %d'%(epoch))

    # model.eval() # Change this if model is not MLP

    norms_list = []
    idx_list = []

    for _, (data, target, idx, _) in enumerate(train_loader):
        minibatch_idx = idx
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        norms = get_grad_batch_norms2(DEVICE, model.parameters())
        idx_list.append(minibatch_idx)
        norms_list.append(norms)

    idx = torch.cat(idx_list)
    norms = torch.cat(norms_list)
    idp_accountant.update_norm(DEVICE, norms, idx)

def computeEpsFunc(args):
    #get_privacy_spent(orders, rdp, target_delta=delta)
    pid, rdps, orders, delta, result, job_idx = args

    job_size = job_idx.shape[0]

    base_progress = max(job_size // 10, 1)

    for i, idx in enumerate(job_idx):

        eps, _, opt_order = get_privacy_spent(orders, rdps[idx], target_delta=delta)
        result[idx] = eps
        if(i%base_progress==0):
            progress = (i+1)/job_size
            #print('computing eps thread %d, progress: %.1f%%'%(pid, progress*100))

    return job_idx, result[job_idx]

def computeRdpFunc(args):
    pid, sigmas, all_q, result, job_idx, orders = args

    job_size = job_idx.shape[0]

    base_progress = max(job_size // 10, 1)



    for i, idx in enumerate(job_idx):
        q = all_q[idx]
        sigma = sigmas[idx]
        
        n_orders = orders.shape[0]

        result_ = compute_rdp(q, sigma, 1, orders=orders)

        result[idx] = result_
        if(i%base_progress==0):
            progress = (i+1)/job_size
            #print('comput rdp thread %d, progress: %.1f%%'%(pid, progress*100))

    return job_idx, result[job_idx]

class PrivacyLossTracker(nn.Module):
    def __init__(self, DEVICE, n, batchsize, sigma, init_norm=10, orders=np.arange(2, 1024, 1), delta=1e-5, rounding=0.1):
        
        self.init_norm = init_norm

        self.norms = torch.zeros(n) + init_norm
        self.norms = self.norms.to(DEVICE)
        
        self.rounding = rounding
        self.all_possible_norms = []
        tmp_norm = rounding
        while tmp_norm < init_norm:
            self.all_possible_norms.append(tmp_norm)
            tmp_norm += rounding

        self.all_possible_norms = torch.tensor(self.all_possible_norms).to(DEVICE)


        self.sigma = sigma
        self.batchsize = batchsize
        
        self.n = n
        self.orig_n = self.n
        self.q = batchsize/n

        # sampling probabilities for all data points
        self.all_q = np.array([self.q]*self.all_possible_norms.shape[0])

        self.delta = delta

        self.orders = orders

        init_rdp = compute_rdp(self.q, sigma, 1, orders=orders)
        print(f"Init- all current RDP")
        print(f"Computing inital rdp_values, with {self.q} and {self.sigma}")
        print(f"max in current: {np.max(init_rdp)}")
        print(f"min in current: {np.min(init_rdp)}\n")
        
        self.accmulated_rdp = torch.zeros(size=(n, orders.shape[0])).to(DEVICE)

        self.current_rdp = torch.zeros(size=(n, orders.shape[0])) + torch.tensor(init_rdp).float()
        self.current_rdp = self.current_rdp.to(DEVICE)

        self.all_levels_rdp = torch.zeros(size=(self.all_possible_norms.shape[0], orders.shape[0])).to(DEVICE)



    
    def round_norms(self, idx): ## always need to call this when the norms or rdps are updated

        norm_diff = torch.abs(self.norms[idx].view(idx.shape[0], 1) - self.all_possible_norms)
        min_diff_idx = torch.argmin(norm_diff, dim=1)
        for i in range(idx.shape[0]):
            self.norms[idx[i]] = self.all_possible_norms[min_diff_idx[i]]
            self.current_rdp[idx[i]] = self.all_levels_rdp[min_diff_idx[i]]

    def update_sigma(self, DEVICE, sigma):
        self.sigma = sigma
        self.update_rdp(DEVICE)

    def get_avg_norm(self):
        return torch.mean(self.norms).item()

    def update_rdp(self, DEVICE):
        different_sigmas = {}

        for i, norm in enumerate(self.all_possible_norms):
            relative_sigma = self.sigma * (self.init_norm/norm).item()
            if(relative_sigma not in different_sigmas.keys()):
                different_sigmas[relative_sigma] = [i]
            else:
                different_sigmas[relative_sigma].append(i)

        sigmas_to_compute = list(different_sigmas.keys())

        
        full_job_size = len(sigmas_to_compute)
        full_idx = np.arange(full_job_size)

        num_workers = os.cpu_count() // 2
        if(full_job_size<20):
            num_workers = 1

        per_workder_load = full_job_size // num_workers

        job_idxs = []

        for i in range(num_workers):
            if(i == num_workers - 1):
                idx = full_idx[i*per_workder_load:]
            else:
                idx = full_idx[i*per_workder_load:(i+1)*per_workder_load]
            job_idxs.append(idx)
        result = np.zeros(shape=(full_job_size, self.orders.shape[0]))

        args_list = []
        for i in range(num_workers):
            #pid, sigmas, q, result, job_idx, orders
            args = [i, sigmas_to_compute, self.all_q, result, job_idxs[i], self.orders]
            args_list.append(args)

        with Pool(num_workers) as p:
            result_tuples = p.map(computeRdpFunc, args_list)
            res_list = []
            for tup in result_tuples:
                res_list.append(tup[1])


            result = np.concatenate(res_list)


        for i in range(full_job_size):
            sigma = sigmas_to_compute[i]
            rdp = result[i]

            idx = different_sigmas[sigma]

            self.all_levels_rdp[idx] = torch.tensor(rdp).float().to(DEVICE)
        print(f"Init- all levels RDP")
        print(f"max in levels: {torch.max(self.all_levels_rdp)}")
        print(f"min in levels: {torch.min(self.all_levels_rdp)}\n")

        self.round_norms(np.arange(self.n))


    def update_norm(self, DEVICE, norms, idx):
        self.norms[idx] = norms.to(DEVICE)
        self.norms[self.norms>self.init_norm] = self.init_norm

        self.round_norms(idx)


    def update_loss(self):
        self.accmulated_rdp += self.current_rdp

    def parallel_get_eps(self):

        full_job_size = self.orig_n
        full_idx = np.arange(full_job_size)

        num_workers = 1 # os.cpu_count() // 2
        per_workder_load = full_job_size // num_workers

        job_idxs = []

        for i in range(num_workers):
            if(i == num_workers - 1):
                idx = full_idx[i*per_workder_load:]
            else:
                idx = full_idx[i*per_workder_load:(i+1)*per_workder_load]
            job_idxs.append(idx)

        result = np.zeros(full_job_size)

        args_list = []
        for i in range(num_workers):
            #pid, rdps, orders, delta, result, job_idx = args
            args = [i, self.accmulated_rdp.cpu().numpy(), self.orders, self.delta, result, job_idxs[i]]
            args_list.append(args)

        with Pool(num_workers) as p:
            result_tuples = p.map(computeEpsFunc, args_list)
            res_list = []
            for tup in result_tuples:
                res_list.append(tup[1])
            
            result = np.concatenate(res_list)

        return result
