
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os


import numpy as np
import torch

def process_grad_batch(params, max_clip):
    n = params[0][1].grad_batch.shape[0]
    grad_norm_list = torch.zeros(n)
    for name, p in params: 
        flat_g = p.grad_batch.reshape(n, -1)
        current_norm_list = torch.norm(flat_g, dim=1)
        grad_norm_list += torch.square(current_norm_list)
    grad_norm_list = torch.sqrt(grad_norm_list)
    scaling = max_clip/grad_norm_list
    scaling[scaling>1] = 1

    for name, p in params:
        p_dim = len(p.shape)
        scaling = scaling.view([n] + [1]*p_dim)
        p.grad_batch *= scaling
        p.grad = torch.mean(p.grad_batch, dim=0)
        p.grad_batch.mul_(0.)

    return grad_norm_list

def get_grad_batch_norms2(DEVICE, parameters):

    grad_norm_list = None
    for (ii, p) in enumerate(parameters):
        if grad_norm_list == None:
            n = p.grad_sample.shape[0]
            grad_norm_list = torch.zeros(n).to(DEVICE)
        per_sample_grad = p.grad_sample
        per_sample_grad = torch.reshape(per_sample_grad, (per_sample_grad.shape[0], per_sample_grad.shape[1], -1))
        dims = list(range(1, len(per_sample_grad.shape)))
        per_sample_grad_norms = per_sample_grad.norm(dim=dims).to(DEVICE)
        grad_norm_list += per_sample_grad_norms ** 2
    return grad_norm_list

def get_grad_batch_norms(params):
    n = params[0][1].grad_batch.shape[0]

    grad_norm_list = torch.zeros(n)
    for name, p in params: 
        if('bn' not in name):
        # print(name)
            flat_g = p.grad_batch.reshape(n, -1)
            current_norm_list = torch.norm(flat_g, dim=1)
            grad_norm_list += torch.square(current_norm_list)
    grad_norm_list = torch.sqrt(grad_norm_list)

    return grad_norm_list




    