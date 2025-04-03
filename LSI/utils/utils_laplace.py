import torch
import numpy as np
from torch.utils.data import TensorDataset
from laplace import Laplace



def get_mean_and_prec(data_set, model, mode, DEVICE):
    data, labels = torch.cat(data_set.data).cpu(), torch.cat(data_set.labels).cpu()
    # labels = torch.from_numpy(np.asarray(labels)).to(torch.long)
    # data = torch.from_numpy(np.asarray(data)).to(torch.float32)
    train_loader = torch.utils.data.DataLoader(
        TensorDataset(data, labels),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )
    la = Laplace(model.features.to(DEVICE), 'classification', subset_of_weights='all', hessian_structure=mode)
    la.fit(train_loader)
    mean = la.mean.cpu().numpy()
    if mode == "kfac":
        post_prec = la.posterior_precision
        post_prec.eigenvalues = [[tensor.cpu() for tensor in ev] for ev in post_prec.eigenvalues]
        post_prec.eigenvectors = [[tensor.cpu() for tensor in ev] for ev in post_prec.eigenvectors]
    else:
        post_prec = la.posterior_precision.cpu().numpy()
    return mean, post_prec