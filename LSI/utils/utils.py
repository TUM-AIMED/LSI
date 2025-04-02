import random
import numpy as np
import torch
import gc


def set_seed(seed):
    """
    Set the random seed for reproducibility across random, numpy, and PyTorch.
    
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
def set_deterministic():
    """
    Configure PyTorch to use deterministic algorithms for reproducibility.
    This may impact performance but ensures consistent results.
    """
    gc.collect()
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False  
