import numpy as np
from datetime import datetime
import torch
import random
import os
from lightning.pytorch import seed_everything

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ensure deterministic CUDA operations for Jax (see https://github.com/google/jax/issues/13672)
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
    else:
        os.environ["XLA_FLAGS"] += " --xla_gpu_deterministic_ops=true"
    seed_everything(seed)

def get_curr_datetime_str():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime('%Y%m%d_%H%M%S')
    return formatted_datetime

def get_timestamped_str(base_str: str):
    return f"instance-{get_curr_datetime_str()}_{base_str}"

def get_instance_id():
    return f"instance-{get_curr_datetime_str()}"