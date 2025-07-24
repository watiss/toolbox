import numpy as np
from datetime import datetime
import os

def set_random_seed(seed: int):
    import torch
    from lightning.pytorch import seed_everything
    import random

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

# copied/adapted from https://github.com/snap-stanford/GEARS
def dataverse_download(url, save_path):
    """
    Dataverse download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """
    import requests
    from tqdm import tqdm

    if os.path.exists(save_path):
        print('Found local copy...')
    else:
        print("Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()