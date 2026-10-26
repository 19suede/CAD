import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import random
import os
import numpy as np

def create_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_dir = os.path.dirname(args.logger_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(args.logger_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def crop_patches(img, patch_size, crop_num):
    B, C, H, W = img.shape
    patches = []
    for _ in range(crop_num):
        top = random.randint(0, H - patch_size)
        left = random.randint(0, W - patch_size)
        patch = img[:, :, top:top+patch_size, left:left+patch_size]
        patches.append(patch)

    return torch.cat(patches, dim=0)

def count_param(model):
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Number of trainable params: %.2f million." % (n_parameters / 1.e6))

