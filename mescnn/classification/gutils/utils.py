import torch
import argparse


def get_proper_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.has_mps:
        return torch.device("mps")
    else:
        torch.device("cpu")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
