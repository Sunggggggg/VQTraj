import torch

def count_param(model):
    return sum([v.numel() for v in model.parameters() if v.requires_grad])

def print_batch(batch):
    for k, v in batch.items() :
        if isinstance(v, torch.Tensor):
            print(k, v.shape)