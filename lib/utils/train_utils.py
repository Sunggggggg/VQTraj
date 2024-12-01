import torch

def prepare_batch(batch):
    for k, v in batch.items() :
        if isinstance(v, torch.Tensor):
            batch[k] = batch[k].float().cuda()

    return batch