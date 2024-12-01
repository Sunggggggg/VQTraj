import os
import yaml
import shutil
from os import path as osp
import logging
import torch

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_dict_to_yaml(obj, filename, mode='w'):
    with open(filename, mode) as f:
        yaml.dump(obj, f, default_flow_style=False)
           
def prepare_output_dir(cfg, cfg_file):

    # ==== create logdir
    logdir = osp.join(cfg.OUTPUT_DIR, cfg.EXP_NAME)
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(src=cfg_file, dst=osp.join(cfg.OUTPUT_DIR, 'config.yaml'))

    cfg.LOGDIR = logdir

    # save config
    save_dict_to_yaml(cfg, osp.join(cfg.LOGDIR, 'config.yaml'))

    return cfg

def prepare_batch(batch):
    for k, v in batch.items() :
        if isinstance(v, torch.Tensor):
            batch[k] = batch[k].float().cuda()

    return batch

def create_logger(logdir, phase='train'):
    os.makedirs(logdir, exist_ok=True)

    log_file = osp.join(logdir, f'{phase}_log.txt')

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file,
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def get_optimizer(cfg, model, optim_type, momentum, stage):
    if stage == 'stage2':
        param_list = [{'params': model.integrator.parameters()}]
        for name, param in model.named_parameters():
            # if 'integrator' not in name and 'motion_encoder' not in name and 'trajectory_decoder' not in name:
            if 'integrator' not in name:
                param_list.append({'params': param, 'lr': cfg.TRAIN.LR_FINETUNE})
    else:
        param_list = [{'params': model.parameters()}]
    
    if optim_type in ['sgd', 'SGD']:
        opt = torch.optim.SGD(lr=cfg.TRAIN.LR, params=param_list, momentum=momentum)
    elif optim_type in ['Adam', 'adam', 'ADAM']:
        opt = torch.optim.Adam(lr=cfg.TRAIN.LR, params=param_list, weight_decay=cfg.TRAIN.WD, betas=(0.9, 0.999))
    else:
        raise ModuleNotFoundError
    
    return opt