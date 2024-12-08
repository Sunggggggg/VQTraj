import torch
import pprint
import random
import numpy as np
import os
from configs.config import parse_args
from lib.utils.train_utils import get_optimizer, create_logger, prepare_output_dir
from lib.data.datasets.amass import AMASSDataset
from lib.data.datasets.dataset_eval import EvalDataset
from lib.core.loss import TrajLoss
from lib.core.trainer import Trainer

from torch.utils.tensorboard import SummaryWriter

def setup_seed(seed):
    """ Setup seed for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(cfg):
    # Seed
    if cfg.SEED_VALUE >= 0:
        setup_seed(cfg.SEED_VALUE)

    logger = create_logger(cfg.LOGDIR, phase='debug' if cfg.DEBUG else 'train')
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')
    logger.info(pprint.pformat(cfg))
    
    writer = SummaryWriter(log_dir=cfg.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)
    
    # ========= Dataloaders ========= #
    amass_dataset = AMASSDataset(cfg)
    train_dataloader = torch.utils.data.DataLoader(amass_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=4,
        shuffle=True,
        pin_memory=True
    )
    eval_dataset = EvalDataset(cfg)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )
    data_loaders = train_dataloader, eval_dataloader

    # ========= Network and Optimizer ========= #
    from lib.models.MHT.network import Network
    network = Network(stage='stage1').cuda()

    #from lib.models.GLAMR.network import Network
    #network = Network().cuda()


    if os.path.isfile(cfg.TRAIN.CHECKPOINT):
        logger.info(f"=> loaded checkpoint '{cfg.TRAIN.CHECKPOINT}' ")
        checkpoint = torch.load(cfg.TRAIN.CHECKPOINT)
        ignore_keys = ['smpl.body_pose', 'smpl.betas', 'smpl.global_orient', 'smpl.J_regressor_extra', 'smpl.J_regressor_eval']
        model_state_dict = {k: v for k, v in checkpoint['model'].items() if k not in ignore_keys}
        network.load_state_dict(model_state_dict, strict=False)
        
    optimizer = get_optimizer(
        cfg,
        model=network, 
        optim_type=cfg.TRAIN.OPTIM,
        momentum=cfg.TRAIN.MOMENTUM,
        stage=cfg.TRAIN.STAGE)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.TRAIN.MILESTONES,
        gamma=cfg.TRAIN.LR_DECAY_RATIO,
        verbose=False,
    )
    
    # ========= Loss function ========= #
    criterion = TrajLoss()

    # ========= Start Training ========= #
    Trainer(
        data_loaders=data_loaders,
        network=network,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        train_stage=cfg.TRAIN.STAGE,
        start_epoch=cfg.TRAIN.START_EPOCH,
        end_epoch=cfg.TRAIN.END_EPOCH,
        checkpoint=cfg.TRAIN.CHECKPOINT,
        device=cfg.DEVICE,
        writer=writer,
        debug=cfg.DEBUG,
        resume=cfg.RESUME,
        logdir=cfg.LOGDIR,
        summary_iter=cfg.SUMMARY_ITER,
    ).fit()

if __name__ == '__main__':
    cfg, cfg_file, _ = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)
    
    main(cfg)