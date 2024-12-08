# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar

from configs import constants as _C
from lib.utils import transforms
from lib.utils.train_utils import AverageMeter, prepare_batch
from lib.models import build_body_model

logger = logging.getLogger(__name__)

class Trainer():
    def __init__(self, 
                 data_loaders,
                 network,
                 optimizer,
                 criterion=None,
                 train_stage='syn',
                 start_epoch=0,
                 checkpoint=None,
                 end_epoch=999,
                 lr_scheduler=None,
                 device=None,
                 writer=None,
                 debug=False,
                 resume=False,
                 logdir='output',
                 performance_type='min',
                 summary_iter=1,
                 ):
        
        self.train_loader, self.valid_loader = data_loaders
        
        # Model and optimizer
        self.network = network
        self.optimizer = optimizer
        
        # Training parameters
        self.train_stage = train_stage
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.writer = writer
        self.debug = debug
        self.resume = resume
        self.logdir = logdir
        self.summary_iter = summary_iter
        
        self.performance_type = performance_type
        self.train_global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.best_performance = float('inf') if performance_type == 'min' else -float('inf')
        self.summary_loss_keys = []

        self.evaluation_accumulators = dict.fromkeys(
            ['pred_traj', 'target_traj', 'pred_orient', 'target_orient'])# 'pred_verts', 'target_verts'])
        
        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.logdir)

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        if checkpoint is not None:
            self.load_pretrained(checkpoint)
        
    def train(self, ):
        # Single epoch training routine

        losses = AverageMeter()
        traj_loss = AverageMeter()
        orient_loss = AverageMeter()

        timer = {
            'data': 0,
            'forward': 0,
            'loss': 0,
            'backward': 0,
            'batch': 0,
        }

        self.network.train()
        start = time.time()
        summary_string = ''
        
        bar = Bar(f'Epoch {self.epoch + 1}/{self.end_epoch}', fill='#', max=len(self.train_loader))
        for i, batch in enumerate(self.train_loader):
            
            # <======= Feedforward 
            batch = prepare_batch(batch)
            x = batch['body_pose']
            timer['data'] = time.time() - start
            start = time.time()
            pred = self.network(batch)
            timer['forward'] = time.time() - start
            start = time.time()
            # =======>

            # <======= Backprop            
            loss, loss_dict = self.criterion(pred, pred)
            timer['loss'] = time.time() - start
            start = time.time()
            
            # Clip gradients
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
            # =======>
            
            # <======= Log training info
            total_loss = loss
            losses.update(total_loss.item(), x.size(0))
            traj_loss.update(loss_dict['traj'].item(), x.size(0))
            orient_loss.update(loss_dict['orient'].item(), x.size(0))
            
            timer['backward'] = time.time() - start
            timer['batch'] = timer['data'] + timer['forward'] + timer['loss'] + timer['backward']
            start = time.time()

            summary_string = f'({i + 1}/{len(self.train_loader)}) | Total: {bar.elapsed_td} ' \
                            f'| loss: {losses.avg:.2f} | traj: {traj_loss.avg:.2f} ' \
                            f'| orient: {orient_loss.avg:.2f} '

            for k, v in loss_dict.items():
                if k in self.summary_loss_keys: 
                    summary_string += f' | {k}: {v:.2f}'
                if (i + 1) % self.summary_iter == 0:
                    self.writer.add_scalar('train_loss/'+k, v, global_step=self.train_global_step)

            if (i + 1) % self.summary_iter == 0:
                self.writer.add_scalar('train_loss/loss', total_loss.item(), global_step=self.train_global_step)

            self.train_global_step += 1
            bar.suffix = summary_string
            bar.next(1)
            if torch.isnan(total_loss):
                exit('Nan value in loss, exiting!...')
            # =======>

        logger.info(summary_string)
        bar.finish()

    def validate(self, ):
        self.network.eval()
        
        start = time.time()
        summary_string = ''
        bar = Bar('Validation', fill='#', max=len(self.valid_loader))

        if self.evaluation_accumulators is not None:
            for k,v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.valid_loader):
                batch = prepare_batch(batch)
                
                # <======= Feedforward 
                pred = self.network(batch)
                
                pred_traj = pred['out_trans_tp']
                target_traj = pred['w_transl_tp']

                pred_orient = pred['out_orient_6d_tp']
                target_orient = pred['w_orient_6d_tp']

                
                self.evaluation_accumulators['pred_traj'].append(pred_traj.cpu().numpy())
                self.evaluation_accumulators['target_traj'].append(target_traj.cpu().numpy())
                
                self.evaluation_accumulators['pred_orient'].append(pred_orient.cpu().numpy())
                self.evaluation_accumulators['target_orient'].append(target_orient.cpu().numpy())
                # =======>
            
                batch_time = time.time() - start

                summary_string = f'({i + 1}/{len(self.valid_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                                f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

                self.valid_global_step += 1
                bar.suffix = summary_string
                bar.next()

        logger.info(summary_string)
            
        bar.finish()
    
    def evaluate(self, ):
        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = np.vstack(v)

        pred_trajs = self.evaluation_accumulators['pred_traj']
        target_trajs = self.evaluation_accumulators['target_traj']

        pred_trajs = torch.from_numpy(pred_trajs).float()
        target_trajs = torch.from_numpy(target_trajs).float()

        pred_orients = self.evaluation_accumulators['pred_orient']
        target_orients = self.evaluation_accumulators['target_orient']

        pred_orients = torch.from_numpy(pred_orients).float()
        target_orients = torch.from_numpy(target_orients).float()

        print(f'Evaluating on {pred_trajs.shape[0]} number of poses...')
        error_trajs = torch.sqrt(((pred_trajs - target_trajs) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        error_orients = torch.sqrt(((pred_orients - target_orients) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        error_trajs = np.mean(error_trajs)
        error_orients = np.mean(error_orients)
        
        eval_dict = {
            'traj': error_trajs,
            'orients': error_orients,
        }

        log_str = f'Epoch {self.epoch}, '
        log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
        logger.info(log_str)

        for k,v in eval_dict.items():
            self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)

        return error_trajs
    
    def save_model(self, performance, epoch):
        save_dict = {
            'epoch': epoch,
            'model': self.network.state_dict(),
            'performance': performance,
            'optimizer': self.optimizer.state_dict(),
        }

        filename = osp.join(self.logdir, 'checkpoint.pth.tar')
        torch.save(save_dict, filename)

        if self.performance_type == 'min':
            is_best = performance < self.best_performance
        else:
            is_best = performance > self.best_performance

        if is_best:
            logger.info('Best performance achived, saving it!')
            self.best_performance = performance
            shutil.copyfile(filename, osp.join(self.logdir, 'model_best.pth.tar'))

            with open(osp.join(self.logdir, 'best.txt'), 'w') as f:
                f.write(str(float(performance)))

    def fit(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.train()
            self.validate()
            performance = self.evaluate()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # log the learning rate
            for param_group in self.optimizer.param_groups[:2]:
                print(f'Learning rate {param_group["lr"]}')
                self.writer.add_scalar('lr', param_group['lr'], global_step=self.epoch)

            logger.info(f'Epoch {epoch+1} performance: {performance:.4f}')

            self.save_model(performance, epoch)
            self.train_loader.dataset.prepare_video_batch()

        self.writer.close()
        
    def load_pretrained(self, model_path):
        if osp.isfile(model_path):
            checkpoint = torch.load(model_path)

            # network
            ignore_keys = ['smpl.body_pose', 'smpl.betas', 'smpl.global_orient', 'smpl.J_regressor_extra', 'smpl.J_regressor_eval']
            ignore_keys2 = [k for k in checkpoint['model'].keys() if 'integrator' in k]
            ignore_keys.extend(ignore_keys2)
            model_state_dict = {k: v for k, v in checkpoint['model'].items() if k not in ignore_keys}
            model_state_dict = {k: v for k, v in model_state_dict.items() if k in self.network.state_dict().keys()}
            self.network.load_state_dict(model_state_dict, strict=False)
            
            if self.resume:
                self.start_epoch = checkpoint['epoch']
                self.best_performance = checkpoint['performance']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            logger.info(f"=> loaded checkpoint '{model_path}' "
                  f"(epoch {self.start_epoch}, performance {self.best_performance})")
        else:
            logger.info(f"=> no checkpoint found at '{model_path}'")