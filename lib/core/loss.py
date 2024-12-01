from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from lib.utils.torch_transform import quat_angle_diff

from configs import constants as _C

class TrajLoss(nn.Module):
    def __init__(self) :
        super().__init__()
        self.l2_loss = nn.MSELoss()

    def compute_trans_mse(self, pred, gt):
        """
        pred    : [T, B, 1, 3]
        gt      : [T, B, 1, 3] init frame [0,0,0]
        """
        pred = pred - pred[:, :1]
        return self.l2_loss(pred, gt)

    def compute_orient_angle_loss(self, pred, gt):
        """
        pred    : [T, B, 1, 4]
        gt      : [T, B, 1, 4] 
        """
        diff_angle = quat_angle_diff(pred, gt)
        return diff_angle.pow(2).mean()

    def compute_orient_6d_loss(self, pred, gt):
        """
        pred    : [T, B, 1, 6]
        gt      : [T, B, 1, 6] 
        """
        return self.l2_loss(pred, gt)

    def compute_local_orient_heading(self, pred):
        """
        pred : [T, B, 1, 6]
        """
        return pred.pow(2).mean()
    
    def compute_dheading(self, pred):
        """
        pred : [T, B, 1, 2]
        """
        return pred.pow(2).mean()

    def forward(self, pred, gt):
        loss = 0.0

        loss_traj = self.compute_trans_mse(pred['out_trans_tp'], gt['w_transl_tp'])
        loss_orient_q =self.compute_orient_angle_loss(pred['out_orient_q_tp'], gt['w_orient_q_tp'])
        loss_commite = pred['commit_loss']
        loss_orient_6d = self.compute_orient_6d_loss(pred['out_orient_6d_tp'], gt['w_orient_6d_tp'])
        loss_local_head = self.compute_local_orient_heading(pred['local_orient'])
        loss_dheading = self.compute_dheading(pred['d_heading_vec'])

        loss_traj *= 100
        loss_orient = loss_orient_q + loss_orient_6d
        loss_orient *= 100
        loss_commite *= 0.1
        loss_local_head *= 10
        loss_dheading *= 10

        loss_dict = {
            'traj' : loss_traj,
            'orient': loss_orient,
            'commit': loss_commite,
            'local_head': loss_local_head,
            'dheading': loss_dheading,
        }

        loss = sum(loss for loss in loss_dict.values())
        return loss, loss_dict