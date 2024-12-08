from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from lib.utils.torch_transform import quat_angle_diff
from lib.utils.torch_transform import vec_to_heading, get_heading, rot6d_to_quat

from configs import constants as _C

class TrajLoss(nn.Module):
    def __init__(self) :
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def compute_trans_mse(self, pred, gt):
        """
        pred    : [T, B, 1, 3]
        gt      : [T, B, 1, 3] init frame [0,0,0]
        """
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
        local_orient = pred[..., 3:-2]
        if local_orient.shape[-1] == 6:
            local_orient = rot6d_to_quat(local_orient)
        heading = get_heading(local_orient)
        mse = heading.pow(2).mean()
        return mse
    
    def compute_dheading(self, pred):
        """
        pred : [T, B, 1, 2]
        """
        local_heading_vec = pred[..., -2:]
        heading = vec_to_heading(local_heading_vec)
        mse = heading.pow(2).mean()
        return mse

    def compute_vae_z_kld(self, data):
        if isinstance(data['p_z_dist'], list):
            kld = sum([q_z_dist.kl(p_z_dist) for p_z_dist, q_z_dist in zip(data['p_z_dist'], data['q_z_dist']) ])
            kld = kld.sum(-1)
            kld = kld.clamp_min_(0.0).mean()
        else :
            kld = data['q_z_dist'].kl(data['p_z_dist'])
            kld = kld.sum(-1)
            kld = kld.clamp_min_(0.0).mean()

        return kld
    
    def compute_foot_contact(self, pred, gt):
        return self.l2_loss(pred, gt)

    def forward(self, pred, gt):
        loss = 0.0

        loss_traj = self.compute_trans_mse(pred['out_trans_tp'], gt['w_transl_tp'])
        loss_orient_q =self.compute_orient_angle_loss(pred['out_orient_q_tp'], gt['w_orient_q_tp'])
        if 'commit_loss' in pred :
            loss_commite = pred['commit_loss'] * 0.001
        else :
            loss_commite = 0.

        if 'q_z_dist' in pred :
            loss_kl = self.compute_vae_z_kld(pred) * 0.001
        else :
            loss_kl = 0.
        
        if 'contact_tp' in pred :
            loss_sliding = self.compute_foot_contact(pred['pred_contact_tp'], pred['contact_tp'])
        else :
            loss_sliding = 0.

        # loss_orient_6d = self.compute_orient_6d_loss(pred['out_orient_6d_tp'], gt['w_orient_6d_tp'])
        loss_local_head = self.compute_local_orient_heading(pred['out_local_traj_tp'])
        loss_dheading = self.compute_dheading(pred['out_local_traj_tp'])

        loss_traj *= 1.0
        loss_orient = loss_orient_q # + loss_orient_6d
        loss_orient *= 1.0
        loss_sliding *= 0.1
        loss_local_head *= 0.1
        loss_dheading *= 0.1

        loss_dict = {
            'traj' : loss_traj,
            'orient': loss_orient,
            'kl': loss_kl,
            'slide': loss_sliding,
            # 'local_head': loss_local_head,
            'dheading': loss_dheading,
        }

        loss = sum(loss for loss in loss_dict.values())
        return loss, loss_dict