import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import constants as _C
from smplx import SMPL
from lib.utils import transforms
from lib.utils.traj_utils import traj_global2local_heading, traj_local2global_heading
from .corse_net import CoarseNetwork
from .fine_net import FineNetwork
from .utils import World2Camera, get_virtual_camera

def compute_contact_label(feet, thr=1e-2, alpha=5):
    """
    feet : [B, T, 4, 3]
    """
    vel = torch.zeros_like(feet[..., 0])
    label = torch.zeros_like(feet[..., 0])
    
    vel[:, 1:-1] = (feet[:, 2:] - feet[:, :-2]).norm(dim=-1) / 2.0
    vel[:, 0] = vel[:, 1].clone()
    vel[:, -1] = vel[:, -2].clone()
    
    label = 1 / (1 + torch.exp(alpha * (thr ** -1) * (vel - thr)))
    return label

def compute_vel(kp3d):
    """
    kp3d : [B, T, J, 3]
    """
    vel = kp3d[:, 1:] - kp3d[:, :-1]
    pad_vel = torch.zeros_like(vel[:, :1])

    vel = torch.cat([pad_vel, vel], dim=1)      # [B, T, ]
    vel_tp = vel.transpose(0, 1).contiguous()
    
    return vel, vel_tp

class Network(nn.Module):
    def __init__(self, stage) :
        super().__init__()
        self.stage = stage

        self.local_orient_type = '6d'

        # ------- SMPL ------- #
        self.smpl = SMPL(_C.BMODEL.FLDR, num_betas=10, ext='pkl')
        J_regressor_wham = np.load(_C.BMODEL.JOINTS_REGRESSOR_WHAM)
        self.register_buffer('J_regressor_wham', torch.tensor(
            J_regressor_wham, dtype=torch.float32))
        
        # ------- Network ------- #
        if stage == 'stage1' :
            self.corsenet = CoarseNetwork()
        elif stage == 'stage2':
            self.corsenet = CoarseNetwork()
            self.finenet = FineNetwork()

    def init_batch_stage1(self, batch):
        data = batch.copy()
        # Input 1. Body pose
        if 'body_pose' in data :
            body_pose_tp = batch['body_pose'].transpose(0, 1).contiguous()
            body_pose_6d_tp = transforms.matrix_to_rotation_6d(body_pose_tp).reshape(body_pose_tp.shape[:2] + (-1,))
            body_pose_aa_tp = transforms.matrix_to_axis_angle(body_pose_tp).reshape(body_pose_tp.shape[:2] + (-1,))

            batch['body_pose_6d_tp'] = body_pose_6d_tp   # [T, B, 138]
            batch['body_pose_aa_tp'] = body_pose_aa_tp   # [T, B, 69]
        
        # Input 2. Keypoints (H)
        if 'h_kp3d_coco' in data :
            h_kp3d = data['h_kp3d_coco']
            h_kp3d = h_kp3d.reshape(h_kp3d.shape[:2] + (-1,))                     # 
            h_kp3d_tp = h_kp3d.transpose(0, 1).contiguous()                       # [T, B, 51]
            h_vel_kp3d_tp = (h_kp3d_tp[1:] - h_kp3d_tp[:-1])
            h_vel_kp3d_tp = torch.cat([torch.zeros_like(h_vel_kp3d_tp[:1]), h_vel_kp3d_tp])
            h_vel_kp3d_tp = h_vel_kp3d_tp.reshape(h_vel_kp3d_tp.shape[:2] + (-1,)) 

            batch['h_kp3d_tp'] = h_kp3d_tp
            batch['h_vel_kp3d_tp'] = h_vel_kp3d_tp

        # Input 3. Trans, Rotation (W)
        if 'w_transl' in data :
            w_orient_rotmat = data['w_root_orient']                                     # [B, T, 1, 3, 3]
            w_transl_tp = data['w_transl'].unsqueeze(-2).transpose(0, 1).contiguous()   # [T, B, 1, 3]

            w_orient_q_tp = transforms.matrix_to_quaternion(w_orient_rotmat).transpose(0, 1).contiguous()      # [T, B, 1, 4]
            w_orient_6d_tp = transforms.matrix_to_rotation_6d(w_orient_rotmat).transpose(0, 1).contiguous()    # [T, B, 1, 3]
            
            local_traj_tp = traj_global2local_heading(w_transl_tp, w_orient_q_tp, local_orient_type=self.local_orient_type)  # [T, B, 1, 11]

            batch['w_orient_6d_tp'] = w_orient_6d_tp
            batch['orient_q_tp'] = w_orient_q_tp
            batch['trans_tp'] = w_transl_tp
            batch['w_orient_q_tp'] = w_orient_q_tp
            batch['w_transl_tp'] = w_transl_tp
            batch['local_traj_tp'] = local_traj_tp

        B, T = batch['body_pose'].shape[:2]
        batch['batch_size'] = B
        batch['seqlen'] = T
        
        return batch
    
    def init_batch_stage2(self, batch):
        B, T = batch['batch_size'], batch['seqlen']
        rotmat = batch['body_pose'].reshape(B*T, -1, 3, 3)
        betas = batch['betas'].reshape(B*T, 10)

        coarse_root_orient = batch['coarse_out_orient'].reshape(B*T, -1, 3, 3)     # [B, T, 3, 3]
        coarse_transl = batch['coarse_out_trans'].reshape(B*T, 3)                  # [B, T, 3]
        w_root = batch['w_root_orient'].reshape(B*T, -1, 3, 3)
        w_transl = batch['w_transl'].reshape(B*T, 3)

        coarse_output = self.smpl(body_pose=rotmat, global_orient=coarse_root_orient, 
                             transl=coarse_transl, betas=betas, pose2rot=False)  # World coordinate
        
        w_output = self.smpl(body_pose=rotmat, global_orient=w_root, 
                             transl=w_transl, betas=betas, pose2rot=False)  # World coordinate
        
        ######  Return (World2Camera)
        w_vertex = w_output.vertices.reshape(B, T, -1, 3)
        w_kp3d_wham = torch.matmul(self.J_regressor_wham, w_vertex)         
        w_kp3d_coco = w_kp3d_wham[..., :17, :3]
        w_kp3d_feet = w_kp3d_coco[..., [15, 16], :3]
        contact = compute_contact_label(w_kp3d_feet)
        batch['w_kp3d_coco'] = w_kp3d_coco
        batch['contact'] = contact
        batch['contact_tp'] = contact.transpose(0, 1).contiguous()
        
        w_offset = w_kp3d_coco[..., [11, 12], :3].mean(dim=-2)    # [B, T, 3]
        w_transl = batch['w_transl']
        w_transl = w_transl - w_offset
        w_transl = w_transl - w_transl[:, :1]
        w_kp3d_coco = w_kp3d_coco - w_offset.unsqueeze(-2)

        c_transl, c_orinet = get_virtual_camera(batch['R'], batch['T'], w_transl, batch['w_root_orient'])
        c_kp3d_coco = World2Camera(w_kp3d_coco, batch['R'], c_transl)
        c_kp3d_coco_tp = c_kp3d_coco.transpose(0, 1).contiguous()
        c_vel_kp3d_coco, c_vel_kp3d_coco_tp = compute_vel(c_kp3d_coco)

        batch['c_transl'] = c_transl
        batch['c_root_orient'] = c_orinet
        batch['c_kp3d_coco'] = c_kp3d_coco
        batch['c_kp3d_coco_tp'] = c_kp3d_coco_tp.reshape(T, B, -1)
        batch['c_vel_kp3d_coco_tp'] = c_vel_kp3d_coco_tp.reshape(T, B, -1)

        #### Return (World2Camera (Stage1))
        w_vertex = coarse_output.vertices.reshape(B, T, -1, 3)
        w_kp3d_wham = torch.matmul(self.J_regressor_wham, w_vertex)         
        w_kp3d_coco = w_kp3d_wham[..., :17, :3]
        batch['coarse_w_kp3d_coco'] = w_kp3d_coco

        w_offset = w_kp3d_coco[..., [11, 12], :3].mean(dim=-2)    # [B, T, 3]
        w_transl = batch['coarse_out_trans'][..., 0, :3]          # [B, T, 1, 3]
        w_transl = w_transl - w_offset
        w_transl = w_transl - w_transl[:, :1]
        w_kp3d_coco = w_kp3d_coco - w_offset.unsqueeze(-2)

        c_transl, c_orinet = get_virtual_camera(batch['R'], batch['T'], w_transl, batch['coarse_out_orient'])
        c_kp3d_coco = World2Camera(w_kp3d_coco, batch['R'], c_transl)
        c_kp3d_coco_tp = c_kp3d_coco.transpose(0, 1).contiguous()
        c_vel_kp3d_coco, c_vel_kp3d_coco_tp = compute_vel(c_kp3d_coco)
        
        batch['coarse_c_transl'] = c_transl
        batch['coarse_c_root_orient'] = c_orinet
        batch['coarse_c_kp3d_coco'] = c_kp3d_coco
        batch['coarse_c_kp3d_coco_tp'] = c_kp3d_coco_tp.reshape(T, B, -1)
        batch['coarse_c_vel_kp3d_coco_tp'] = c_vel_kp3d_coco_tp.reshape(T, B, -1)

        return batch

    def forward_smpl(self, batch):
        ## LBS (World coordinate)
        B, T = batch['body_pose'].shape[:2]
        
        w_root = batch['w_root_orient']
        w_transl = batch['w_transl']
        #w_transl = w_transl - w_transl[:, :1]

        # 
        rotmat = batch['body_pose'].reshape(B*T, -1, 3, 3)
        betas = batch['betas'].reshape(B*T, 10)
        h_root = torch.zeros_like(w_root).reshape(B*T, -1, 3, 3)
        h_transl = torch.zeros_like(w_transl).reshape(B*T, 3)

        h_output = self.smpl(body_pose=rotmat, global_orient=h_root, 
                             transl=h_transl, betas=betas, pose2rot=False)  # Human-centric
        
        # Return (Human-centric)
        h_vertex = h_output.vertices.reshape(B, T, -1, 3)
        h_kp3d_wham = torch.matmul(self.J_regressor_wham, h_vertex)         # [B, T, 31, 3]
        h_kp3d_coco = h_kp3d_wham[..., :17, :3]
        
        batch['h_kp3d_coco'] = h_kp3d_coco
        batch['w_transl'] = w_transl

        return batch

    def forward_infer(self, batch):
        batch = self.forward_smpl(batch)
        batch = self.init_batch_stage1(batch)

        if self.stage == 'stage1':
            self.corsenet.forward_infer(batch)
        elif self.stage == 'stage2':
            self.corsenet.forward_infer(batch)
            self.init_batch_stage2(batch)
            self.finenet.forward_infer(batch)

        return batch

    def forward(self, batch):
        batch = self.forward_smpl(batch)
        batch = self.init_batch_stage1(batch)
        if self.stage == 'stage1':
            self.corsenet(batch)
        elif self.stage == 'stage2':
            self.corsenet(batch)
            batch = self.init_batch_stage2(batch)
            from lib.utils.print_utils import print_batch
            print_batch(batch)
            self.finenet(batch)
        
        return batch