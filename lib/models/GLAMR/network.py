import numpy as np
import torch
import torch.nn as nn
from configs import constants as _C
from lib.utils import transforms
from lib.utils.print_utils import count_param
from lib.utils.traj_utils import traj_global2local_heading, traj_local2global_heading
from smplx import SMPL
from .traj_vae import *

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


class Network(nn.Module):
    def __init__(self, ) :
        super().__init__()
        num_joints = 17
        # input_dict
        self.input_dict = {
            "body_pose_aa_tp": 23*3,
            # "w_orient_6d_tp": 6,
            "c_kp3d_tp": num_joints*3,
            "c_vel_kp3d_tp": num_joints*3,
        }
        input_dim = sum([v for v in self.input_dict.values()])

        self.local_orient_type = '6d'

        self.smpl = SMPL(_C.BMODEL.FLDR, num_betas=10, ext='pkl')
        J_regressor_wham = np.load(_C.BMODEL.JOINTS_REGRESSOR_WHAM)
        self.register_buffer('J_regressor_wham', torch.tensor(
            J_regressor_wham, dtype=torch.float32))
        
        # Model
        self.context_encoder = ContextEncoder()
        self.data_encoder = DataEncoder()
        self.data_decoder = DataDecoder()
    
        print(f">> Input dimension : {input_dim}")
        print(f"# of weight : {count_param(self)}")

    def init_batch(self, batch):
        data = batch.copy()
        # Input 1. Body pose
        if 'body_pose' in data :
            body_pose_tp = batch['body_pose'].transpose(0, 1).contiguous()
            body_pose_6d_tp = transforms.matrix_to_rotation_6d(body_pose_tp)
            body_pose_aa_tp = transforms.matrix_to_axis_angle(body_pose_tp)

            batch['body_pose_6d_tp'] = body_pose_6d_tp.reshape(body_pose_6d_tp.shape[:2] + (-1,))   # [T, B, 69]
            batch['body_pose_aa_tp'] = body_pose_aa_tp.reshape(body_pose_aa_tp.shape[:2] + (-1,))   # [T, B, 69]

        # Input 2. Keypoints (C)
        if 'c_kp3d' in data :
            c_kp3d = data['c_kp3d']
            c_kp3d_tp = c_kp3d.transpose(0, 1).contiguous()
            c_vel_kp3d_tp = (c_kp3d_tp[1:] - c_kp3d_tp[:-1])
            c_vel_kp3d_tp = torch.cat([torch.zeros_like(c_vel_kp3d_tp[:1]), c_vel_kp3d_tp], dim=0) 

            batch['c_kp3d_tp'] = c_kp3d_tp.reshape(c_kp3d_tp.shape[:2] + (-1,))                     # [T, B, 51]
            batch['c_vel_kp3d_tp'] = c_vel_kp3d_tp.reshape(c_vel_kp3d_tp.shape[:2] + (-1,))         #  ''
            batch['in_joint_pos_tp'] = c_kp3d_tp.reshape(c_kp3d_tp.shape[:2] + (-1,))               # [T, B, 51]

        # Input 3. Trans, Rotation
        if 'w_transl' in data :
            w_orient_rotmat = data['w_root_orient']         # [B, T, 1, 3, 3]
            w_transl = data['w_transl'].unsqueeze(-2)       # [B, T, 1, 3]

            w_orient_q = transforms.matrix_to_quaternion(w_orient_rotmat)   # [B, T, 1, 4]
            w_orient_6d = transforms.matrix_to_rotation_6d(w_orient_rotmat) # [B, T, 1, 6]
            
            w_orient_q_tp = w_orient_q.transpose(0, 1).contiguous()         # [T, B, 1, 4]
            w_transl_tp = w_transl.transpose(0, 1).contiguous()             # [T, B, 1, 3]
            w_orient_6d_tp = w_orient_6d.transpose(0, 1).contiguous()       # [T, B, 1, 6]
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

        input_batch = torch.cat([batch[k].reshape(T, B, -1) for k in self.input_dict], dim=-1)    # [T, B, dim]
        batch['input_tp'] = input_batch
        
        return batch

    def forward_smpl(self, batch):
        ## LBS (World coordinate)
        B, T = batch['body_pose'].shape[:2]

        rotmat = batch['body_pose'].reshape(B*T, -1, 3, 3)
        betas = batch['betas'].reshape(B*T, 10)

        w_root = batch['w_root_orient'].reshape(B*T, -1, 3, 3)
        w_transl = batch['w_transl'].reshape(B*T, 3)

        c_root = torch.zeros_like(w_root)
        c_transl = torch.zeros_like(w_transl)
        
        c_output = self.smpl(body_pose=rotmat, global_orient=c_root, 
                             transl=c_transl, betas=betas, pose2rot=False)
        
        # Return
        c_vertex = c_output.vertices.reshape(B, T, -1, 3)
        c_kp3d_wham = torch.matmul(self.J_regressor_wham, c_vertex)         # [B, T, 31, 3]
        c_kp3d_coco = c_kp3d_wham[..., :17, :3]
        #c_kp3d_feet = c_kp3d_wham[..., [15, 16], :3]

        batch['c_root_orient'] = c_root
        batch['c_transl'] = c_transl
        batch['c_kp3d'] = c_kp3d_coco
        #batch['c_feet'] = c_kp3d_feet
        #batch['contact'] = contact

        return batch

    def forward_model(self, data):
        self.context_encoder(data)
        self.data_encoder(data)
        self.data_decoder(data, mode='train')
        return data
    
    def forward(self, batch):
        batch = self.forward_smpl(batch)
        batch = self.init_batch(batch)
        batch = self.forward_model(batch)
        return batch