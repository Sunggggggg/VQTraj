import numpy as np
import torch
import torch.nn as nn
from lib.models.TransVQTraj import config as _C
from configs import constants
from lib.utils.print_utils import count_param
from ..codebook import QuantizeEMAReset
from ..layers import ContextEncoder, TrajEncoder, TrajDecoder
from .rotation_conversions import *
from lib.utils import transforms
from lib.utils.traj_utils import traj_global2local_heading, traj_local2global_heading
from smplx import SMPL

def model_freeze(model):
    for param in model.parameters() :
        param.requires_grad = False
    return model

class TransNetwork(nn.Module):
    def __init__(self, cfg, codebook_train=True) :
        super().__init__()
        num_joints = 17
        self.local_orient_type = '6d'

        # Body model
        self.smpl= SMPL(constants.BMODEL.FLDR, num_betas=10, ext='pkl')
        J_regressor_wham = np.load(constants.BMODEL.JOINTS_REGRESSOR_WHAM)
        self.register_buffer('J_regressor_wham', torch.tensor(
            J_regressor_wham, dtype=torch.float32))
        
        # Model
        self.context_encoder = ContextEncoder(hid_dim=512, out_dim=512)
        self.encoder = TrajEncoder(con_dim=512, hid_dim=256, out_dim=_C.CODEBOOK.code_dim)
        self.decoder = TrajDecoder(in_dim=_C.CODEBOOK.code_dim, hid_dim=256, num_tokens=self.encoder.token_num)
        self.codebook = QuantizeEMAReset(nb_code=_C.CODEBOOK.nb_code, code_dim=_C.CODEBOOK.code_dim)

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
            batch['w_orient_q_tp'] = w_orient_q_tp
            batch['w_transl_tp'] = w_transl_tp
            batch['local_traj_tp'] = local_traj_tp

        B, T = batch['body_pose'].shape[:2]
        batch['batch_size'] = B
        batch['seqlen'] = T

        #input_batch = torch.cat([batch[k].reshape(T, B, -1) for k in self.input_dict], dim=-1)    # [T, B, dim]
        #batch['input_tp'] = input_batch
        
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

        batch['c_root_orient'] = c_root
        batch['c_transl'] = c_transl
        batch['c_kp3d'] = c_kp3d_coco

        return batch
    
    def forward_model(self, batch):
        """
        batch : [T, B, dim]
        """
        batch = self.context_encoder(batch)
        batch = self.encoder(batch)
        x_quantized, commit_loss, _ = self.codebook(batch['encoded_feat'])  # [T-1, B, dim, N]
        batch['quantized_feat'] = x_quantized
        batch['commit_loss'] = commit_loss

        batch = self.decoder(batch)
        return batch


    def forward(self, batch):
        batch = self.forward_smpl(batch)
        batch = self.init_batch(batch)      # input_tp : [T, B, dim]
        batch = self.forward_model(batch)

        batch['w_transl_tp'] = batch['w_transl_tp'][1:]
        batch['w_orient_q_tp'] = batch['w_orient_q_tp'][1:]
        batch['out_orient_6d_tp'] = batch['out_orient_6d_tp'][1:]
        
        return batch
    
    def forward_classifier(self, x_past):
        """
        """
        cls_logits_softmax = self.classifier(x_past)                       # [B, N, C]
        decode_feat = self.codebook.dequantize_logits(cls_logits_softmax)  # [B, code_dim, N]
        decode_feat = decode_feat.permute(0, 2, 1)

        x_past_embed = self.joint_embedding(x_past)
        trans_dict = self.decoder(decode_feat, x_past_embed)      
        x_trans_feat = trans_dict['x_trans'].reshape(-1, 1, self.hid_dim)
        x_trans_kp3d = trans_dict['kp3d'].reshape(-1, 17, 3)

        x_curr = x_past.reshape(-1, 17, 3) + x_trans_kp3d
        return x_trans_feat, x_curr