import numpy as np
import torch
import torch.nn as nn
from lib.models.TransVQTraj import config as _C
from configs import constants
from ..codebook import QuantizeEMAReset
from ..layers import Encoder_v2, Decoder_v2
from ..head import OutHead
from .classifier import TokenClassifier
from .rotation_conversions import *
from lib.utils import transforms
from lib.utils.traj_utils import traj_global2local_heading, traj_local2global_heading

from smplx import SMPL

data_info = {
    'joint':3,
    'vel_joint': 3,
    'body_pose': 6,
    'root_pose': 6,
    'root_transl': 3,
}

def model_freeze(model):
    for param in model.parameters() :
        param.requires_grad = False
    return model

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

def rot_output(past_6d, trans_6d):
    B = past_6d.shape[0]
    past = transforms.rotation_6d_to_matrix(past_6d.reshape(B, 1, 6))        # [B, 1, 3, 3]
    trans = transforms.rotation_6d_to_matrix(trans_6d.reshape(B, 1, 6))      # [B, 1, 3, 3]
    curr = torch.matmul(trans, past)

    return transforms.matrix_to_rotation_6d(curr).reshape(B, 1, 6)

class TransNetwork(nn.Module):
    def __init__(self, cfg, codebook_train=True) :
        super().__init__()
        num_joints = 17
        # input_dict
        self.input_dict = {
            "body_pose_aa_tp": 23*3,
            # "w_orient_6d_tp": 6,
            "w_kp3d_tp": num_joints*3,
            "w_vel_kp3d_tp": num_joints*3,
        }
        in_dim = sum([v for v in self.input_dict.values()])
        #in_dim = num_joints*3 + num_joints*3 + 23*3

        token_num = 20
        token_inter_dim = 64
        hid_dim = self.hid_dim = 256
        hid_inter_dim = 256
        num_blocks = 4
        cal_token_num = token_num * 2**(3-2)
        self.local_orient_type = '6d'

        self.smpl= SMPL(constants.BMODEL.FLDR, num_betas=10, ext='pkl')
        J_regressor_wham = np.load(constants.BMODEL.JOINTS_REGRESSOR_WHAM)
        self.register_buffer('J_regressor_wham', torch.tensor(
            J_regressor_wham, dtype=torch.float32))

        self.encoder = Encoder_v2(in_dim=in_dim,
                 hid_dim=hid_dim,
                 out_dim=_C.CODEBOOK.code_dim,
                 token_num=token_num,
                 up_sample_rate=3,
                 down_sample_rate=2,
                 res_depth=3,
                 dilation_growth_rate=3)
        
        self.codebook = QuantizeEMAReset(nb_code=_C.CODEBOOK.nb_code, code_dim=_C.CODEBOOK.code_dim)
        self.decoder = Decoder_v2(in_dim=_C.CODEBOOK.code_dim,
                 hid_dim=hid_dim,
                 num_tokens=cal_token_num,
                 div_rate=2,
                 down_sample_rate=2,
                 res_depth=3,
                 dilation_growth_rate=3,)
        
        self.head = OutHead(hid_dim,  hid_dims=[512, 128], out_dim=11)
    
        if not codebook_train :
            self.joint_embedding = nn.Linear(17*3, in_dim)
            self.classifier = TokenClassifier(num_blocks=4,
                 in_dim=51,
                 hid_dim=64,
                 hid_inter_dim=256,
                 token_num=cal_token_num,
                 token_inter_dim=64,
                 class_num=_C.CODEBOOK.nb_code)
            self.codebook = model_freeze(self.codebook)
            self.decoder = model_freeze(self.decoder)
    
    def init_batch(self, batch):
        data = batch.copy()
        if 'body_pose' in data :
            body_pose_tp = batch['body_pose'].transpose(0, 1).contiguous()
            body_pose_6d_tp = transforms.matrix_to_rotation_6d(body_pose_tp)
            body_pose_aa_tp = transforms.matrix_to_axis_angle(body_pose_tp)
            
            batch['body_pose_aa_tp'] = body_pose_aa_tp.reshape(body_pose_6d_tp.shape[:2] + (-1,))   # [T, B, 69]
            batch['body_pose_6d_tp'] = body_pose_6d_tp.reshape(body_pose_6d_tp.shape[:2] + (-1,))   # [T, B, 138]
        
        if 'w_kp3d' in data :
            w_kp3d = data['w_kp3d']
            w_kp3d_tp = w_kp3d.transpose(0, 1).contiguous()
            w_vel_kp3d_tp = (w_kp3d_tp[1:] - w_kp3d_tp[:-1]) # FPS
            w_vel_kp3d_tp = torch.cat([torch.zeros_like(w_vel_kp3d_tp[:1]), w_vel_kp3d_tp], dim=0)    # [B, T, J, 3]

            batch['w_kp3d_tp'] = w_kp3d_tp.reshape(w_kp3d_tp.shape[:2] + (-1,))         # [T, B, 51]
            batch['w_vel_kp3d_tp'] = w_vel_kp3d_tp.reshape(w_kp3d_tp.shape[:2] + (-1,)) # [T, B, 51]

        # For GT
        if 'w_transl' in data :
            w_orient_rotmat = data['w_root_orient'] # [B, T, 1, 3, 3]
            w_transl = data['w_transl']             # [B, T, 1, 3]

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

        input_batch = torch.cat([batch[k].reshape(T, B, -1) for k in self.input_dict], dim=-1)    # [T, B, dim]
        batch['input_tp'] = input_batch
        
        return batch

    def forward_smpl(self, batch):
        ## LBS (World coordinate) input : rot_6d
        B, T = batch['body_pose'].shape[:2]

        rotmat = batch['body_pose'].reshape(B*T, -1, 3, 3)
        root = batch['w_root_orient'].reshape(B*T, -1, 3, 3)
        betas = batch['betas'].reshape(B*T, 10)
        transl = batch['w_transl'].reshape(B*T, 3)

        output = self.smpl(body_pose=rotmat,
                             global_orient=root,
                             betas=betas,
                             transl=transl,
                             pose2rot=False
                            )
        
        w_kp3d = torch.matmul(self.J_regressor_wham, output.vertices)
        batch['w_transl'] = batch['w_transl'].unsqueeze(-2)
        batch['w_kp3d'] = w_kp3d[..., :17, :3]                       # root align
        # batch['w_feet'] = output.feet + batch['w_transl']                   # [B, T, 1, 3]

        # batch['contact'] = compute_contact_label(batch['w_feet'])
        return batch

    def forward_model(self, batch):
        """
        batch : [T, B, dim]
        """
        B, T = batch['body_pose'].shape[:2]
        out_local_traj_tp_list, commit_loss_list = [],[]
        for t in range(T):
            if t == 0 :
                out_local_traj_tp = torch.zeros_like(batch['local_traj_tp'][:1])    # [1, B, 11]
                
                init_xy = torch.zeros_like(out_local_traj_tp[:1, ..., :2])    # [1, B, 2]
                init_heading_vec = torch.tensor([0., 1.], device=init_xy.device).expand_as(out_local_traj_tp[:1, ..., -2:])
                out_local_traj_tp[..., :2] = init_xy
                out_local_traj_tp[..., 9:] = init_heading_vec
            else :
                x_curr, x_past = batch['input_tp'][t], batch['input_tp'][t-1]
                local_traj = batch['local_traj_tp'][t-1]
                past_z = local_traj[..., 2:3]
                past_local_orient = local_traj[..., 3:9]

                x_enc = self.encoder(x_curr, x_past)                # [B, code_dim, N]

                x_quantized, commit_loss, _ = self.codebook(x_enc)  # [B, code_dim, N]
                decoded_feat = self.decoder(x_quantized)
                x_trans = self.head({'decoded_feat':decoded_feat})  # [B, 1, 11]
                
                out_local_traj_tp = torch.zeros_like(local_traj.unsqueeze(0))
                out_local_traj_tp[..., :2] = x_trans[..., :2]
                out_local_traj_tp[..., 2:3] = past_z + x_trans[..., 2:3]
                out_local_traj_tp[..., 3:9] = rot_output(past_local_orient, x_trans[..., 3:9])
                out_local_traj_tp[..., 9:] = x_trans[..., 9:]
                
                commit_loss_list.append(commit_loss)
            out_local_traj_tp_list.append(out_local_traj_tp)                        # [B, 1, 11]
        
        out_local_traj_tp = torch.cat(out_local_traj_tp_list, dim=0)              # [T, B, 1, 11]
        out_local_traj = out_local_traj_tp.transpose(0, 1).contiguous()
        out_trans_tp, out_orient_q = traj_local2global_heading(out_local_traj_tp, local_orient_type=self.local_orient_type, )
        
        batch['out_trans_tp'] = out_trans_tp        # GT w_transl
        batch['out_orient_q_tp'] = out_orient_q     # w_orient_q_tp
        out_orient_tp = transforms.quaternion_to_matrix(out_orient_q)
        batch['out_orient_6d_tp'] = transforms.matrix_to_rotation_6d(out_orient_tp)
        batch['local_orient'] = out_local_traj[..., 3:9]
        batch['d_heading_vec'] = out_local_traj[..., 9:]

        batch['commit_loss'] = sum(commit_loss_list)
        return batch

    def forward(self, batch):
        batch = self.forward_smpl(batch)
        batch = self.init_batch(batch)      # input_tp : [T, B, dim]
        batch = self.forward_model(batch)
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