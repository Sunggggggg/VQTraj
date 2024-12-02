import torch
import torch.nn as nn
from configs import constants as _C
from lib.utils import transforms
from lib.utils.print_utils import count_param
from lib.utils.traj_utils import traj_global2local_heading, traj_local2global_heading
from lib.models import build_body_model, Encoder, Decoder, QuantizeEMAReset, OutHead
from lib.models.layers import CrossAtten
from lib.models.MotionCLIP import get_model


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
    def __init__(self, cfg) :
        super().__init__()
        num_joints = 17
        # input_dict
        self.input_dict = {
            "body_pose_6d_tp": 23*6,
            # "w_orient_6d_tp": 6,
            "w_kp3d_tp": num_joints*3,
            "w_vel_kp3d_tp": num_joints*3,
        }
        input_dim = sum([v for v in self.input_dict.values()])

        self.local_orient_type = '6d'

        smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
        self.smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
        self.clip = get_model()
        self.cross_att = CrossAtten()

        depth = 3
        down_t = 2
        num_tokens = 64
        #down_t = 3
        # 
        self.encoder = Encoder(num_tokens=num_tokens, input_emb_width=input_dim, output_emb_width = 512, down_t = down_t,
                 stride_t = 2, width = 512, depth = depth, dilation_growth_rate = 3, activation='relu', norm=None)
        self.decoder = Decoder(num_tokens=num_tokens, input_emb_width = 256, output_emb_width = 512, down_t = down_t, 
                               stride_t = 2, width = 512, depth = depth, dilation_growth_rate = 3, activation='relu', norm=None)
        
        self.codebook = QuantizeEMAReset(nb_code=512, code_dim=512)
        self.head = OutHead(in_dim=256, hid_dims=[256, 128], out_dim=11)

        print(f">> Input dimension : {input_dim}")
        print(f"# of weight : {count_param(self)}")

    def init_batch(self, batch):
        data = batch.copy()
        if 'body_pose' in data :
            body_pose_tp = batch['body_pose'].transpose(0, 1).contiguous()
            body_pose_6d_tp = transforms.matrix_to_rotation_6d(body_pose_tp)
            batch['body_pose_6d_tp'] = body_pose_6d_tp.reshape(body_pose_6d_tp.shape[:2] + (-1,))
        
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
        
        if 'w_kp3d' in data :
            w_kp3d = data['w_kp3d']
            w_kp3d_tp = w_kp3d.transpose(0, 1).contiguous()
            w_vel_kp3d_tp = (w_kp3d_tp[1:] - w_kp3d_tp[:-1]) * 30 # FPS
            w_vel_kp3d_tp = torch.cat([torch.zeros_like(w_vel_kp3d_tp[:1]), w_vel_kp3d_tp], dim=0)    # [B, T, J, 3]

            batch['w_kp3d_tp'] = w_kp3d_tp.reshape(w_kp3d_tp.shape[:2] + (-1,))
            batch['w_vel_kp3d_tp'] = w_vel_kp3d_tp.reshape(w_kp3d_tp.shape[:2] + (-1,))

        B, T = batch['body_pose'].shape[:2]
        batch['batch_size'] = B
        batch['seqlen'] = T

        input_batch = torch.cat([batch[k].reshape(T, B, -1) for k in self.input_dict], dim=-1)    # [T, B, dim]
        batch['input_tp'] = input_batch
        
        return batch

    def post_processing(self, batch):
        d_xy = batch['d_xy']
        z = batch['z']
        local_orient = batch['local_orient']
        d_heading_vec = batch['d_heading_vec']
        
        init_xy = torch.zeros_like(batch['local_traj_tp'][:1, ..., :2])    # [1, B, 2]
        init_heading_vec = torch.tensor([0., 1.], device=init_xy.device).expand_as(batch['local_traj_tp'][:1, ..., -2:])
        d_xy = torch.cat([init_xy, d_xy[1:, ..., :2]], dim=0)                              # [T, B, 2]
        d_heading_vec = torch.cat([init_heading_vec, d_heading_vec[1:, ..., -2:]], dim=0)  # [T, B, 2]

        out_local_traj_tp = torch.cat([d_xy, z, local_orient, d_heading_vec], dim=-1)       # [T, B, 11]
        out_trans_tp, out_orient_q = traj_local2global_heading(out_local_traj_tp, local_orient_type=self.local_orient_type, )
        
        batch['out_trans_tp'] = out_trans_tp        # GT w_transl
        batch['out_orient_q_tp'] = out_orient_q     # w_orient_q_tp
        out_orient_tp = transforms.quaternion_to_matrix(out_orient_q)
        batch['out_orient_6d_tp'] = transforms.matrix_to_rotation_6d(out_orient_tp)
        return batch

    def forward_clip(self, batch):
        batch = self.clip(batch)
        return batch

    def forward_smpl(self, batch):
        ## LBS (World coordinate) input : rot_6d
        B, T = batch['body_pose'].shape[:2]

        rotmat = batch['body_pose'].reshape(B*T, -1, 3, 3)
        root = batch['w_root_orient'].reshape(B*T, -1, 3, 3)
        betas = batch['betas'].reshape(B*T, 10)

        output = self.smpl.get_output(body_pose=rotmat,
                             global_orient=root,
                             betas=betas,
                             pose2rot=False)
        output.offset = output.offset.reshape(B, T, 3)  # 
        output.joints = output.joints.reshape(B, T, -1, 3)
        output.feet = output.feet.reshape(B, T, -1, 3)
        
        batch['w_transl'] = batch['w_transl'] - output.offset
        batch['w_transl'] = batch['w_transl'] - batch['w_transl'][:, :1]    # [B, T, 3]
        batch['w_transl'] = batch['w_transl'].unsqueeze(-2)
        batch['w_kp3d'] = output.joints[..., :17, :3]                       # root align
        batch['coco'] = batch['w_kp3d'].permute(0, 2, 3, 1)
        
        batch['w_feet'] = output.feet + batch['w_transl']                   # [B, T, 1, 3]

        batch['contact'] = compute_contact_label(batch['w_feet'])
        return batch

    def forward_model(self, batch):
        batch = self.encoder(batch)
        x_d, commit_loss, perplexity = self.codebook(batch['encoded_feat'])  # [B, dim, T]
        # batch['quantized_feat'] = x_d
        batch['commit_loss'] = commit_loss

        batch['quantized_feat'] = self.cross_att(batch['z'], x_d)
        batch = self.decoder(batch)                                         # [B, dim, T]
        batch = self.head(batch, True)
        
        return batch
    
    def forward(self, batch):
        batch = self.forward_smpl(batch)
        batch = self.forward_clip(batch)
        batch = self.init_batch(batch)
        batch = self.forward_model(batch)
        batch = self.post_processing(batch)
        return batch