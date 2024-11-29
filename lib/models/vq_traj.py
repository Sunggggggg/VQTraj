import torch
import torch.nn as nn
from configs import constants as _C
from lib.utils import transforms
from lib.utils.traj_utils import traj_global2local_heading
from lib.models import build_body_model, Encoder, Decoder, QuantizeEMAReset, OutHead

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
            "body_pose_6d_tp": 6,
            "w_orient_6d_tq": 6,
            "w_kp3d_tp": num_joints*3,
            "w_vel_kp3d_tp": num_joints*3,
        }
        input_dim = sum([v for v in self.input_dict.values()])

        self.local_orient_type = '6d'

        smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
        self.smpl = build_body_model(cfg.DEVICE, smpl_batch_size)

        # 
        self.encoder = Encoder(input_emb_width=input_dim, output_emb_width = 512, down_t = 3,
                 stride_t = 2, width = 512, depth = 3, dilation_growth_rate = 3, activation='relu', norm=None)
        self.decoder = Decoder(input_emb_width = input_dim, output_emb_width = 512, down_t = 3, 
                               stride_t = 2, width = 512, depth = 3, dilation_growth_rate = 3, activation='relu', norm=None)
        
        self.codebook = QuantizeEMAReset(nb_code=512, code_dim=512)
        self.head = OutHead(in_dim=512,  hid_dims=[512, 128], out_dim=11)


    def init_batch(self, batch):
        data = batch.copy()
        if 'body_pose' in data :
            body_pose_tp = batch['body_pose'].transpose(0, 1).contiguous()
            body_pose_6d_tp = transforms.matrix_to_rotation_6d(body_pose_tp)
            batch['body_pose_6d_tp'] = body_pose_6d_tp
        
        if 'w_transl' in data :
            w_orient_rotmat = data['w_root_orient'] # [B, T, 1, 3, 3]
            w_transl = data['w_transl']             # [B, T, 3]

            w_orinet_q = transforms.matrix_to_quaternion(w_orient_rotmat)
            w_orient_6d = transforms.matrix_to_rotation_6d(w_orient_rotmat)
            
            w_orinet_q_tq = w_orinet_q.transpose(0, 1).contiguous()
            w_transl_tq = w_transl.transpose(0, 1).contiguous()
            w_orient_6d_tq = w_orient_6d.transpose(0, 1).contiguous()
            local_traj_tp = traj_global2local_heading(w_transl_tq, w_orinet_q_tq, local_orient_type=self.local_orient_type)  # [T, B, 11]

            batch['w_orient_6d_tq'] = w_orient_6d_tq
            batch['w_orinet_q_tq'] = w_orinet_q_tq
            batch['w_transl_tq'] = w_transl_tq
            batch['local_traj_tp'] = local_traj_tp
        
        if 'w_kp3d' in data :
            w_kp3d = data['w_kp3d']
            w_kp3d_tp = w_kp3d.transpose(0, 1).contiguous()
            w_vel_kp3d_tp = (w_kp3d_tp[1:] - w_kp3d_tp[:-1]) * 30 # FPS
            w_vel_kp3d_tp = torch.cat([torch.zeros_like[w_vel_kp3d_tp[:, :1]], w_vel_kp3d_tp], dim=1)    # [B, T, J, 3]

            batch['w_kp3d_tp'] = w_kp3d_tp.reshape(w_kp3d_tp.shape[:-2] + (-1,))
            batch['w_vel_kp3d_tp'] = w_vel_kp3d_tp.reshape(w_kp3d_tp.shape[:-2] + (-1,))

        input_batch = torch.cat([batch[k] for k in self.input_dict], dim=-1)    # [T, B, dim]
        
        batch['input'] = input_batch
        batch['batch_size'] = batch['body_pose'].shape[0]
        batch['seqlen'] = batch['body_pose'].shape[1]
    
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
        batch['w_kp3d'] = output.joints                                     # root align
        batch['w_feet'] = output.feet + batch['w_transl'].unsqueeze(-2)     # [B, T, 1, 3]

        batch['contact'] = compute_contact_label(batch['w_feet'])
        return batch

    def forward_model(self, batch):
        batch = self.encoder(batch)
        batch['quantized_feat'] = self.codebook(batch)
        batch = self.decoder(batch)
        batch = self.head(batch)
        
        return batch
    
    def forward(self, batch):
        batch = self.init_batch(batch)
        batch = self.forward_smpl(batch)
        batch = self.forward_model(batch)


        return batch