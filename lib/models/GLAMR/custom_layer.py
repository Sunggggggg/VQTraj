import torch
import numpy as np
import torch.nn as nn
from ..head import OutHead
from ..rnn import RNN
from ..layers import MLPBlock, FCBlock
from ..resnet import Resnet1D   
from lib.utils import transforms
from lib.utils.traj_utils import convert_traj_world2heading, traj_local2global_heading


class ContextEncoder(nn.Module):
    def __init__(self, hid_dim, out_dim) :
        super().__init__()
        coco_num_joints = 17
        smpl_num_joints = 23
        num_layers = 2
        self.input_dict = {'c_kp3d_tp': coco_num_joints*3, 
                           #'c_vel_kp3d_tp': coco_num_joints*3,
                           #'body_pose_aa_tp': smpl_num_joints*3
                           }
        in_dim = sum(v for v in self.input_dict.values())
        print(f">> Context Encoder in_dim : {in_dim}")
        """ Input mlp"""
        self.input_layer = MLPBlock(in_dim, hid_dim, hid_dim)

        """ Temporal """
        self.temporal_net = nn.ModuleList()
        cur_dim = hid_dim
        for _ in range(num_layers):
            net = RNN(cur_dim, hid_dim, 'lstm', bi_dir=True)
            cur_dim = hid_dim
            self.temporal_net.append(net)
        
        """ Output mlp """
        self.output_layer = MLPBlock(cur_dim, hid_dim, out_dim)

    def forward(self, batch):
        """
        x : [T, B, dim]
        """
        x = torch.cat([batch[k] for k in self.input_dict], dim=-1)

        x = self.input_layer(x)
        for net in self.temporal_net :
            x = net(x)
        
        x = self.output_layer(x)
        batch['context'] = x
        return batch
    
class TrajEncoder(nn.Module):
    def __init__(self, con_dim, hid_dim=256, out_dim=256) :
        super().__init__()
        down_sample_rate = 2
        num_layers = 2
        self.input_dict = {#'orient_q': 4, 
                           'orient_aa': 3,
                           'trans': 3}
        in_dim = sum(v for v in self.input_dict.values())

        """ Input mlp"""
        self.input_layer = MLPBlock(in_dim, hid_dim, hid_dim)

        """ Temporal """
        self.temporal_net = nn.ModuleList()
        cur_dim = hid_dim
        for _ in range(num_layers):
            net = RNN(cur_dim, hid_dim, 'lstm', bi_dir=True)
            cur_dim = hid_dim
            self.temporal_net.append(net)

        """ Fusing mlp """
        fusing_layers = []
        layer_list = [hid_dim+con_dim, hid_dim, hid_dim]
        for i in range(1, len(layer_list)) :
            fusing_layers.append(nn.Conv1d(layer_list[i-1], layer_list[i], 3, 1, 1))
            fusing_layers.append(nn.ReLU())
        self.fusing = nn.Sequential(*fusing_layers)

        """ Out """
        output_layer = []
        output_layer.append(nn.Upsample(scale_factor=2))
        output_layer.append(nn.Conv1d(hid_dim, hid_dim, 3, 1, 1))
        output_layer.append(nn.ReLU())

        output_layer.append(nn.Upsample(scale_factor=2))
        output_layer.append(nn.Conv1d(hid_dim, hid_dim, 3, 1, 1))
        output_layer.append(nn.ReLU())

        output_layer.append(nn.Conv1d(hid_dim, hid_dim, 3, 1, 1))
        output_layer.append(nn.ReLU())
        output_layer.append(nn.Conv1d(hid_dim, out_dim, 3, 1, 1))
        output_layer.append(nn.ReLU())
        self.output_layer = nn.Sequential(*output_layer)
        
    def forward(self, batch):
        """
        x : [T, B, ]
        """
        T = batch['seqlen']
        B = batch['batch_size']
        context = batch['context']
        
        """ Input mlp"""
        init_heading_orient, init_heading_trans =\
              convert_traj_world2heading(batch['w_orient_q_tp'], batch['w_transl_tp'])
        init_heading_orient = transforms.quaternion_to_axis_angle(init_heading_orient)
        x = torch.cat([init_heading_trans, init_heading_orient], dim=-1)    # [T, B, 3+3]
        
        x = self.input_layer(x).reshape(T, B, -1)

        """ Temporal """
        for net in self.temporal_net :
            x = net(x)                          # [T, B, dim]

        """ Fusing mlp """
        x = torch.cat([x, context], dim=-1)     # [T, B, dim]
        x = x.permute(1, 2, 0)                  # [B, dim, T]
        x = self.fusing(x).permute(2, 0, 1)     # [T, B, dim]

        """ Output mlp"""
        x = x.permute(1, 2, 0)                  # [B, dim, T]
        x = self.output_layer(x)                # [B, dim, T*4]

        batch['encoded_feat'] = x  
        return batch
    
class TrajDecoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 ) :
        super().__init__()
        div_rate = 2
        down_sample_rate = 2

        """ Decoder """
        decoder_layers = []
        decoder_layers.append(nn.Conv1d(in_dim, hid_dim, 3, 1, 1))
        decoder_layers.append(nn.ReLU())

        for i in range(div_rate):
           decoder_layers.append(nn.Upsample(scale_factor=1/2))
           decoder_layers.append(nn.Conv1d(hid_dim, hid_dim, 3, 1, 1))
           decoder_layers.append(nn.ReLU())
        
        for i in range(down_sample_rate):
            input_dim = hid_dim
            block = nn.Sequential(
                Resnet1D(hid_dim, 2, 3, reverse_dilation=True, activation='relu', norm=False),
                nn.Conv1d(hid_dim, input_dim, 3, 1, 1)
            )
            decoder_layers.append(block)
        self.decoder = nn.Sequential(*decoder_layers)
        self.head = OutHead(hid_dim,  hid_dims=[512, 128], out_dim=11)

    def rot_output(past_6d, trans_6d):
        B = past_6d.shape[0]
        past = transforms.rotation_6d_to_matrix(past_6d.reshape(B, 1, 6))        # [B, 1, 3, 3]
        trans = transforms.rotation_6d_to_matrix(trans_6d.reshape(B, 1, 6))      # [B, 1, 3, 3]
        curr = torch.matmul(trans, past)

        return transforms.matrix_to_rotation_6d(curr).reshape(B, 1, 6)
    
    def forward(self, batch):
        T = batch['seqlen']
        x_quant = batch['quantized_feat']   # [B, dim, T]
    
        x_dec = self.decoder(x_quant).permute(2, 0, 1)  # [B, T, 1, dim]
        out = self.head(x_dec)
        
        d_xy = out[..., :2]
        z = out[..., 2:3]
        local_orient = out[..., 3:9]
        d_heading_vec = out[..., 9:]

        batch['d_xy'] = d_xy
        batch['z'] = z
        batch['local_orient'] = local_orient
        batch['d_heading_vec'] = d_heading_vec

        return batch
    

# local_traj = batch['local_traj_tp']         # [T, B, 1, 11]
# past_z = local_traj[1:, ..., 2:3]           # [T-1, B, 1]
# past_local_orient = local_traj[1:, ..., 3:9]

# out_local_traj_tp[..., 2:3] = past_z + out_local_traj_tp[..., 2:3]
# out_local_traj_tp[..., 3:9] = self.rot_output(past_local_orient, out_local_traj_tp[..., 3:9])

# out_trans_tp, out_orient_q = traj_local2global_heading(out_local_traj_tp, local_orient_type='6d', )
# out_local_traj = out_local_traj_tp.transpose(0, 1).contiguous()

# batch['out_trans_tp'] = out_trans_tp        # GT w_transl
# batch['out_orient_q_tp'] = out_orient_q     # w_orient_q_tp
# out_orient_tp = transforms.quaternion_to_matrix(out_orient_q)
# batch['out_orient_6d_tp'] = transforms.matrix_to_rotation_6d(out_orient_tp)
# batch['local_orient'] = out_local_traj[..., 3:9]
# batch['d_heading_vec'] = out_local_traj[..., 9:]