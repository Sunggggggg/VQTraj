import torch
import torch.nn as nn
from collections import defaultdict
from .mlp import MLP
from .dist import Normal
from .rnn import RNN
from lib.utils.traj_utils import *
from lib.utils import transforms

class MotionEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.input_noise = 0.01
        self.use_jvel = True
        cur_dim = in_dim

        """ in MLP """        
        self.in_mlp = MLP(cur_dim, [512, 512, 256], 'relu')
        cur_dim = self.in_mlp.out_dim
        
        """ temporal network """
        self.t_net_type = 'lstm'
        num_layers = 2
        self.temporal_net = nn.ModuleList()
        for _ in range(num_layers):
            net = RNN(cur_dim, 256, self.t_net_type, bi_dir=True)
            cur_dim = 256
            self.temporal_net.append(net)

        """ out MLP """
        self.out_mlp = MLP(cur_dim, [512, 512], 'relu')
        cur_dim = self.out_mlp.out_dim

    def forward(self, data):
        x_in = data['input_tp']
        x_in += torch.randn_like(x_in) * self.input_noise
        
        x = x_in
        x = self.in_mlp(x)
        for net in self.temporal_net:
            x = net(x)         
        x = self.out_mlp(x)
        data['motion'] = x     # [T, B, dim]

class Posterior(nn.Module):
    def __init__(self) :
        super().__init__()
        self.orient_type = 'axis_angle'
        self.latent_size = 48
        cur_dim = {'axis_angle': 3+3, 'quat': 4+3, '6d': 6+3}[self.orient_type]
        cur_dim = cur_dim + 512
        
        """ in MLP input : (x_t-1, x_t) """
        cur_dim = cur_dim*2
        self.in_mlp = MLP(cur_dim, [512, 512, 512, 512, self.latent_size*2], 'relu')
        cur_dim = self.in_mlp.out_dim

    def forward(self, x_past, x_curr):
        """
        x_past : [B, dim]  - which is consist of oreint. and trans.
        """
        x_in = torch.cat([x_past, x_curr], dim=-1)
        x_out = self.in_mlp(x_in)

        mu = x_out[..., :self.latent_size]
        logvar = x_out[..., self.latent_size:]
        
        return mu, logvar
    
class Prior(nn.Module):
    def __init__(self) :
        super().__init__()
        self.orient_type = 'axis_angle'
        self.latent_size = 48
        cur_dim = {'axis_angle': 3+3, 'quat': 4+3, '6d': 6+3}[self.orient_type]
        cur_dim += 512
        
        """ in MLP input : (x_t-1) """
        self.in_mlp = MLP(cur_dim, [512, 512, 512, 512, self.latent_size*2], 'relu')
        cur_dim = self.in_mlp.out_dim

    def forward(self, x_past):
        """
        x_past : [B, dim]  - which is consist of oreint. and trans.
        """
        x_out = self.in_mlp(x_past)

        mu = x_out[..., :self.latent_size]
        logvar = x_out[..., self.latent_size:]
        
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self) :
        super().__init__()
        self.latent_size = 48
        self.orient_type = 'axis_angle'
        cur_dim = {'axis_angle': 3+3, 'quat': 4+3, '6d': 6+3}[self.orient_type]
        cur_dim += 512
        out_dim = 2+1+6+2+2 # dxy(2), z(1), rot(6), d_head(2), contact(2)
        
        """ in MLP input : (x_t-1, z) """
        cur_dim = cur_dim + self.latent_size
        self.in_mlp = MLP(cur_dim, [512, 512, 512, 512, out_dim], 'relu')

    def forward(self, x_past, z):
        B = z.shape[0]

        x_in = torch.cat([x_past, z], dim=-1)
        x_out = self.in_mlp(x_in)

        x_out = x_out.reshape(B, 1, -1)
        
        return x_out

class HumorModel(nn.Module):
    def __init__(self) :
        super().__init__()
        self.local_orient_type = '6d'
        self.orient_type = 'axis_angle'
        self.deheading_local = True
        self.use_conditional_prior = True

        self.posterior = Posterior()
        self.prior = Prior()
        self.decoder = Decoder()

    def rot_output(self, past_6d, trans_6d):
        past = transforms.rotation_6d_to_matrix(past_6d)        # [B, 1, 3, 3]
        trans = transforms.rotation_6d_to_matrix(trans_6d)      # [B, 1, 3, 3]
        curr = torch.matmul(trans, past)

        return transforms.matrix_to_rotation_6d(curr)           # [B, 1, 6]

    def forward(self, data, sample_num=1):
        T = data['seqlen']

        # Prepare
        local_traj_tp = data['local_traj_tp']
        motion = data['motion'].unsqueeze(-2)       # [T, B, dim]
        z = local_traj_tp[..., 2:2+1]               # [T, B, 1, 1]
        local_orient = local_traj_tp[..., 3:3+6]    # [T, B, 1, 6]

        init_heading_orient, init_heading_trans = convert_traj_world2heading(data['orient_q_tp'], data['trans_tp'])
        if self.orient_type == 'axis_angle':
            init_heading_orient = transforms.quaternion_to_angle_axis(init_heading_orient)
        elif self.orient_type == '6d':
            init_heading_orient = quat_to_rot6d(init_heading_orient)
        x_in = torch.cat([init_heading_trans, init_heading_orient, motion], dim=-1) # [T, B, 1, 3+3+256]

        stack_res = defaultdict(list)
        for t in range(1, T) :
            # Prepare
            x_past, x_curr = x_in[t-1], x_in[t]     # [B, dim]
            z_past = z[t-1]; local_orient_past = local_orient[t-1]  # [B, 1, dim]

            # Model forward
            qm, qv = self.posterior(x_past, x_curr)

            if self.use_conditional_prior :
                pm, pv = self.prior(x_past)
            else :
                pm, pv = torch.zeros_like(qm), torch.ones_like(qv)

            q_z_dist = Normal(mu=qm, logvar=qv)
            p_z_dist = Normal(mu=pm, logvar=pv)

            q_z_samp = q_z_dist.rsample()

            dec_out = self.decoder(x_past, q_z_samp) # [B, 1, 11] (local coordinate)

            d_xy = dec_out[..., :2]
            d_z = dec_out[..., 2:2+1]
            d_local_orient = dec_out[..., 3:3+6]
            d_heading_vec = dec_out[..., 9:11]
            contact_label = dec_out[..., 11:13]

            z_curr = z_past + d_z
            local_orient_curr = self.rot_output(local_orient_past, d_local_orient)

            out_local_traj_tp = torch.cat([d_xy, z_curr, local_orient_curr, d_heading_vec], dim=-1)

            stack_res['pred_contact'].append(contact_label)
            stack_res['q_z_dist'].append(q_z_dist)  
            stack_res['p_z_dist'].append(p_z_dist)
            stack_res['out_local_traj_tp'].append(out_local_traj_tp)

        out_local_traj_tp = torch.stack(stack_res['out_local_traj_tp'], dim=0)     # [T-1, B, 1, 11]
        
        init_xy = data['local_traj_tp'][:1, ..., :2]
        init_heading_vec = data['local_traj_tp'][:1, ..., -2:]

        out_local_traj_tp[..., :2] = torch.cat([init_xy, out_local_traj_tp[1:, ..., :2]], dim=0)   # d_xy
        out_local_traj_tp[..., -2:] = torch.cat([init_heading_vec, out_local_traj_tp[1:, ..., -2:]], dim=0)   # d_heading_vec

        out_trans_tp, out_orient_q = traj_local2global_heading(out_local_traj_tp, local_orient_type=self.local_orient_type, deheading_local=self.deheading_local)
        
        out_orient_tp = transforms.quaternion_to_matrix(out_orient_q)
        
        data['out_trans'] = out_trans_tp.transpose(0, 1).contiguous()
        data['out_orient_aa'] = transforms.quaternion_to_axis_angle(out_orient_q).transpose(0, 1).contiguous()
        data['out_orient'] = transforms.quaternion_to_matrix(out_orient_q).transpose(0, 1).contiguous()
        
        data['out_orient_q_tp'] = out_orient_q
        data['out_orient_6d_tp'] = transforms.matrix_to_rotation_6d(out_orient_tp)
        data['out_local_traj_tp'] = out_local_traj_tp
        data['out_trans_tp'] = out_trans_tp
        
        data['pred_contact_tp'] = torch.stack(stack_res['pred_contact'], dim=0)    # [T-1, B, 1, 2]
        data['q_z_dist'] = stack_res['q_z_dist']
        data['p_z_dist'] = stack_res['p_z_dist']