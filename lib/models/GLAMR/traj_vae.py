import torch
import torch.nn as nn
from .mlp import MLP
from .rnn import RNN
from .dist import Normal
from lib.utils.traj_utils import *
from lib.utils import transforms

def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

class ContextEncoder(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()
        self.input_noise = 0.01
        self.use_jvel = False
        in_dim = 51
        if self.use_jvel:
            in_dim += 51
        cur_dim = in_dim

        """ in MLP """        
        self.in_mlp = MLP(cur_dim, [512, 256], 'relu')
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
        self.out_mlp = MLP(cur_dim, [512, 256], 'relu')
        cur_dim = self.out_mlp.out_dim

    def forward(self, data):
        x_in = data['in_joint_pos_tp']
        x_in += torch.randn_like(x_in) * self.input_noise
        
        x = x_in
        x = self.in_mlp(x)
        for net in self.temporal_net:
            x = net(x)         
        x = self.out_mlp(x)
        data['context'] = x


class DataEncoder(nn.Module):
    """ Inference (encoder) model q(z|X,C) in CVAE p(X|C) """
    def __init__(self):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()
        self.nz = 128    # dimension of latent code z
        self.input = 'init_heading_coord'
        self.orient_type = 'axis_angle'
        assert self.orient_type in {'axis_angle', 'quat', '6d'}
        self.pooling = 'mean'
        self.append_context = 'late'
        self.use_jvel = False
        if self.input == 'local_traj':
            cur_dim = 11
        else:
            cur_dim = {'axis_angle': 6, 'quat': 7, '6d': 9}[self.orient_type]

        """ in MLP """
        self.in_mlp = MLP(cur_dim, [512, 256], 'relu')
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
        self.out_mlp = MLP(cur_dim, [512, 256], 'relu')
        cur_dim = self.out_mlp.out_dim


        """ fusion MLP """
        cur_dim += 256
        self.fusion_mlp = MLP(cur_dim, [512, 256], 'relu')
        cur_dim = self.fusion_mlp.out_dim

        num_dist_params = 2 * self.nz
        self.q_z_net = nn.Linear(cur_dim, num_dist_params)
        initialize_weights(self.q_z_net.modules())

    def forward(self, data):
        context = data['context']     
        init_heading_orient, init_heading_trans = convert_traj_world2heading(data['orient_q_tp'], data['trans_tp'])
        if self.orient_type == 'axis_angle':
            init_heading_orient = transforms.quaternion_to_angle_axis(init_heading_orient)
        elif self.orient_type == '6d':
            init_heading_orient = quat_to_rot6d(init_heading_orient)
        x_in = torch.cat([init_heading_trans, init_heading_orient], dim=-1)


        x = x_in
        x = self.in_mlp(x)
        x = x.reshape(data['seqlen'], data['batch_size'], -1)

        for net in self.temporal_net:
            x = net(x)

        x = self.out_mlp(x)

        x = torch.cat([x, context], dim=-1)
        x = self.fusion_mlp(x)
        
        x = x.mean(dim=0)

        q_z_params = self.q_z_net(x)
        data['q_z_dist'] = Normal(params=q_z_params)
        data['q_z_samp'] = data['q_z_dist'].rsample()


class DataDecoder(nn.Module):
    """ Likelihood (decoder) model p(X|z,C) in CVAE p(X|C) """
    def __init__(self, use_clip=False):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()
        self.nz = 128    # dimension of latent code z
        self.pooling = 'mean'
        self.learn_prior = True
        self.deheading_local = True
        self.local_orient_type = '6d'
        self.use_jvel = False
        self.traj_dim = 11 if self.local_orient_type == '6d' else 8
        cur_dim = 256 + self.nz

        """ CLIP """
        self.use_clip = use_clip
        if use_clip :
            self.clip_mlp = MLP(256 + self.nz + 512, [512, 512], 'relu')
            cur_dim = self.clip_mlp.out_dim

        """ out MLP """
        self.out_mlp = MLP(cur_dim, [512, 256], 'relu')
        cur_dim = self.out_mlp.out_dim


        self.out_fc = nn.Linear(cur_dim, self.traj_dim)
        initialize_weights(self.out_fc.modules())

        """ Prior """
        cur_dim = 256
        self.prior_mlp = MLP(cur_dim, [512, 256], 'relu')
        cur_dim = self.prior_mlp.out_dim
   
        num_dist_params = 2 * self.nz
        self.p_z_net = nn.Linear(cur_dim, num_dist_params)
        initialize_weights(self.p_z_net.modules())


    def forward(self, data, mode, sample_num=1):
        if mode in {'train', 'recon'}:
            assert sample_num == 1
        context = data['context']
        if sample_num > 1:
            context = context.repeat_interleave(sample_num, dim=1)
        
        if self.pooling == 'mean':
            h = context.mean(dim=0)
        h = self.prior_mlp(h)
        p_z_params = self.p_z_net(h)
        data['p_z_dist' + ('_infer' if mode == 'infer' else '')] = Normal(params=p_z_params)

        if mode == 'train':
            z = data['q_z_samp']
        elif mode == 'recon':
            z =  data['q_z_dist'].mode()
        elif mode == 'infer':
            eps = data['in_traj_latent'] if 'in_traj_latent' in data else None
            z = data['p_z_dist_infer'].sample(eps)
        else:
            raise ValueError('Unknown Mode!')

        if self.use_clip :
            clip_feat = data['z'].view(1, data['batch_size'], 512).repeat((context.shape[0], 1, 1))
            x_in = torch.cat([z.repeat((context.shape[0], 1, 1)), context, clip_feat], dim=-1)
            x_in = self.clip_mlp(x_in)
        else :
            x_in = torch.cat([z.repeat((context.shape[0], 1, 1)), context], dim=-1)

        x = self.out_mlp(x_in)
        x = self.out_fc(x)
        x = x.view(-1, data['batch_size'], sample_num, x.shape[-1])
        
        orig_out_local_traj_tp = x
        if mode in {'recon', 'train'}:
             orig_out_local_traj_tp = orig_out_local_traj_tp.squeeze(2)
        data[f'{mode}_orig_out_local_traj_tp'] = orig_out_local_traj_tp

        out_local_traj_tp = x.clone()
        if 'init_xy' in data:
            init_xy = data['init_xy'].unsqueeze(0).unsqueeze(2).repeat((1, 1, sample_num, 1))
            init_heading_vec = heading_to_vec(data['init_heading']).unsqueeze(0).unsqueeze(2).repeat((1, 1, sample_num, 1))
        elif 'local_traj_tp' in data:
            init_xy = data['local_traj_tp'][:1, ..., :2].repeat((1, 1, sample_num, 1))
            init_heading_vec = data['local_traj_tp'][:1, ..., -2:].repeat((1, 1, sample_num, 1))
        else:
            init_xy = torch.zeros_like(out_local_traj_tp[:1, ..., :2])
            init_heading_vec = torch.tensor([0., 1.], device=init_xy.device).expand_as(out_local_traj_tp[:1, ..., -2:])
        out_local_traj_tp[..., :2] = torch.cat([init_xy, x[1:, ..., :2]], dim=0)   # d_xy
        out_local_traj_tp[..., -2:] = torch.cat([init_heading_vec, x[1:, ..., -2:]], dim=0)   # d_heading_vec
        
        data[f'out_local_traj_tp'] = out_local_traj_tp
        out_trans_tp, out_orient_q = traj_local2global_heading(out_local_traj_tp, local_orient_type=self.local_orient_type, deheading_local=self.deheading_local)

        out_orient_tp = transforms.quaternion_to_matrix(out_orient_q)
        data['out_trans_tp'] = out_trans_tp        # GT w_transl
        data['out_trans'] = out_trans_tp.transpose(0, 1).contiguous()        # GT w_transl
        
        data['out_orient_q_tp'] = out_orient_q     # w_orient_q_tp
        data['out_orient_6d_tp'] = transforms.matrix_to_rotation_6d(out_orient_tp)
        data['out_local_traj_tp'] = out_local_traj_tp
        data['out_orient_aa'] = transforms.quaternion_to_axis_angle(out_orient_q).transpose(0, 1).contiguous()     # w_orient_q_tp
        data['out_orient'] = transforms.quaternion_to_matrix(out_orient_q).transpose(0, 1).contiguous()


class DataEncoder_V2(nn.Module):
    """ Inference (encoder) model q(z|X,C) in CVAE p(X|C) """
    def __init__(self):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()
        self.nz = 128    # dimension of latent code z
        self.input = 'init_heading_coord'
        self.orient_type = 'axis_angle'
        assert self.orient_type in {'axis_angle', 'quat', '6d'}
        self.pooling = 'mean'
        self.append_context = 'late'
        self.use_jvel = False
        if self.input == 'local_traj':
            cur_dim = 11
        else:
            cur_dim = {'axis_angle': 6, 'quat': 7, '6d': 9}[self.orient_type]

        """ in MLP """
        self.in_mlp = MLP(cur_dim, [256, 256], 'relu')
        cur_dim = self.in_mlp.out_dim
   
        """ condtion """
        self.cond_mlp = MLP(cur_dim*2, [1024, 512, 512], 'relu')
        cur_dim = self.cond_mlp.out_dim

        num_dist_params = 2 * self.nz
        self.q_z_net = nn.Linear(cur_dim, num_dist_params)
        initialize_weights(self.q_z_net.modules())

    def forward(self, data):
        T, B = data['seqlen'], data['batch_size']  
        init_heading_orient, init_heading_trans = convert_traj_world2heading(data['orient_q_tp'], data['trans_tp'])
        if self.orient_type == 'axis_angle':
            init_heading_orient = transforms.quaternion_to_angle_axis(init_heading_orient)
        elif self.orient_type == '6d':
            init_heading_orient = quat_to_rot6d(init_heading_orient)
        x_in = torch.cat([init_heading_trans, init_heading_orient], dim=-1)


        x = x_in
        x = self.in_mlp(x)  # [T, B, 256]
        
        q_z_dist_list, q_z_samp_list = [], []
        for t in range(1, T):
            x_past, x_curr = x[t-1], x[t] # 
            x_in = torch.cat([x_past, x_curr], dim=-1)  # [B, 512]
            x_in = self.cond_mlp(x_in)                  # [B, 512]
            q_z_params = self.q_z_net(x_in)                # [B, 256]
            q_z_dist = Normal(params=q_z_params)
            q_z_samp = q_z_dist.rsample()

            q_z_dist_list.append(q_z_dist)              # [B]
            q_z_samp_list.append(q_z_samp)

        data['q_z_dist'] = q_z_dist_list
        data['q_z_samp'] = torch.stack(q_z_samp_list, dim=0)  # [T-1, B, 256]

class DataDecoder_V2(nn.Module):
    """ Likelihood (decoder) model p(X|z,C) in CVAE p(X|C) """
    def __init__(self):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()
        self.nz = 128    # dimension of latent code z
        self.pooling = 'mean'
        self.learn_prior = True
        self.deheading_local = True
        self.local_orient_type = '6d'
        self.use_jvel = False
        self.traj_dim = 11 if self.local_orient_type == '6d' else 8
        cur_dim = self.nz

        """ out MLP """
        self.out_mlp = MLP(cur_dim, [512, 256], 'relu')
        cur_dim = self.out_mlp.out_dim

        self.out_fc = nn.Linear(cur_dim, self.traj_dim)
        initialize_weights(self.out_fc.modules())

        """ Prior """
        cur_dim = 6 # Axis angle
        self.prior_mlp = MLP(cur_dim, [512, 256], 'relu')
        cur_dim = self.prior_mlp.out_dim
   
        num_dist_params = 2 * self.nz
        self.p_z_net = nn.Linear(cur_dim, num_dist_params)
        initialize_weights(self.p_z_net.modules())

    def rot_output(self, past_6d, trans_6d):
        t, B = past_6d.shape[:2]
        past = transforms.rotation_6d_to_matrix(past_6d)        # [B, 1, 3, 3]
        trans = transforms.rotation_6d_to_matrix(trans_6d)      # [B, 1, 3, 3]
        curr = torch.matmul(trans, past)

        return transforms.matrix_to_rotation_6d(curr).reshape(t, B, 1, 6)

    def forward(self, data, mode, sample_num=1):
        if mode in {'train', 'recon'}:
            assert sample_num == 1

        if mode == 'train':
            z = data['q_z_samp']    # [T-1, B, 256]
        elif mode == 'recon':
            z =  data['q_z_dist'].mode()
        elif mode == 'infer':
            eps = data['in_traj_latent'] if 'in_traj_latent' in data else None
            z = data['p_z_dist_infer'].sample(eps)
        else:
            raise ValueError('Unknown Mode!')

        x = self.out_mlp(z)
        x = self.out_fc(x)
        delta_local_traj_tp = x.view(-1, data['batch_size'], sample_num, x.shape[-1]) # [T-1, B, 1, 11]
        local_traj_tp = data['local_traj_tp']               # [T, B, 1, 11]
        out_local_traj_tp = torch.zeros_like(local_traj_tp[1:])

        past_z = local_traj_tp[:-1, ..., 2:3]               # [T-1, B, 1, 1]
        past_local_orient = local_traj_tp[:-1, ..., 3:9]    # [T-1, B, 1, 6]

        init_xy = data['local_traj_tp'][:1, ..., :2].repeat((1, 1, sample_num, 1))
        init_heading_vec = data['local_traj_tp'][:1, ..., -2:].repeat((1, 1, sample_num, 1))

        out_local_traj_tp[..., :2] = delta_local_traj_tp[..., :2]   # d_xy
        out_local_traj_tp[..., 2:3] = past_z + delta_local_traj_tp[..., 2:3]
        out_local_traj_tp[..., 3:9] = self.rot_output(past_local_orient, delta_local_traj_tp[..., 3:9])
        out_local_traj_tp[..., -2:] = delta_local_traj_tp[..., -2:]   # d_heading_vec
        
        data[f'out_local_traj_tp'] = out_local_traj_tp
        out_trans_tp, out_orient_q = traj_local2global_heading(out_local_traj_tp, local_orient_type=self.local_orient_type, deheading_local=self.deheading_local)

        out_orient_tp = transforms.quaternion_to_matrix(out_orient_q)
        data['out_trans_tp'] = out_trans_tp        # GT w_transl
        data['out_trans'] = out_trans_tp.transpose(0, 1).contiguous()        # GT w_transl
        
        data['out_orient_q_tp'] = out_orient_q     # w_orient_q_tp
        data['out_orient_6d_tp'] = transforms.matrix_to_rotation_6d(out_orient_tp)
        data['out_local_traj_tp'] = out_local_traj_tp
        data['out_orient_aa'] = transforms.quaternion_to_axis_angle(out_orient_q).transpose(0, 1).contiguous()     # w_orient_q_tp
        data['out_orient'] = transforms.quaternion_to_matrix(out_orient_q).transpose(0, 1).contiguous()