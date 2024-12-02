import torch
import torch.nn as nn
from .mlp import MLP
from .rnn import RNN
from .dist import Normal
from lib.utils.traj_utils import *

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
    def __init__(self, cfg, specs, ctx, **kwargs):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()
        self.cfg = cfg
        self.specs = specs
        self.ctx = ctx
        self.input_noise = specs.get('input_noise', None)
        self.use_jvel = specs.get('use_jvel', False)
        in_dim = 69
        if self.use_jvel:
            in_dim += 69
        cur_dim = in_dim

        """ in MLP """
        if 'in_mlp' in specs:
            in_mlp_cfg = specs['in_mlp']
            self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.in_mlp.out_dim
        else:
            self.in_mlp = None
        
        """ temporal network """
        temporal_cfg = specs['temporal_net']
        self.t_net_type = temporal_cfg['type']
        num_layers = temporal_cfg.get('num_layers', 1)
        self.temporal_net = nn.ModuleList()
        for _ in range(num_layers):
            net = RNN(cur_dim, temporal_cfg['hdim'], self.t_net_type, bi_dir=temporal_cfg.get('bi_dir', True))
            cur_dim = temporal_cfg['hdim']
            self.temporal_net.append(net)

        """ out MLP """
        if 'out_mlp' in specs:
            out_mlp_cfg = specs['out_mlp']
            self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.out_mlp.out_dim
        else:
            self.out_mlp = None

        if 'context_dim' in specs:
            self.fc = nn.Linear(cur_dim, specs['context_dim'])
            cur_dim = specs['context_dim']
        else:
            self.fc = None
        ctx['context_dim'] = cur_dim

    def forward(self, data):
        x_in = data['in_joint_pos_tp']
        if self.use_jvel:
            x_in = torch.cat([x_in, data['in_joint_vel_tp']], dim=-1)

        if self.training and self.input_noise is not None:
            x_in += torch.randn_like(x_in) * self.input_noise
        
        x = x_in
        if self.in_mlp is not None:
            x = self.in_mlp(x)

        for net in self.temporal_net:
            x = net(x)         

        if self.out_mlp is not None:
            x = self.out_mlp(x)
        if self.fc is not None:
            x = self.fc(x)

        data['context'] = x


class DataEncoder(nn.Module):
    """ Inference (encoder) model q(z|X,C) in CVAE p(X|C) """
    def __init__(self, cfg, specs, ctx, **kwargs):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()
        self.cfg = cfg
        self.specs = specs
        self.ctx = ctx
        self.nz = ctx['nz']    # dimension of latent code z
        self.input = specs.get('input', 'init_heading_coord')
        self.orient_type = specs.get('orient_type', 'axis_angle')
        assert self.orient_type in {'axis_angle', 'quat', '6d'}
        self.pooling = specs['pooling']
        self.append_context = specs['append_context']
        self.use_jvel = specs.get('use_jvel', False)
        if self.input == 'local_traj':
            cur_dim = 11
        else:
            cur_dim = {'axis_angle': 6, 'quat': 7, '6d': 9}[self.orient_type]
        if self.append_context == 'early':
            cur_dim += ctx['context_dim']

        """ in MLP """
        if 'in_mlp' in specs:
            in_mlp_cfg = specs['in_mlp']
            self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.in_mlp.out_dim
        else:
            self.in_mlp = None

        """ temporal network """
        temporal_cfg = specs['temporal_net']
        self.t_net_type = temporal_cfg['type']
        num_layers = temporal_cfg.get('num_layers', 1)
        self.temporal_net = nn.ModuleList()
        for _ in range(num_layers):
            net = RNN(cur_dim, temporal_cfg['hdim'], self.t_net_type, bi_dir=temporal_cfg.get('bi_dir', True))
            cur_dim = temporal_cfg['hdim']
            self.temporal_net.append(net)

        """ out MLP """
        if 'out_mlp' in specs:
            out_mlp_cfg = specs['out_mlp']
            self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.out_mlp.out_dim
        else:
            self.out_mlp = None

        """ fusion MLP """
        if self.append_context == 'late':
            cur_dim += ctx['context_dim']
            fusion_mlp_cfg = specs['fusion_mlp']
            self.fusion_mlp = MLP(cur_dim, fusion_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.fusion_mlp.out_dim
        else:
            self.fusion_mlp = None
        num_dist_params = 2 * self.nz
        self.q_z_net = nn.Linear(cur_dim, num_dist_params)
        initialize_weights(self.q_z_net.modules())

    def forward(self, data):
        context = data['context']
        if self.input == 'global_traj':
            orient_key = {'axis_angle': '', '6d': '_6d', 'quat': '_q_'}[self.orient_type]
            x_in = torch.cat([data['trans_tp'], data[orient_key]], dim=-1)
        elif self.input == 'init_heading_coord':
            init_heading_orient, init_heading_trans = convert_traj_world2heading(data['orient_q_tp'], data['trans_tp'])
            if self.orient_type == 'axis_angle':
                init_heading_orient = quaternion_to_angle_axis(init_heading_orient)
            elif self.orient_type == '6d':
                init_heading_orient = quat_to_rot6d(init_heading_orient)
            x_in = torch.cat([init_heading_trans, init_heading_orient], dim=-1)
        else:
            x_in = data['local_traj_tp'].clone()
            x_in[0, :, [0, 1, -2, -1]] = x_in[1, :, [0, 1, -2, -1]]   # frame 0 stores the abs values of x, y, yaw, copy the relative value from frame 1

        if self.append_context == 'early':
            x_in = torch.cat([x_in, context], dim=-1)

        x = x_in
        if self.in_mlp is not None:
            x = self.in_mlp(x)

        for net in self.temporal_net:
            x = net(x)

        if self.out_mlp is not None:
            x = self.out_mlp(x)

        if self.append_context == 'late':
            x = torch.cat([x, context], dim=-1)
            x = self.fusion_mlp(x)
        
        if self.pooling == 'mean':
            x = x.mean(dim=0)
        else:
            x = x.max(dim=0)

        q_z_params = self.q_z_net(x)
        data['q_z_dist'] = Normal(params=q_z_params)
        data['q_z_samp'] = data['q_z_dist'].rsample()


class DataDecoder(nn.Module):
    """ Likelihood (decoder) model p(X|z,C) in CVAE p(X|C) """
    def __init__(self, cfg, specs, ctx, **kwargs):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()
        self.cfg = cfg
        self.specs = specs
        self.ctx = ctx
        self.nz = ctx['nz']    # dimension of latent code z
        self.pooling = specs['pooling']
        self.learn_prior = specs['learn_prior']
        self.deheading_local = ctx['deheading_local']
        self.local_orient_type = ctx['local_orient_type']
        self.use_jvel = specs.get('use_jvel', False)
        self.traj_dim = 11 if self.local_orient_type == '6d' else 8
        cur_dim = ctx['context_dim'] + self.nz

        """ in MLP """
        if 'in_mlp' in specs:
            in_mlp_cfg = specs['in_mlp']
            self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.in_mlp.out_dim
        else:
            self.in_mlp = None
        
        """ temporal network """
        if 'temporal_net' in specs:
            temporal_cfg = specs['temporal_net']
            self.t_net_type = temporal_cfg['type']
            num_layers = temporal_cfg.get('num_layers', 1)
            self.temporal_net = nn.ModuleList()
            for _ in range(num_layers):
                net = RNN(cur_dim, temporal_cfg['hdim'], self.t_net_type, bi_dir=temporal_cfg.get('bi_dir', True))
                cur_dim = temporal_cfg['hdim']
                self.temporal_net.append(net)
        else:
            self.temporal_net = None

        """ out MLP """
        if 'out_mlp' in specs:
            out_mlp_cfg = specs['out_mlp']
            self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.out_mlp.out_dim
        else:
            self.out_mlp = None

        self.out_fc = nn.Linear(cur_dim, self.traj_dim)
        initialize_weights(self.out_fc.modules())

        """ Prior """
        if self.learn_prior:
            cur_dim = ctx['context_dim']
            if 'prior_mlp' in specs:
                prior_mlp_cfg = specs['prior_mlp']
                self.prior_mlp = MLP(cur_dim, prior_mlp_cfg['hdim'], ctx['mlp_htype'])
                cur_dim = self.prior_mlp.out_dim
            else:
                self.prior_mlp = None
            num_dist_params = 2 * self.nz
            self.p_z_net = nn.Linear(cur_dim, num_dist_params)
            initialize_weights(self.p_z_net.modules())


    def forward(self, data, mode, sample_num=1):
        if mode in {'train', 'recon'}:
            assert sample_num == 1
        context = data['context']
        if sample_num > 1:
            context = context.repeat_interleave(sample_num, dim=1)
        # prior p(z) or p(z|C)
        if self.learn_prior:
            if self.pooling == 'mean':
                h = context.mean(dim=0)
            else:
                h = context.max(dim=0)
            h = self.prior_mlp(h)
            p_z_params = self.p_z_net(h)
            data['p_z_dist' + ('_infer' if mode == 'infer' else '')] = Normal(params=p_z_params)
        else:
            data['p_z_dist' + ('_infer' if mode == 'infer' else '')] = Normal(params=torch.zeros(context.shape[1], 2 * self.nz).type_as(context))

        if mode == 'train':
            z = data['q_z_samp']
        elif mode == 'recon':
            z =  data['q_z_dist'].mode()
        elif mode == 'infer':
            eps = data['in_traj_latent'] if 'in_traj_latent' in data else None
            z = data['p_z_dist_infer'].sample(eps)
        else:
            raise ValueError('Unknown Mode!')

        x_in = torch.cat([z.repeat((context.shape[0], 1, 1)), context], dim=-1)
            
        if self.in_mlp is not None:
            x = self.in_mlp(x_in.view(-1, x_in.shape[-1])).view(*x_in.shape[:2], -1)
        else:
            x = x_in

        if self.temporal_net is not None:
            for net in self.temporal_net:
                x = net(x)

        if self.out_mlp is not None:
            x = self.out_mlp(x)
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
            init_xy = data['local_traj_tp'][:1, :, :2].unsqueeze(2).repeat((1, 1, sample_num, 1))
            init_heading_vec = data['local_traj_tp'][:1, :, -2:].unsqueeze(2).repeat((1, 1, sample_num, 1))
        else:
            init_xy = torch.zeros_like(out_local_traj_tp[:1, ..., :2])
            init_heading_vec = torch.tensor([0., 1.], device=init_xy.device).expand_as(out_local_traj_tp[:1, ..., -2:])
        out_local_traj_tp[..., :2] = torch.cat([init_xy, x[1:, ..., :2]], dim=0)   # d_xy
        out_local_traj_tp[..., -2:] = torch.cat([init_heading_vec, x[1:, ..., -2:]], dim=0)   # d_heading_vec
        if mode in {'recon', 'train'}:
             out_local_traj_tp = out_local_traj_tp.squeeze(2)
        data[f'{mode}_out_local_traj_tp'] = out_local_traj_tp
        data[f'{mode}_out_trans_tp'], data[f'{mode}_out_orient_q_tp'] = traj_local2global_heading(out_local_traj_tp, local_orient_type=self.local_orient_type, deheading_local=self.deheading_local)