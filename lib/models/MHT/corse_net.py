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
    def __init__(self):
        """
        Encoding context feat. 
        Input   : Ego-centric coord. motion feature
        Output  : Context feature
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
        x_in = data['h_kp3d_tp']
        if self.input_noise :
            x_in += torch.randn_like(x_in) * self.input_noise
        
        x = x_in
        x = self.in_mlp(x)
        for net in self.temporal_net:
            x = net(x)         
        x = self.out_mlp(x)
        data['context'] = x

class Posterior(nn.Module):
    """
    Input   : World Coordinate traj(1:T), Context feat.(1:T)
    Output  : latent vec. z
    """
    def __init__(self) :
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

        # World coor. trajectory 
        x = x_in
        x = self.in_mlp(x)
        x = x.reshape(data['seqlen'], data['batch_size'], -1)

        for net in self.temporal_net:
            x = net(x)

        x = self.out_mlp(x)

        # World coor. trajectory + Ego coor. context
        x = torch.cat([x, context], dim=-1)
        x = self.fusion_mlp(x) 
        
        x = x.mean(dim=0)

        q_z_params = self.q_z_net(x)        # [B, dim]
        data['q_z_dist'] = Normal(params=q_z_params)
        data['q_z_samp'] = data['q_z_dist'].rsample()
    
class Decoder(nn.Module):
    def __init__(self, use_clip=False):
        super().__init__()
        self.nz = 128    # dimension of latent code z
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
        context = data['context']       # [T, B, dim]
        context = context.unsqueeze(-2) # [T, B, 1, dim]

        if sample_num > 1:
            context = context.repeat_interleave(sample_num, dim=-2) # [T, B, N, dim]
        
        h = context.mean(dim=0)
        h = self.prior_mlp(h)
        p_z_params = self.p_z_net(h)
        data['p_z_dist'] = Normal(params=p_z_params)

        if mode == 'train':
            z = data['q_z_samp']
            #z = z.view(1, data['batch_size'], 1, -1)
        elif mode == 'recon':
            z =  data['q_z_dist'].mode()
        elif mode == 'infer':
            z = data['p_z_dist'].mode()
        else:
            raise ValueError('Unknown Mode!')

        z = z.view(1, data['batch_size'], 1, -1)
        x_in = torch.cat([z.repeat((context.shape[0], 1, 1, 1)), context], dim=-1)
        x = self.out_mlp(x_in)
        x = self.out_fc(x)
        x = x.view(-1, data['batch_size'], sample_num, x.shape[-1])

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
        
        data['coarse_out_local_traj_tp'] = out_local_traj_tp
        out_trans_tp, out_orient_q_tp = traj_local2global_heading(out_local_traj_tp, local_orient_type=self.local_orient_type, deheading_local=self.deheading_local)

        out_orient_tp = transforms.quaternion_to_matrix(out_orient_q_tp)
        out_orient_6d_tp = transforms.matrix_to_rotation_6d(out_orient_tp)

        data['coarse_out_orient_q_tp'] = out_orient_q_tp 
        data['coarse_out_orient_6d_tp'] = out_orient_6d_tp
        data['coarse_out_local_traj_tp'] = out_local_traj_tp
        data['coarse_out_trans_tp'] = out_trans_tp
    
        data['coarse_out_trans'] = out_trans_tp.transpose(0, 1).contiguous()
        data['coarse_out_orient'] = out_orient_tp.transpose(0, 1).contiguous()

class CoarseNetwork(nn.Module):
    def __init__(self, stage='coarse'):
        super().__init__()
        self.stage = stage
        self.context_encoder = ContextEncoder()
        self.encoder = Posterior()
        self.decoder = Decoder()

    def forward_infer(self, data):
        self.context_encoder(data)
        self.decoder(data, 'infer')

    def forward(self, data) :
        """
        motion feat : Ego-cent. coord. kp feat 3d
        """
        self.context_encoder(data)
        self.encoder(data)
        self.decoder(data, 'train')
    
            