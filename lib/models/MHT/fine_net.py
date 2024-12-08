import torch
import torch.nn as nn
from .mlp import MLP
from .rnn import RNN
from .dist import Normal
from .transformer import CausalTransformer
from lib.utils.traj_utils import *
from lib.utils import transforms
from collections import defaultdict

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
    """
    Context feature extraction at Camera space
    Input   : Body pose w/ Trajectoy (T, J, 3)
    Output  : latent z
    """
    def __init__(self, in_dim):
        super().__init__()
        self.input_noise = 0.01
        cur_dim = in_dim

        """ in MLP """        
        self.in_mlp = MLP(cur_dim, [512, 512, 256], 'relu')
        cur_dim = self.in_mlp.out_dim
        
        """ temporal network """
        self.temporal_net = CausalTransformer(d_model=256, nhead=4, nlayers=3)

        """ out MLP """
        self.out_mlp = MLP(cur_dim, [512, 256], 'relu')
        cur_dim = self.out_mlp.out_dim

    def forward(self, x_in):
        """
        x_in    : [T, B, dim]
        context : [T, B, 256]
        """
        if self.input_noise :
            x_in += torch.randn_like(x_in) * self.input_noise

        x = x_in
        x = self.in_mlp(x)
        x = self.temporal_net(x)    # [T-1, B, dim]
        x = self.out_mlp(x)
        return x

class Posterior(nn.Module):
    def __init__(self, in_dim) :
        super().__init__()
        self.latent_size = 48
        cur_dim = in_dim
        cur_dim = cur_dim + 256
        
        """ in MLP input : (x_t-1, x_t) """
        cur_dim = cur_dim*2
        self.in_mlp = MLP(cur_dim, [512, 512, 512, 512, self.latent_size*2], 'relu')
        cur_dim = self.in_mlp.out_dim

    def forward(self, x_past, x_curr):
        """
        x_past : [B, dim]
        """
        x_in = torch.cat([x_past, x_curr], dim=-1)
        x_out = self.in_mlp(x_in)

        mu = x_out[..., :self.latent_size]
        logvar = x_out[..., self.latent_size:]
        
        return mu, logvar

class Prior(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.latent_size = 48
        cur_dim = in_dim
        cur_dim = cur_dim + 256
        
        """ in MLP input : (x_t-1) """
        self.in_mlp = MLP(cur_dim, [512, 512, 512, 512, self.latent_size*2], 'relu')
        cur_dim = self.in_mlp.out_dim

    def forward(self, x_past):
        """
        x : [B, dim] 
        """
        x_out = self.in_mlp(x_past)

        mu = x_out[..., :self.latent_size]
        logvar = x_out[..., self.latent_size:]
        
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, in_dim) :
        super().__init__()
        self.latent_size = 48
        cur_dim = in_dim
        cur_dim += 256
        out_dim = 3 + 6 + 2   # trans(3), orient(6), contact(2)

        """ in MLP input : (x_t-1, z) """
        cur_dim = cur_dim + self.latent_size
        self.in_mlp = MLP(cur_dim, [512, 512, 512, 512, out_dim], 'relu')

    def forward(self, x_past, z):
        B = z.shape[0]

        x_in = torch.cat([x_past, z], dim=-1)
        x_out = self.in_mlp(x_in)

        x_out = x_out.reshape(B, 1, -1)
        return x_out

class FineNetwork(nn.Module):
    """
    Context network
         Input : 3D keypoints (Camera coordinate)
        Output : Context feature
    
    Prosterior network
        Input : 3D kepoints (World coordinate), Context feature
        Output : latent feature z
    
    Prior network
        Input   : Context feature
        Output  : latent feature z
    
    Decoder network
        Input   : latent feature z
        Output  : translation, orientation (World coordinate)
    """
    def __init__(self, num_joints=17, stage='fine') :
        super().__init__()
        self.use_conditional_prior = True
        self.stage = stage
        in_dim = num_joints*3 + num_joints*3

        self.context_encoder = ContextEncoder(in_dim)
        self.posterior = Posterior(in_dim)
        self.prior = Prior(in_dim)
        self.decoder = Decoder(in_dim)

    def rot_output(self, past_6d, trans_6d):
        past = transforms.rotation_6d_to_matrix(past_6d)        # [B, 1, 3, 3]
        trans = transforms.rotation_6d_to_matrix(trans_6d)      # [B, 1, 3, 3]
        curr = torch.matmul(trans, past)

        return transforms.matrix_to_rotation_6d(curr)           # [B, 1, 6]

    def forward_infer(self, data):
        T = data['seqlen']

        x_in = data['input_tp']     # [T, B, (17*3+17*3) + (6+6) + ]
        transl = data['w_transl_tp']   # [T, B, 3]
        orinet = data['w_orient_tp']   # [T, B, 3]

        stack_res = defaultdict(list)
        stack_res['out_trans'].append(transl[:1])
        stack_res['out_orient'].append(orinet[:1])

        for t in range(1, T):
            x_past = x_in[t-1]
            transl_past = transl[t-1]               # [B, 3]
            orinet_past = orinet[t-1]               # [B, 6]
            
            pm, pv = self.prior(x_past)
            p_z_dist = Normal(mu=pm, logvar=pv)

            p_z_samp = p_z_dist.rsample()
            dec_out = self.decoder(x_past, p_z_samp) # [B, 1, 11] (local coordinate)

            del_trans = dec_out[..., :3]
            del_orient = dec_out[..., 3:3+6]
            pred_contact = dec_out[..., 9:]

            pred_trans = transl_past + del_trans
            pred_oreint = self.rot_output(orinet_past, del_orient)

            stack_res['out_trans_tp'].append(pred_trans)
            stack_res['out_orient_tp'].append(pred_oreint)
            stack_res['pred_contact_tp'].append(pred_contact)

        out_trans_tp = torch.stack(stack_res['out_trans_tp'])           # [T, B, 3]
        out_orient_tp = torch.stack(stack_res['out_orient_tp'])         # [T, B, 3, 3]
        pred_contact_tp = torch.stack(stack_res['pred_contact_tp'])     # [T, B, 3, 3]

        data['fine_trans'] = out_trans_tp.transpose(0, 1).contiguous()
        data['fine_orient'] = out_orient_tp.transpose(0, 1).contiguous()
        data['pred_contact'] = pred_contact_tp.transpose(0, 1).contiguous()

    def forward(self, data):
        T = data['seqlen']
        
        x_in = torch.cat([data['c_kp3d_coco_tp'], data['c_vel_kp3d_coco_tp']], dim=-1)  # [T, B, dim]
        context = self.context_encoder(x_in)
        x_in = torch.cat([x_in, context], dim=-1)

        transl = data['w_transl_tp']        # [T, B, 1, 3]
        orinet = data['w_orient_6d_tp']     # [T, B, 1, 6]
        contact = data['contact_tp']

        stack_res = defaultdict(list)
        stack_res['out_trans_tp'].append(transl[0])     # [B, 1, 3]
        stack_res['out_orient_tp'].append(orinet[0])    # [B, 1, 6]
        stack_res['out_contact_tp'].append(contact[0])  # [B, 2]

        for t in range(1, T):
            x_past, x_curr = x_in[t-1], x_in[t]     # [B, dim]
            transl_past = transl[t-1]               # [B, 1, 3]
            orinet_past = orinet[t-1]               # [B, 1, 6]
            
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
            
            del_trans = dec_out[..., :3]
            del_orient = dec_out[..., 3:3+6]
            pred_contact = dec_out[..., 0, 9:]      # [B, 2]

            pred_trans = transl_past + del_trans    # [B, 1, 3]
            pred_oreint = self.rot_output(orinet_past, del_orient)

            stack_res['p_z_dist'].append(p_z_dist)
            stack_res['q_z_dist'].append(q_z_dist)
            stack_res['out_trans_tp'].append(pred_trans)
            stack_res['out_orient_tp'].append(pred_oreint)
            stack_res['out_contact_tp'].append(pred_contact)

        out_trans_tp = torch.stack(stack_res['out_trans_tp'])           # [T, B, 3]
        out_orient_tp = torch.stack(stack_res['out_orient_tp'])         # [T, B, 3, 3]
        out_contact_tp = torch.stack(stack_res['out_contact_tp'])     # [T, B, 3, 3]

        data['fine_trans'] = out_trans_tp.transpose(0, 1).contiguous()
        data['fine_orient'] = out_orient_tp.transpose(0, 1).contiguous()
        data['out_contact'] = out_contact_tp.transpose(0, 1).contiguous()

