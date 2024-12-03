import torch
import numpy as np
import torch.nn as nn
from .head import OutHead
from lib.utils import transforms
from lib.utils.traj_utils import convert_traj_world2heading, traj_local2global_heading
from .resnet import Resnet1D

class FCBlock(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ff(x)

class MLPBlock(nn.Module):
    def __init__(self, dim, inter_dim, out_dim=None, dropout_ratio=0.1):
        super().__init__()
        if out_dim is None : out_dim = dim
        self.ff = nn.Sequential(
            nn.Linear(dim, inter_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(inter_dim, out_dim),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x):
        return self.ff(x)

class MixerLayer(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 hidden_inter_dim, 
                 token_dim, 
                 token_inter_dim, 
                 dropout_ratio):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.MLP_token = MLPBlock(token_dim, token_inter_dim, dropout_ratio)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.MLP_channel = MLPBlock(hidden_dim, hidden_inter_dim, dropout_ratio)

    def forward(self, x):
        y = self.layernorm1(x)
        y = y.transpose(2, 1)
        y = self.MLP_token(y)
        y = y.transpose(2, 1)
        z = self.layernorm2(x + y)
        z = self.MLP_channel(z)
        out = x + y + z
        return out

class Encoder(nn.Module):
    def __init__(self,
                 num_tokens=40,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        blocks.append(nn.Upsample(num_tokens//2))
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, 3, 1, 1),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def preproces(self, x):
        """
        x : [T, B, dim] +> [B, dim, T]
        """
        return x.permute(1, 2, 0)

    def forward(self, batch):
        x = batch['input_tp']
        x = self.preproces(x)
        batch['encoded_feat'] = self.model(x)
        return batch

class Decoder(nn.Module):
    def __init__(self,
                 seqlen=81,
                 num_tokens=40,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        print(f'Num of tokens --> {num_tokens}')
        for i in list(np.linspace(seqlen, num_tokens, 1, endpoint=False, dtype=int)[::-1]):
            blocks.append(nn.Upsample(i))
            blocks.append(nn.Conv1d(width, width, 3, 1, 1))
            blocks.append(nn.ReLU())

        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                # nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def post_process(self, x):
        """
        x : [B, dim, T] => [T, B, dim]
        """
        return x.permute(2, 0, 1)

    def forward(self, batch):
        x = self.model(batch['quantized_feat'])  # [B, dim, T]
        x = self.post_process(x)
        batch['decoded_feat'] = x 
        return batch
    
class Encoder_v2(nn.Module):
    def __init__(self, 
                 in_dim,
                 hid_dim,
                 out_dim,
                 token_num,
                 up_sample_rate,
                 down_sample_rate,
                 res_depth,
                 dilation_growth_rate,
                 ):
        super().__init__()
        self.token_num = token_num
        # Fusing
        fusing_layers = []
        layer_list = [in_dim*2, 1024, 1024]
        for i in range(1, len(layer_list)) :
            fusing_layers.append(nn.Linear(layer_list[i-1], layer_list[i]))
            fusing_layers.append(nn.ReLU())
        fusing_layers.append(FCBlock(1024, hid_dim*token_num))
        self.fusing = nn.Sequential(*fusing_layers)

        # Encoder
        stride = 2
        encoder_layers = []
        filter, pad = stride * 2, stride // 2

        for _ in range(up_sample_rate):
            encoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            encoder_layers.append(nn.Conv1d(hid_dim, hid_dim, 3, 1, 1))
            encoder_layers.append(nn.ReLU())

        for i in range(down_sample_rate):
            input_dim = hid_dim
            block = nn.Sequential(
                nn.Conv1d(input_dim, hid_dim, filter, stride, pad),
                Resnet1D(hid_dim, res_depth, dilation_growth_rate, activation='relu', norm=False),
            )
            encoder_layers.append(block)
        
        encoder_layers.append(nn.Conv1d(hid_dim, out_dim, 3, 1, 1))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x_curr, x_past):
        """
        x : [B, dim]
        """
        B = x_curr.shape[0]

        x_post = torch.cat([x_past, x_curr], dim=1)                 # [B, 2dim]
        x_post = self.fusing(x_post).reshape(B, -1, self.token_num) # [B, dim, N]

        x = self.encoder(x_post)
        return x
    
class Decoder_v2(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_tokens,
                 div_rate,
                 down_sample_rate,
                 res_depth,
                 dilation_growth_rate,
                 ) :
        super().__init__()
        self.token_num = num_tokens
        print(f"# of tokens -- > {num_tokens}")

        decoder_layers = []
        decoder_layers.append(nn.Conv1d(in_dim, hid_dim, 3, 1, 1))
        decoder_layers.append(nn.ReLU())

        for i in list(np.linspace(1, num_tokens, div_rate, endpoint=False, dtype=int)[::-1]):
            decoder_layers.append(nn.Upsample(i))
            decoder_layers.append(nn.Conv1d(hid_dim, hid_dim, 3, 1, 1))
            decoder_layers.append(nn.ReLU())
        
        for i in range(down_sample_rate):
            input_dim = hid_dim
            block = nn.Sequential(
                Resnet1D(hid_dim, res_depth, dilation_growth_rate, reverse_dilation=True, activation='relu', norm=False),
                nn.Conv1d(hid_dim, input_dim, 3, 1, 1)
            )
            decoder_layers.append(block)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x_trans):
        """
        x_trans : [B, code_dim, N]
        """
        B = x_trans.shape[0]
        x_trans = self.decoder(x_trans)     # [B, d, 1]
        x_trans = x_trans.reshape(B, -1)    # [B, dim]
        
        return x_trans
    
class CrossAtten(nn.Module):
    def __init__(self):
        super().__init__()
        latent_dim = 512
        num_heads = 4
        ff_size = 1024
        dropout = 0.1
        activation = "gelu"
        num_layers = 2

        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation=activation,
                                                          batch_first=True)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=num_layers)

    def forward(self, z, x_quant):
        z = z[None].permute(1, 0, 2)
        x_quant = x_quant.permute(0, 2, 1)    # [B, N, dim]
        output = self.seqTransDecoder(tgt=x_quant, memory=z)
        output = output.permute(0, 2, 1)
        return output

from .rnn import RNN
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
        self.token_num = token_num = 3
        down_sample_rate = 2
        self.input_dict = {#'orient_q': 4, 
                           'orient_aa': 3,
                           'trans': 3}
        in_dim = sum(v for v in self.input_dict.values())

        """ Input mlp"""
        self.input_layer = MLPBlock(in_dim, hid_dim, hid_dim)

        """ Fusing mlp """
        fusing_layers = []
        layer_list = [hid_dim+con_dim, hid_dim, hid_dim]
        for i in range(1, len(layer_list)) :
            fusing_layers.append(nn.Conv1d(layer_list[i-1], layer_list[i], 3, 1, 1))
            fusing_layers.append(nn.ReLU())
        fusing_layers.append(nn.Conv1d(layer_list[-1], token_num*hid_dim, 3, 1, 1))
        self.fusing = nn.Sequential(*fusing_layers)

        """ Conditional encoder """
        encoder_layers = []
        input_dim = hid_dim*2
        for i in range(down_sample_rate):
            block = nn.Sequential(
                nn.Conv1d(input_dim, hid_dim, 3, 1, 1),
                Resnet1D(hid_dim, 2, 3, activation='relu', norm=False),
            )
            input_dim = hid_dim
            encoder_layers.append(block)
        
        encoder_layers.append(nn.Conv1d(hid_dim, out_dim, 3, 1, 1))
        self.encoder = nn.Sequential(*encoder_layers)
        
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
        x = x.reshape(T, B, -1)
        x = self.input_layer(x)

        """ Fusing mlp """
        x = torch.cat([x, context], dim=-1) # [T, B, 256+512]
        x = x.permute(1, 2, 0)              # [B, dim, T]
        x = self.fusing(x).reshape(B, -1, self.token_num, T) # [T, B, N*dim]

        """ Conditional encoder """
        encoded_feat = []
        for t in range(1, T):
            x_past, x_curr = x[...,t-1], x[...,t]               # [B, dim, N]
            x_in = torch.cat([x_past, x_curr], dim=1)           # [B, 2*dim, N]
            x_enc = self.encoder(x_in)                          
            encoded_feat.append(x_enc)                          # [T-1, B, dim, 3]
        
        batch['encoded_feat'] = torch.cat(encoded_feat, dim=-1)  # [B, dim, 3*80]
        return batch
    
class TrajDecoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_tokens,
                 ) :
        super().__init__()
        div_rate = 1
        down_sample_rate = 2
        self.token_num = num_tokens
        print(f"# of tokens -- > {num_tokens}")

        """ Decoder """
        decoder_layers = []
        decoder_layers.append(nn.Conv1d(in_dim, hid_dim, 3, 1, 1))
        decoder_layers.append(nn.ReLU())
        
        decoder_layers.append(nn.Upsample(1))
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

    def rot_output(self, past_6d, trans_6d):
        t, B = past_6d.shape[:2]
        past = transforms.rotation_6d_to_matrix(past_6d)        # [B, 1, 3, 3]
        trans = transforms.rotation_6d_to_matrix(trans_6d)      # [B, 1, 3, 3]
        curr = torch.matmul(trans, past)

        return transforms.matrix_to_rotation_6d(curr).reshape(t, B, 1, 6)
    
    def forward(self, batch):
        T = batch['seqlen']
        B = batch['batch_size']
        x_quant = batch['quantized_feat']   # [T-1, B, dim, 3]
        x_quant = x_quant.reshape(B, -1, 3, T-1)

        x_dec_list = []
        for t in range(T-1):
            x_dec = self.decoder(x_quant[..., t])   # [B, dim, 1]
            x_dec_list.append(x_dec)
        x_dec = torch.cat(x_dec_list, dim=-1)       # [B, dim, T-1]
        x_dec = x_dec.permute(2, 0, 1)
        out_local_traj_tp = self.head(x_dec)            # [T-1, B, 1, 11]
        
        local_traj_tp = batch['local_traj_tp']             # [T, B, 1, 11]
        past_z = local_traj_tp[1:, ..., 2:3]               # [T-1, B, 1, 1]
        past_local_orient = local_traj_tp[1:, ..., 3:9]    # [T-1, B, 1, 6]

        out_local_traj_tp = torch.zeros_like(local_traj_tp[1:])
        
        out_local_traj_tp[..., 2:3] = past_z + out_local_traj_tp[..., 2:3]
        out_local_traj_tp[..., 3:9] = self.rot_output(past_local_orient, out_local_traj_tp[..., 3:9])
        
        out_trans_tp, out_orient_q = traj_local2global_heading(out_local_traj_tp, local_orient_type='6d', )
        out_local_traj = out_local_traj_tp.transpose(0, 1).contiguous()
        
        batch['out_local_traj_tp'] = out_local_traj_tp
        batch['out_trans_tp'] = out_trans_tp        # GT w_transl
        batch['out_orient_q_tp'] = out_orient_q     # w_orient_q_tp
        out_orient_tp = transforms.quaternion_to_matrix(out_orient_q)
        batch['out_orient_6d_tp'] = transforms.matrix_to_rotation_6d(out_orient_tp)
        batch['local_orient'] = out_local_traj[..., 3:9]
        batch['d_heading_vec'] = out_local_traj[..., 9:]

        return batch