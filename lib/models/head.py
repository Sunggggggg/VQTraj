import torch
import torch.nn as nn

class OutHead(nn.Module):
    def __init__(self, in_dim,  hid_dims=[512, 128], out_dim=11) :
        super().__init__()
        fc_layer = []
        dim_list = [in_dim] + hid_dims
        for i in range(1, len(dim_list)) :
            fc_layer.append(nn.Linear(dim_list[i-1], dim_list[i]))
            fc_layer.append(nn.ReLU())
        self.fc_layer = nn.Sequential(*fc_layer)

        self.output_head = nn.Linear(dim_list[-1], out_dim)

    def forward(self, batch, return_dict=False):
        """
        decoded_feat : [T, B, dim]
        """
        x = self.fc_layer(batch['decoded_feat'].unsqueeze(-2))
        out = self.output_head(x)   # [B]

        d_xy = out[..., :2]
        z = out[..., 2:3]
        local_orient = out[..., 3:9]
        d_heading_vec = out[..., 9:]

        batch['d_xy'] = d_xy
        batch['z'] = z
        batch['local_orient'] = local_orient
        batch['d_heading_vec'] = d_heading_vec
        
        if return_dict :
            return batch
        else :
            return out