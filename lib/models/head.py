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

    def forward(self, x):
        """
        decoded_feat : [T, B, dim]
        """
        x = self.fc_layer(x.unsqueeze(-2))
        out = self.output_head(x)   # [B]

        return out