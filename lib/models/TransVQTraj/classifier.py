import torch
import torch.nn as nn
from ..layers import *

class TokenClassifier(nn.Module):
    def __init__(self,
                 num_blocks,
                 in_dim,
                 hid_dim,
                 hid_inter_dim,
                 token_num,
                 token_inter_dim,
                 class_num=None) :
        super().__init__()
        self.token_num = token_num
        self.mixer_trans = FCBlock(in_dim, token_num*hid_dim)

        mixer_head_list = [MixerLayer(hid_dim, hid_inter_dim, token_num, token_inter_dim, 0.) for _ in range(num_blocks)]
        self.mixer_head = nn.Sequential(*mixer_head_list)
        
        self.mixer_norm_layer = FCBlock(hid_dim, hid_dim)
        
        self.class_num = class_num
        if class_num is None :
            self.class_pred_layer = nn.Identity()
        else :
            self.class_pred_layer = nn.Linear(hid_dim, class_num)
        
    def forward(self, x_past):
        """
        x_past : [B, J*dim]
        """
        B = x_past.shape[0]

        cls_feat = self.mixer_trans(x_past).reshape(B, self.token_num, -1) # [B, Token, dim]
        cls_feat = self.mixer_head(cls_feat)
        cls_feat = self.mixer_norm_layer(cls_feat)      # [B, T, dim]

        if self.class_num is None :
            return cls_feat
        else :
            cls_logits = self.class_pred_layer(cls_feat)    # [B, Token, Class]
            cls_logits_softmax = cls_logits.softmax(dim=-1) 
            return cls_logits_softmax