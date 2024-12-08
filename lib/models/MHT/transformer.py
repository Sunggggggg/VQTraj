import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)    # 
        pe[:, 1::2] = torch.cos(position * div_term)    # 
        pe = pe.unsqueeze(1)                            # [5000, 1, dim]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        x : [T, B, dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CausalTransformer(nn.Module):
    def __init__(self, d_model, nhead, nlayers):
        super(CausalTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        
        self.pe = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, 
                                                   dim_feedforward=int(d_model*2.), dropout=0.1,
                                                   activation='gelu', batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
    
    def forward(self, x):
        T = x.shape[0]

        x = self.pe(x)  # [T, B, dim]
        src_mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(x.device)

        out = self.encoder(x, mask=src_mask)    
        return out

if __name__ == '__main__':
    # Example usage
    model = TransformerModel(d_model=512, nhead=8, nlayers=6).cuda()

    # Input data
    seq_len, batch_size, d_model = 10, 32, 512
    src = torch.randn(seq_len, batch_size, d_model).cuda()

    output = model(src)
    print(output.shape)