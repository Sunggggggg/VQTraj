import os, sys
from configs import constants as _C
from .layers import Encoder, Decoder
from .head import OutHead
from .codebook import QuantizeEMAReset
from .smpl import SMPL

def build_body_model(device, batch_size=1, gender='neutral', **kwargs):
    sys.stdout = open(os.devnull, 'w')
    body_model = SMPL(
        model_path=_C.BMODEL.FLDR,
        gender=gender,
        batch_size=batch_size,
        create_transl=False).to(device)
    sys.stdout = sys.__stdout__
    return body_model

def build_network(cfg):
    from .vq_traj import Network
    return