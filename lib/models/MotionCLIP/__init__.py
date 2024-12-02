import torch
import torch.nn as nn
from .layers import Encoder_TRANSFORMER, Decoder_TRANSFORMER

class MotionCLIP(nn.Module):
    def __init__(self, encoder, decoder, device, latent_dim,
                 pose_rep, glob, glob_rot, translation, jointstype, vertstrans, **kwargs):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
    
    def make_mask(self, batch):
        x = batch['coco']
        B, J, d, T = x.shape
        mask = torch.ones((B, T), dtype=bool, device=x.device)
        y = torch.ones((B), dtype=int, device=x.device)
        batch['y'] = y
        batch['mask'] = mask
        return batch

    def forward(self, batch):
        """
        batch["coco"]   : [B, J, 3, T] (축변환해줘야함)
        batch["y"]      : [B]
        batch["mask"]   : [B, T]
        """
        batch = self.make_mask(batch)
        batch.update(self.encoder(batch))

        batch["z"] = batch["mu"]
        return batch

def get_model():
    parameters = {}
    ## Encoder
    parameters['modeltype'] = 'motionclip'
    parameters['njoints'] = 17
    parameters['nfeats'] = 3
    parameters['num_frames'] = 81
    parameters['num_classes'] = 1000
    parameters['translation'] = True
    parameters['pose_rep'] = 'rot6d'
    parameters['glob'] = True
    parameters['glob_rot'] = [3.141592653589793, 0, 0]
    parameters['latent_dim'] = 512
    parameters['ff_size'] = 1024
    parameters['num_layers'] = 8
    parameters['num_heads'] = 4
    parameters['dropout'] = 0.1
    parameters['ablation'] = None
    parameters['activation'] = "gelu"
    parameters['vertstrans'] = False


    encoder = Encoder_TRANSFORMER(**parameters)
    decoder = Decoder_TRANSFORMER(**parameters)

    parameters['device'] = 'cuda'    
    parameters['jointstype'] = 'coco'

    model = MotionCLIP(encoder, decoder, **parameters).to(parameters["device"])
    loader_weight = torch.load('/mnt2/SKY/MotionCLIP/exps/81_frame_no_transl/checkpoint_0100.pth.tar')
    model.load_state_dict(loader_weight, strict=True)
    
    for param in model.parameters():
        param.requires_grad = False

    return model 