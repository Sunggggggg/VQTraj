import numpy as np
import torch
import torch.nn as nn
import joblib
from lib.utils import transforms
from lib.data._dataset import BaseDataset
from lib.data.utils.augmentor import *

from configs import constants as _C

def compute_contact_label(feet, thr=1e-2, alpha=5):
    vel = torch.zeros_like(feet[..., 0])
    label = torch.zeros_like(feet[..., 0])
    
    vel[1:-1] = (feet[2:] - feet[:-2]).norm(dim=-1) / 2.0
    vel[0] = vel[1].clone()
    vel[-1] = vel[-2].clone()
    
    label = 1 / (1 + torch.exp(alpha * (thr ** -1) * (vel - thr)))
    return label


class AMASSDataset(BaseDataset):
    def __init__(self, cfg) :
        label_pth = _C.PATHS.AMASS_LABEL
        super(AMASSDataset, self).__init__(cfg, training=True)

        self.labels = joblib.load(label_pth)
        self.SMPLAugmentor = SMPLAugmentor(cfg)

        self.prepare_video_batch()
    
    @property
    def __name__(self, ):
        return 'AMASS'
    
    def generate_mask(self, target):
        """
        visble = True, 
        """
        frame_mask = np.random.rand(target['body_pose'].shape[0]).astype(np.float32) < 0.03 # [T]
        target['mask'] = frame_mask
        return target

    def load_amass(self, index, target):
        """
        World coordinate / 
        """
        start_index, end_index = self.video_indices[index]
        
        # Load AMASS labels
        pose = torch.from_numpy(self.labels['pose'][start_index:end_index].copy())
        pose = transforms.axis_angle_to_matrix(pose.reshape(-1, 24, 3))
        root_orient = pose[:, :1]   # [T, 1, 3, 3]
        body_pose = pose[:, 1:]     # [T, 23, 3, 3]

        transl = torch.from_numpy(self.labels['transl'][start_index:end_index].copy())
        betas = torch.from_numpy(self.labels['betas'][start_index:end_index].copy())
        
        # Stack GT
        target.update({'vid': self.labels['vid'][start_index], 
                       'w_root_orient': root_orient, 
                       'body_pose': body_pose, 
                       'w_transl': transl, 
                       'betas': betas})

        return target

    def augment_data(self, target):
        # Augmentation 1. SMPL params augmentation
        target = self.SMPLAugmentor(target)
        
        return target


    def get_single_sequence(self, index):
        target = {}
        target = self.load_amass(index, target)
        target = self.augment_data(target)
        target = self.generate_mask(target)

        return target