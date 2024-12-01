from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import joblib
import numpy as np

from .._dataset import BaseDataset
from ..utils.augmentor import *
from ...utils import data_utils as d_utils
from ...utils import transforms
from ...models import build_body_model


class Dataset3D(BaseDataset):
    def __init__(self, cfg, fname, training):
        super(Dataset3D, self).__init__(cfg, training)

        self.epoch = 0
        self.labels = joblib.load(fname)
        self.n_frames = cfg.DATASET.SEQLEN + 1

        if self.training:
            self.prepare_video_batch()

        self.smpl = build_body_model('cpu', self.n_frames)
        self.SMPLAugmentor = SMPLAugmentor(cfg, False)

    def __getitem__(self, index):
        return self.get_single_sequence(index)
    
    def get_labels(self, index, target):
        start_index, end_index = self.video_indices[index]
        
        # SMPL parameters
        # NOTE: We use NeuralAnnot labels for Human36m and MPII3D only for the 0th frame input.
        #       We do not supervise the network on SMPL parameters.
        pose = transforms.axis_angle_to_matrix(
            self.labels['pose'][start_index:end_index+1].clone().reshape(-1, 24, 3))
        root_orient = pose[:, :1]   # [T, 1, 3, 3]
        body_pose = pose[:, 1:]     # [T, 23, 3, 3]

        target['w_root_orient'] = root_orient
        target['body_pose'] = body_pose
        target['betas'] = self.labels['betas'][start_index:end_index+1].clone()        # No t
        target['w_transl'] = torch.zeros((target['betas'].shape[0], 3))
        # Apply SMPL augmentor (y-axis rotation and initial frame noise)
        target = self.SMPLAugmentor(target)

        return target

    def get_single_sequence(self, index):
        # Universal target
        target = {}

        target = self.get_labels(index, target)
        return target