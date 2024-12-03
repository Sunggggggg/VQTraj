from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import torch
import joblib

from configs import constants as _C
from ...utils import transforms
from ...utils import data_utils as d_utils
from ...utils.kp_utils import root_centering

FPS = 30

def create_chunks(n, chunk_size=82):
    numbers = list(range(n + 1))
    chunks = [numbers[i:i + chunk_size] for i in range(0, len(numbers), chunk_size) if len(numbers[i:i + chunk_size]) == chunk_size]
    return chunks

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(EvalDataset, self).__init__()
        eval_label = '/mnt2/SKY/WHAM/dataset/trans_data/KIT_12_RightTurn10_poses.pt'
        labels = joblib.load(eval_label)

        self.root_pose = labels['root_pose']
        self.body_pose = labels['body_pose']
        self.transl = labels['trans']
        self.betas = labels['shape']

        N = len(self.body_pose)
        self.idx_list = create_chunks(N, cfg.DATASET.SEQLEN)

    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, index):
        idx_list = self.idx_list[index]

        root_pose = self.root_pose[idx_list]
        body_pose = self.body_pose[idx_list]
        transl = self.transl[idx_list]
        betas = self.betas[idx_list]

        root_pose = torch.from_numpy(root_pose)
        root_pose = transforms.axis_angle_to_matrix(root_pose.reshape(-1, 1, 3))

        body_pose = torch.from_numpy(body_pose)
        body_pose = transforms.axis_angle_to_matrix(body_pose.reshape(-1, 23, 3))
        
        target = {'w_root_orient': root_pose,
                  'body_pose': body_pose,
                  'w_transl': transl, 
                  'betas': betas}

        return target

    