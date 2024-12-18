from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np

from .amass import AMASSDataset

class DataFactory(torch.utils.data.Dataset):
    def __init__(self, cfg, train_stage):
        super(DataFactory, self).__init__()

        if train_stage == 'Traj':
            self.datasets = [AMASSDataset(cfg)]
            self.dataset_names = ['AMASS']
        elif train_stage == 'stage1':
            self.datasets = [AMASSDataset(cfg)]
            self.dataset_names = ['AMASS']
        elif train_stage == 'stage2':
            pass

        self._set_partition(cfg.DATASET.RATIO)
        self.lengths = [len(ds) for ds in self.datasets]

    @property
    def __name__(self, ):
        return 'MixedData'

    def prepare_video_batch(self):
        [ds.prepare_video_batch() for ds in self.datasets]
        self.lengths = [len(ds) for ds in self.datasets]

    def _set_partition(self, partition):
        self.partition = partition
        self.ratio = partition
        self.partition = np.array(self.partition).cumsum()
        self.partition /= self.partition[-1]

    def __len__(self):
        return int(np.array([l for l, r in zip(self.lengths, self.ratio) if r > 0]).mean())

    def __getitem__(self, index):
        # Get the dataset to sample from
        p = np.random.rand()
        for i in range(len(self.datasets)):
            if p <= self.partition[i]:
                if len(self.datasets) == 1:
                    return self.datasets[i][index % self.lengths[i]]
                else:
                    d_index = np.random.randint(0, self.lengths[i])
                    return self.datasets[i][d_index]