import torch
import pprint
import random
import numpy as np

from configs.config import parse_args
from lib.utils.train_utils import prepare_batch
from lib.utils.print_utils import count_param
from lib.data.datasets.dataset_eval import EvalDataset
from lib.models.clip_vq_traj import Network
#from lib.models.vq_traj import Network


def main(cfg):
    # ========= Dataloaders ========= #
    eval_dataset = EvalDataset(cfg)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
        batch_size=3,
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )

    # ========= Network and Optimizer ========= #
    network = Network(cfg).cuda()
    print(count_param(network)) # 12,436,171

    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            batch = prepare_batch(batch)
            pred = network(batch)
            
            pred_traj = pred['out_trans_tp'][:, 0].detach().cpu().numpy()
            target_traj = pred['w_transl_tp'][:, 0].detach().cpu().numpy()

            print("Traj`s shape : ", pred_traj.shape)

            break

if __name__ == '__main__':
    cfg, cfg_file, _ = parse_args()

    main(cfg)