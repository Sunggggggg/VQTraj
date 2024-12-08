import torch
import pprint
import random
import numpy as np

from configs.config import parse_args
from lib.utils.train_utils import prepare_batch
from lib.utils.print_utils import count_param, print_batch
from lib.data.datasets.dataset_eval import EvalDataset
#from lib.models.TransVQTraj.trans_vq_traj import TransNetwork

def main(cfg):
    # ========= Dataloaders ========= #
    eval_dataset = EvalDataset(cfg)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
        batch_size=5,
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )

    # ========= Network and Optimizer ========= #
    # from lib.models.clip_vq_traj import Network
    #from lib.models.GLAMR.network_clip import Network
    # from lib.models.vq_traj_mask import Network
    from lib.models.MHT.network import Network

    network = Network(stage='stage1').cuda()
    print(count_param(network)) #  3,254,562

    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            batch = prepare_batch(batch)
            print_batch(pred)
            pred = network(batch)

            print_batch(pred)
            
            pred_traj = pred['out_trans_tp'][:, 0].detach().cpu().numpy()
            target_traj = pred['w_transl_tp'][:, 0].detach().cpu().numpy()

            print("Traj`s shape : ", pred_traj.shape)

            break

if __name__ == '__main__':
    cfg, cfg_file, _ = parse_args()

    main(cfg)