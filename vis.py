import torch
import numpy as np
from os import path as osp
import imageio
from progress.bar import Bar
from configs.config import parse_args
from lib.utils.train_utils import prepare_batch
from lib.data.datasets.dataset_eval import EvalDataset
from lib.models.GLAMR.network import Network
#from lib.models.clip_vq_traj import Network
#from lib.models.TransVQTraj.trans_vq_traj import TransNetwork
from lib.utils.visualization import traj_vis_different_view, traj_vis_different_traj
from lib.utils import transforms
from lib.models import build_body_model
from lib.vis.renderer import Renderer, get_global_cameras

import matplotlib.pyplot as plt

def main(cfg):
    # ========= Dataloaders ========= #
    eval_dataset = EvalDataset(cfg)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        pin_memory=True
    )

    width = height = 1000
    focal_length = (width ** 2 + height ** 2) ** 0.5
    fps = 30.
    output_pth = './'
    n_frames = 81

    # ========= Network and Optimizer ========= #
    #from lib.models.GLAMR.network import Network
    from lib.models.GLAMR.network_clip import Network
    network = Network().cuda()
    CHECKPOINT = '/mnt2/SKY/VQTraj/experiments/vae_clip/checkpoint.pth.tar'
    checkpoint = torch.load(CHECKPOINT)
    model_state_dict = {k: v for k, v in checkpoint['model'].items()}
    network.load_state_dict(model_state_dict, strict=False)
    smpl = build_body_model('cuda', n_frames)
    
    pred_traj_list = []
    target_traj_list = []
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            batch = prepare_batch(batch)
            pred = network(batch)
            
            pred_traj = pred['out_trans_tp'][:, 0].detach().cpu().numpy()
            target_traj = pred['w_transl_tp'][:, 0].detach().cpu().numpy() 

            pred_traj = pred_traj - pred_traj[:1]
            target_traj = target_traj - target_traj[:1]

            print("Traj`s shape : ", pred_traj.shape)
            pred_traj_list.append(pred_traj)
            target_traj_list.append(target_traj)

            break
    
    pred_traj = np.concatenate(pred_traj_list)
    target_traj = np.concatenate(target_traj_list)
    print(pred_traj.shape)
    traj_vis_different_traj(pred_traj, target_traj)
    
    plt.close()

if __name__ == '__main__':
    cfg, cfg_file, _ = parse_args()
    main(cfg)