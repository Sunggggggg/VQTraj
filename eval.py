import torch
import numpy as np
from os import path as osp
import imageio
from progress.bar import Bar
from configs.config import parse_args
from lib.utils.train_utils import prepare_batch
from lib.data.datasets.dataset_eval import EvalDataset
from lib.models.vq_traj import Network
#from lib.models.clip_vq_traj import Network
#from lib.models.TransVQTraj.trans_vq_traj import TransNetwork
from lib.utils.visualization import traj_vis_different_view, traj_vis_different_traj
from lib.utils import transforms
from lib.models import build_body_model
from lib.vis.renderer import Renderer, get_global_cameras

def forward_smpl(smpl, body_param, use_GT=False):
    yup2ydown = transforms.axis_angle_to_matrix(torch.tensor([np.pi, 0, 0])).float().to('cuda')  # [1, 3]
    
    
    if use_GT :
        global_orient = yup2ydown.mT @ body_param['w_root_orient'].reshape(-1, 1, 3, 3)    # [1000, 3]
        trans_world = (yup2ydown.mT @ body_param['w_transl'].reshape(-1, 3).unsqueeze(-1)).squeeze(-1)
        global_output = smpl.get_output(
                body_pose=body_param['body_pose'].reshape(-1, 23, 3, 3), 
                global_orient=global_orient,
                betas=body_param['betas'].reshape(-1, 10),
                transl=trans_world,
                pose2rot=False)
    else :
        global_orient = yup2ydown.mT @ body_param['out_orient'].reshape(-1, 1, 3, 3)    # [1000, 3]
        trans_world = (yup2ydown.mT @ body_param['out_trans'].reshape(-1, 3).unsqueeze(-1)).squeeze(-1)
        global_output = smpl.get_output(
                body_pose=body_param['body_pose'].reshape(-1, 23, 3, 3), 
                global_orient=global_orient,
                betas=body_param['betas'].reshape(-1, 10),
                transl=trans_world,
                pose2rot=False)
    
    verts_glob = global_output.vertices.cpu()
    verts_glob[..., 1] = verts_glob[..., 1] - verts_glob[..., 1].min()
    cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
    sx, sz = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]]
    scale = max(sx.item(), sz.item()) * 1.5

    return verts_glob, (cx, cz), scale



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
    network = Network().cuda()
    CHECKPOINT = '/mnt2/SKY/VQTraj/experiments/codebook/model_best.pth.tar'
    checkpoint = torch.load(CHECKPOINT)
    model_state_dict = {k: v for k, v in checkpoint['model'].items()}
    network.load_state_dict(model_state_dict, strict=False)
    smpl = build_body_model('cuda', n_frames)
    
    mpvpe = 0
    num_data = 0
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            batch = prepare_batch(batch)
            body_param = network(batch)
            gt_verts_glob, (cx, cz), scale = forward_smpl(smpl, body_param, True)       # [81, J, 3]
            pred_verts_glob, (cx, cz), scale = forward_smpl(smpl, body_param, False)
            diff = pred_verts_glob - gt_verts_glob
            dist = torch.norm(diff, dim=2)
            mpvpe += dist.mean(dim=1).sum() * 1000
            num_data += diff.shape[0]

    mpvpe /= num_data
    print(mpvpe.item()) 

    

if __name__ == '__main__':
    cfg, cfg_file, _ = parse_args()
    main(cfg)