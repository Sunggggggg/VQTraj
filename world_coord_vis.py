import torch
import numpy as np
from os import path as osp
import imageio
from progress.bar import Bar
from configs.config import parse_args
from lib.utils.train_utils import prepare_batch
from lib.data.datasets.dataset_eval import EvalDataset
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

    return verts_glob

def main(cfg):
    # ========= Dataloaders ========= #
    eval_dataset = EvalDataset(cfg)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )

    width = height = 1000
    focal_length = (width ** 2 + height ** 2) ** 0.5
    fps = 30.
    output_pth = './'
    n_frames = 81

    # ========= Network and Optimizer ========= #
    #from lib.models.GLAMR.network import Network
    #   from lib.models.GLAMR.network_clip import Network
    from lib.models.HuMoR.network import Network
    network = Network().cuda()
    CHECKPOINT = '/mnt2/SKY/VQTraj/experiments/vae_trans/model_best.pth.tar'
    checkpoint = torch.load(CHECKPOINT)
    model_state_dict = {k: v for k, v in checkpoint['model'].items()}
    network.load_state_dict(model_state_dict, strict=False)
    smpl = build_body_model('cuda', n_frames)
    
    pred_verts_glob_list, gt_verts_glob_list = [], []
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            batch = prepare_batch(batch)
            body_param = network(batch)
            pred_verts_glob = forward_smpl(smpl, body_param, False)
            gt_verts_glob = forward_smpl(smpl, body_param, True)
            pred_verts_glob_list.append(pred_verts_glob)
            gt_verts_glob_list.append(gt_verts_glob)
    
    pred_verts_glob = torch.cat(pred_verts_glob_list); verts_glob = torch.cat(gt_verts_glob_list)

    verts_glob[..., 1] = verts_glob[..., 1] - verts_glob[..., 1].min()
    cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
    sx, sz = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]]
    scale = max(sx.item(), sz.item()) * 1.5

    renderer = Renderer(width, height, focal_length, 'cuda', smpl.faces)
    renderer.set_ground(scale, cx.item(), cz.item())
    global_R, global_T, global_lights = get_global_cameras(verts_glob, 'cuda', position=(5, 5, 0))
    default_R, default_T = torch.eye(3), torch.zeros(3)

    writer = imageio.get_writer(
        osp.join(output_pth, 'Human_centric.mp4'), 
        fps=fps, mode='I', format='FFMPEG', macro_block_size=1
    )
    bar = Bar('Rendering results ...', fill='#', max=len(verts_glob))

    frame_i = 0
    _global_R, _global_T = None, None
    plattle = {
        'w': torch.ones((1, 4)).float().to('cuda'),
        'r': torch.Tensor((0.7, 0., 0., 1.)).unsqueeze(0).float().to('cuda'),
        'g': torch.Tensor((0., 1., 0., 1.)).unsqueeze(0).float().to('cuda'),
        'b': torch.Tensor((0., 0., 0.7, 1.)).unsqueeze(0).float().to('cuda')
    }
    # run rendering
    for frame_i in range(len(verts_glob)):
        if frame_i == 300 : break
        verts = verts_glob[[frame_i]].to('cuda')
        pred_verts = pred_verts_glob[[frame_i]].to('cuda')
        faces = renderer.faces.clone().squeeze(0).to('cuda')
        colors = plattle['r']

        if _global_R is None:
            _global_R = global_R[frame_i].clone(); _global_T = global_T[frame_i].clone()

        cameras = renderer.create_camera(global_R[frame_i], global_T[frame_i])
        img_glob = renderer.render_with_ground_multimesh(verts, pred_verts, faces, plattle['b'], plattle['r'], cameras, global_lights)
        
        writer.append_data(img_glob)
        bar.next()
    writer.close()

    

if __name__ == '__main__':
    cfg, cfg_file, _ = parse_args()
    main(cfg)