import torch

from progress.bar import Bar
from configs.config import parse_args
from lib.utils.train_utils import prepare_batch
from lib.data.datasets.dataset_eval import EvalDataset
from lib.models.vq_traj import Network
#from lib.models.TransVQTraj.trans_vq_traj import TransNetwork
from lib.utils.visualization import traj_vis_different_view, traj_vis_different_traj


def main(cfg):
    # ========= Dataloaders ========= #
    eval_dataset = EvalDataset(cfg)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=4,
        shuffle=True,
        pin_memory=True
    )

    # ========= Network and Optimizer ========= #
    network = Network(cfg).cuda()
    CHECKPOINT = '/mnt2/SKY/VQTraj/experiments/codebook/checkpoint.pth.tar'
    checkpoint = torch.load(CHECKPOINT)
    model_state_dict = {k: v for k, v in checkpoint['model'].items()}
    network.load_state_dict(model_state_dict, strict=True)

    bar = Bar('Validation', fill='#', max=len(eval_dataloader))
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            batch = prepare_batch(batch)
            pred = network(batch)
            
            pred_traj = pred['out_trans_tp'][:, 0].detach().cpu().numpy()
            target_traj = pred['w_transl_tp'][:, 0].detach().cpu().numpy()

            print("Traj`s shape : ", pred_traj.shape)
            traj_vis_different_traj(pred_traj, target_traj)

            break
    

if __name__ == '__main__':
    cfg, cfg_file, _ = parse_args()
    main(cfg)