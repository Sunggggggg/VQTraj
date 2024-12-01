import torch
from configs.config import parse_args
from lib.utils.print_utils import print_batch
from lib.utils.train_utils import prepare_batch
from lib.data.datasets.amass import AMASSDataset
from lib.models.vq_traj import Network

def main(cfg):
    amass_dataset = AMASSDataset(cfg)
    data_loader = torch.utils.data.DataLoader(amass_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=4,
        shuffle=True,
        pin_memory=True
    )

    model = Network(cfg).cuda()

    for idx, batch in enumerate(data_loader):
        batch = prepare_batch(batch)
        model(batch)

        print_batch(batch)
            

        break

if __name__ == '__main__':
    cfg, cfg_file, _ = parse_args()
    
    main(cfg)