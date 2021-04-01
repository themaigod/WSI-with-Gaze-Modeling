import sys
import os
import argparse
import logging
import json
import time


import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from wsi.data.wsi_producer import GridWSIPatchDataset  # noqa
from wsi.model import MODELS  # noqa


parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
parser.add_argument('--wsi_path', default="/home/omnisky/ajmq/slideclassify/data_no_process", metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--ckpt_path', default="/home/omnisky/ajmq/slideclassify/ncrf_save/best.ckpt", metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('--cfg_path', default="/home/omnisky/ajmq/slideclassify/ncrf_save/cfg.json", metavar='CFG_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')
parser.add_argument('--mask_path', default="/home/omnisky/ajmq/slideclassify/npy_ncrf", metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input WSI file')
parser.add_argument('--probs_map_path', default="/home/omnisky/ajmq/slideclassify/probmap_ncrf", metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--GPU', default='1', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=5, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                    ' of the 8 direction predictions for each patch,'
                    ' default 0, which means disabled')


def get_probs_map(model, dataloader):
    probs_map = np.zeros(dataloader.dataset._mask.shape)
    print(probs_map.shape)
    num_batch = len(dataloader)
    # only use the prediction of the center patch within the grid
    idx_center = dataloader.dataset._grid_size // 2

    count = 0
    time_now = time.time()
    for (data, x_mask, y_mask) in dataloader:
        data = Variable(data.cuda(non_blocking=True), volatile=True)
        output = model(data)
        # because of torch.squeeze at the end of forward in resnet.py, if the
        # len of dim_0 (batch_size) of data is 1, then output removes this dim.
        # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
        if len(output.shape) == 1:
            probs = output[idx_center].sigmoid().cpu().data.numpy().flatten()
        else:
            probs = output[:,
                           idx_center].sigmoid().cpu().data.numpy().flatten()
        probs_map[x_mask, y_mask] = probs
        count += 1

        time_spent = time.time() - time_now
        time_now = time.time()
        logging.info(
            '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                dataloader.dataset._rotate, count, num_batch, time_spent))

    return probs_map


def make_dataloader(args, cfg, wsi, mask, flip='NONE', rotate='NONE'):
    batch_size = cfg['batch_size'] * 2
    num_workers = args.num_workers

    dataloader = DataLoader(
        GridWSIPatchDataset(wsi, mask,
                            image_size=cfg['image_size'],
                            patch_size=cfg['patch_size'],
                            crop_size=cfg['crop_size'], normalize=True,
                            flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if cfg['image_size'] % cfg['patch_size'] != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(cfg['image_size'], cfg['patch_size']))

    patch_per_side = cfg['image_size'] // cfg['patch_size']
    grid_size = patch_per_side * patch_per_side

    for filename in os.listdir(args.wsi_path):
        mask = np.load(os.path.join(args.mask_path,filename+".npy"))
        mask_path = os.path.join(args.mask_path,filename+".npy")
        wsi_path = os.path.join(args.wsi_path, filename)
        ckpt = torch.load(args.ckpt_path)
        model = MODELS[cfg['model']](num_nodes=grid_size, use_crf=cfg['use_crf'])
        model.load_state_dict(ckpt['state_dict'])
        model = model.cuda().eval()

        if not args.eight_avg:
            dataloader = make_dataloader(
                args, cfg, wsi_path, mask_path, flip='NONE', rotate='NONE')
            probs_map = get_probs_map(model, dataloader)
        else:
            probs_map = np.zeros(mask.shape)

            dataloader = make_dataloader(
                args, cfg, wsi_path, mask_path, flip='NONE', rotate='NONE')
            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(
                args, cfg, wsi_path, mask_path, flip='NONE', rotate='ROTATE_90')
            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(
                args, cfg, wsi_path, mask_path, flip='NONE', rotate='ROTATE_180')
            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(
                args, cfg, wsi_path, mask_path, flip='NONE', rotate='ROTATE_270')
            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(
                args, cfg, wsi_path, mask_path, flip='FLIP_LEFT_RIGHT', rotate='NONE')
            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(
                args, cfg, wsi_path, mask_path, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_90')
            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(
                args, cfg, wsi_path, mask_path, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_180')
            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(
                args, cfg, wsi_path, mask_path, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_270')
            probs_map += get_probs_map(model, dataloader)

            probs_map /= 8

        np.save(os.path.join(args.probs_map_path,filename+".npy"), probs_map)




def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
