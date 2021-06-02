import sys
import os
import argparse
import logging
import json
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from wsi_producer import GridWSIPatchDataset  # noqa
from wsi.model import MODELS  # noqa
from run_process import *

parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                             ' patch predictions given a WSI')
parser.add_argument('--wsi_path', default="/home/omnisky/ajmq/patch_slide_relate/wsi5.28", metavar='WSI_PATH',
                    type=str,
                    help='Path to the input WSI file')
parser.add_argument('--ckpt_path', default="/home/omnisky/ajmq/patch_slide_relate/save5.28/best.ckpt",
                    metavar='CKPT_PATH',
                    type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('--cfg_path', default="/home/omnisky/ajmq/patch_slide_relate/save5.28/cfg.json", metavar='CFG_PATH',
                    type=str,
                    help='Path to the config file in json format related to'
                         ' the ckpt file')
parser.add_argument('--mask_path', default="/home/omnisky/ajmq/patch_slide_relate/mask", metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input WSI file')
parser.add_argument('--probs_map_path', default="/home/omnisky/ajmq/patch_slide_relate/probmap_5.28",
                    metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--GPU', default='0,1', type=str, help='which GPU to use'
                                                           ', default 0')
parser.add_argument('--num_workers', default=5, type=int, help='number of '
                                                               'workers to use to make batch, default 5')
parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                                                             ' of the 8 direction predictions for each patch,'
                                                             ' default 0, which means disabled')

path1 = "/home/omnisky/ajmq/patch_slide_relate/img5.28/grey"
path2 = "/home/omnisky/ajmq/patch_slide_relate/img5.28/two"
path3 = "/home/omnisky/ajmq/patch_slide_relate/img5.28/three"

restart = None


def get_probs_map(model, dataloader):
    probs_map = np.zeros(dataloader.dataset._mask.shape)
    sum_map = np.zeros(dataloader.dataset._mask.shape)
    pro_map = np.zeros(dataloader.dataset._mask.shape)
    p_map = np.zeros(dataloader.dataset._mask.shape)
    new_map = np.zeros(dataloader.dataset._mask.shape)
    r = 64 // dataloader.dataset._resolution
    print(r)
    [x_i, y_i] = dataloader.dataset._mask.shape
    print(probs_map.shape)
    num_batch = len(dataloader)
    # only use the prediction of the center patch within the grid
    # idx_center = dataloader.dataset._grid_size // 2
    grid = 1

    count = 0
    time_now = time.time()
    for (data, x_mask, y_mask) in dataloader:
        data = Variable(data.cuda(non_blocking=True))
        output = model(data)
        # output = torch.sigmoid(output)
        # because of torch.squeeze at the end of forward in resnet.py, if the
        # len of dim_0 (batch_size) of data is 1, then output removes this dim.
        # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
        for i in range(grid):
            if len(output.shape) == 1:
                probs = output[i].sigmoid().cpu().data.numpy().flatten()
            else:
                probs = output[:,
                        i].sigmoid().cpu().data.numpy().flatten()
            x = (i // 3) - 1
            y = (i % 3) - 1
            x_max = x_mask + r / 2 + x * r
            x_min = x_mask - r / 2 + x * r
            y_max = y_mask + r / 2 + y * r
            y_min = y_mask - r / 2 + y * r
            x_max = np.int_(np.array(x_max))
            x_min = np.int_(np.array(x_min))
            y_max = np.int_(np.array(y_max))
            y_min = np.int_(np.array(y_min))
            if not isinstance(p_map[x_mask, y_mask], np.float64):
                for j in range(len(x_max)):
                    if x_max[j] <= x_i and y_max[j] <= y_i and x_min[j] >= 0 and y_min[j] >= 0:
                        probs_map[x_min[j]: x_max[j], y_min[j]:y_max[j]] = probs_map[x_min[j]: x_max[j],
                                                                           y_min[j]:y_max[j]] + probs[j]
                        for x_r in range(x_min[j], x_max[j]):
                            for y_r in range(y_min[j], y_max[j]):
                                if sum_map[x_r, y_r] == 0:
                                    pro_map[x_r, y_r] = probs[j]
                        sum_map[x_min[j]: x_max[j], y_min[j]:y_max[j]] = sum_map[x_min[j]: x_max[j],
                                                                         y_min[j]:y_max[j]] + 1
            else:
                print([x_min, x_max, y_min, y_max])
                x_min = x_min[0]
                x_max = x_max[0]
                y_min = y_min[0]
                y_max = y_max[0]

                if x_max <= x_i and y_max <= y_i and x_min >= 0 and y_min >= 0:
                    probs_map[x_min: x_max, y_min:y_max] = probs_map[x_min: x_max,
                                                           y_min:y_max] + probs
                    for x_r in range(x_min, x_max):
                        for y_r in range(y_min, y_max):
                            if sum_map[x_r, y_r] == 0:
                                pro_map[x_r, y_r] = probs
                    sum_map[x_min: x_max, y_min:y_max] = sum_map[x_min: x_max,
                                                         y_min:y_max] + 1

        if len(output.shape) == 1:
            probs = output.sigmoid().cpu().data.numpy().flatten()
        else:
            probs = output[:].sigmoid().cpu().data.numpy().flatten()
        if len(output.shape) == 1:
            pr = torch.nn.functional.sigmoid(output).mean().cpu().data.numpy().flatten()
        else:
            pr = torch.nn.functional.sigmoid(output).mean(dim=1).cpu().data.numpy().flatten()
        p_map[x_mask, y_mask] = probs
        print(pr)
        if max(probs) > 0.2:
            print(6)
        new_map[x_mask, y_mask] = pr

        count += 1

        time_spent = time.time() - time_now
        time_now = time.time()
        logging.info(
            '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
                .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                dataloader.dataset._rotate, count, num_batch, time_spent))
    [x, y] = dataloader.dataset._mask.shape
    for x_id in range(x):
        for y_id in range(y):
            if sum_map[x_id, y_id] != 0:
                probs_map[x_id, y_id] = probs_map[x_id, y_id] / sum_map[x_id, y_id]

    return probs_map, pro_map, p_map, new_map


def make_dataloader(args, cfg, wsi, mask, flip='NONE', rotate='NONE'):
    batch_size = cfg['batch_size'] * 2
    num_workers = args.num_workers

    dataloader = DataLoader(
        GridWSIPatchDataset(wsi, mask, normalize=True,
                            flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def npy2jpg_single(img, filename, path_save1=path1, path_save2=path2, path_save3=path3):
    img2 = np.uint8(img * 255)
    img2 = img2.T
    img3 = img2.copy()
    img4 = img2.copy()
    img3[img3 < 127] = 0
    img3[img3 >= 127] = 255
    img4[(166 > img4) & (img4 > 84)] = 127
    img4[img4 >= 166] = 255
    img4[img4 <= 84] = 0
    cv2.imwrite(os.path.join(path_save1, filename + ".jpg"), img2)
    cv2.imwrite(os.path.join(path_save2, filename + ".jpg"), img3)
    cv2.imwrite(os.path.join(path_save3, filename + ".jpg"), img4)


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

    file_box = []

    if restart is not None:
        file_list = os.listdir(args.wsi_path)
        length = len(restart)
        file_dir = [i[:length] for i in file_list]
        file_index = file_dir.index(restart)
        print(file_index)
        file_box = file_list[:(file_index + 1)]

    for filename in os.listdir(args.wsi_path):
        if filename not in file_box:
            mask = np.load(os.path.join(args.mask_path, filename + ".npy"))
            mask_path = os.path.join(args.mask_path, filename + ".npy")
            wsi_path = os.path.join(args.wsi_path, filename)
            # ckpt = torch.load(args.ckpt_path)
            # model = MIL(key=cfg['model_mil'], pretrained=True)
            # mil = ckpt['state_dict_mil']
            # model.load_state_dict(mil)
            # # model = model.cuda().eval()
            # model = DataParallel(model, device_ids=None)
            # model = model.cuda()
            #
            # model.eval()
            model_crf, model_mil = produce_model_no_cuda(cfg)

            # load_weight(args, model_crf, model_mil)
            mil_dict = model_mil.state_dict()
            weight = torch.load(args.ckpt_path)
            model_dict = weight['state_dict_crf']
            stat_dict = {}
            for k, v in model_dict.items():
                new_k = "model" + k[6:]
                stat_dict[new_k] = v
            # stat_dict = {"model" + k[6:]: v for k, v in model_dict.items()}
            mil_dict.update(stat_dict)
            model_mil.load_state_dict(mil_dict)

            model_crf, model = cuda_model(model_crf, model_mil)
            model.eval()

            if not args.eight_avg:
                dataloader = make_dataloader(
                    args, cfg, wsi_path, mask_path, flip='NONE', rotate='NONE')
                probs_map, pro_map, p_map, new_map = get_probs_map(model, dataloader)
            else:
                probs_map = np.zeros(mask.shape)
                pro_map = np.zeros(mask.shape)
                p_map = np.zeros(mask.shape)
                new_map = np.zeros(mask.shape)

                dataloader = make_dataloader(
                    args, cfg, wsi_path, mask_path, flip='NONE', rotate='NONE')
                probs_map_sud, pro_map_sud, p_map_sud, new_map_sud = get_probs_map(model, dataloader)
                probs_map = probs_map + probs_map_sud
                pro_map = pro_map + pro_map_sud
                p_map = p_map + p_map_sud
                new_map = new_map + new_map_sud

                dataloader = make_dataloader(
                    args, cfg, wsi_path, mask_path, flip='NONE', rotate='ROTATE_90')
                probs_map_sud, pro_map_sud, p_map_sud, new_map_sud = get_probs_map(model, dataloader)
                probs_map = probs_map + probs_map_sud
                pro_map = pro_map + pro_map_sud
                p_map = p_map + p_map_sud
                new_map = new_map + new_map_sud

                dataloader = make_dataloader(
                    args, cfg, wsi_path, mask_path, flip='NONE', rotate='ROTATE_180')
                probs_map_sud, pro_map_sud, p_map_sud, new_map_sud = get_probs_map(model, dataloader)
                probs_map = probs_map + probs_map_sud
                pro_map = pro_map + pro_map_sud
                p_map = p_map + p_map_sud
                new_map = new_map + new_map_sud

                dataloader = make_dataloader(
                    args, cfg, wsi_path, mask_path, flip='NONE', rotate='ROTATE_270')
                probs_map_sud, pro_map_sud, p_map_sud, new_map_sud = get_probs_map(model, dataloader)
                probs_map = probs_map + probs_map_sud
                pro_map = pro_map + pro_map_sud
                p_map = p_map + p_map_sud
                new_map = new_map + new_map_sud

                dataloader = make_dataloader(
                    args, cfg, wsi_path, mask_path, flip='FLIP_LEFT_RIGHT', rotate='NONE')
                probs_map_sud, pro_map_sud, p_map_sud, new_map_sud = get_probs_map(model, dataloader)
                probs_map = probs_map + probs_map_sud
                pro_map = pro_map + pro_map_sud
                p_map = p_map + p_map_sud
                new_map = new_map + new_map_sud

                dataloader = make_dataloader(
                    args, cfg, wsi_path, mask_path, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_90')
                probs_map_sud, pro_map_sud, p_map_sud, new_map_sud = get_probs_map(model, dataloader)
                probs_map = probs_map + probs_map_sud
                pro_map = pro_map + pro_map_sud
                p_map = p_map + p_map_sud
                new_map = new_map + new_map_sud

                dataloader = make_dataloader(
                    args, cfg, wsi_path, mask_path, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_180')
                probs_map_sud, pro_map_sud, p_map_sud, new_map_sud = get_probs_map(model, dataloader)
                probs_map = probs_map + probs_map_sud
                pro_map = pro_map + pro_map_sud
                p_map = p_map + p_map_sud
                new_map = new_map + new_map_sud

                dataloader = make_dataloader(
                    args, cfg, wsi_path, mask_path, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_270')
                probs_map_sud, pro_map_sud, p_map_sud, new_map_sud = get_probs_map(model, dataloader)
                probs_map = probs_map + probs_map_sud
                pro_map = pro_map + pro_map_sud
                p_map = p_map + p_map_sud
                new_map = new_map + new_map_sud

                probs_map /= 8
                pro_map /= 8
                p_map /= 8
                new_map /= 8

            np.save(os.path.join(args.probs_map_path, filename + "-mean.npy"), probs_map)
            np.save(os.path.join(args.probs_map_path, filename + "-patch.npy"), pro_map)
            np.save(os.path.join(args.probs_map_path, filename + "-origin.npy"), p_map)
            np.save(os.path.join(args.probs_map_path, filename + "-grid.npy"), new_map)
            npy2jpg_single(probs_map, filename + "-mean")
            npy2jpg_single(pro_map, filename + "-patch")
            npy2jpg_single(p_map, filename + "-origin")
            npy2jpg_single(new_map, filename + "-grid")


def load_weight(args, model_crf, model_mil):
    weight = torch.load(args.ckpt_path)
    model_crf = model_crf.load_state_dict(weight['state_dict_crf'])
    model_mil = model_mil.load_state_dict(weight['state_dict_mil'])


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
