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
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import SGD

from tensorboardX import SummaryWriter

from pre_model import (ResnetAll, MIL)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from wsi.data.eye_image_producer import GridImageDataset
from wsi.model import MODELS

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--cfg_path', default="../../configs/resnet34_new_dataset.json",
                    metavar='CFG_PATH',
                    type=str,
                    help='Path to the config file in json format')
parser.add_argument('--save_path', default="/home/omnisky/ajmq/patch_slide_relate/save", metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument('--num_workers', default=6, type=int, help='number of'
                                                               ' workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='0,1', type=str, help='comma'
                                                                  ' separated indices of GPU to use, e.g. 0,1 for using GPU_0'
                                                                  ' and GPU_1, default 0.')
parser.add_argument('--dataset_produce_path', default='/home/omnisky/ajmq/process_operate_local', type=str,
                    help='where to import dataset')

args = parser.parse_args()

import_path = args.dataset_produce_path
if import_path[-1] == '/':
    import_path = import_path[:-1]
# import_path = "/home/omnisky/ajmq/process_operate_local"
sys.path.append(import_path)
from StartProcess import FullProcess


def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    run(args)


def run(args):
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:
        json.dump(cfg, f, indent=1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    num_GPU = len(args.device_ids.split(','))
    batch_size_train = cfg['batch_size'] * num_GPU
    batch_size_valid = cfg['batch_size'] * num_GPU * 2
    num_workers = args.num_workers * num_GPU

    if cfg['image_size'] % cfg['patch_size'] != 0:
        raise Exception('Image size / patch size != 0 : {} / {}'.
                        format(cfg['image_size'], cfg['patch_size']))

    model_crf = ResnetAll(key=cfg['model_crf'], grid_size=grid_size, pretrained=False, use_crf=cfg['use_crf'])

    model_mil = MIL(key=cfg['model_mil'], pretrained=True)

    model_crf = DataParallel(model_crf, device_ids=None)
    model_crf = model_crf.cuda()

    model_mil = DataParallel(model_mil, device_ids=None)
    model_mil = model_mil.cuda()

    # if cfg['other_dict'] == 1:
    #     stat_dict = torch.load(cfg['dict_path'])['state_dict']
    #     model_dict = model.state_dict()
    #     stat_dict = {k: v for k, v in stat_dict.items() if
    #                  k in model_dict.keys() and k != 'module.fc.weight' and k != 'module.fc.bias'}
    #     model_dict.update(stat_dict)
    #     model.load_state_dict(model_dict)
    # 保留但暂时不考虑

    loss_fn = BCEWithLogitsLoss().cuda()
    loss_fn2 = torch.nn.BCELoss().cuda()
