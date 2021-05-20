import sys
import argparse
import logging
from run_process import *
from train_gan.train_gan import *
from train_gan.val_gan import *
from train_gan.val_train_gan import *
import torch

# !!!!!!!!!!!!
# batch_size禁止为1
# （对1没有进行特殊处理，dataloader输出少一维， 会产生各种问题）

from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
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

args_out = parser.parse_args()

import_path = args_out.dataset_produce_path
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
    cfg = import_config(args)

    batch_size_train, batch_size_valid, num_workers = get_run_parameter(args, cfg)

    model_crf, model_mil = produce_model(cfg)

    # model = load_dict(model, cfg)

    loss_fn_with_sigmoid, loss_fn_without_sigmoid = get_loss_func()

    optimizer_crf, optimizer_mil = get_optimzer(cfg, model_crf, model_mil)

    summary_train, summary_valid, summary_valid_train = get_summary_mil_based()

    summary_writer = SummaryWriter(args.save_path)

    loss_valid_best = float('inf')

    summary_writer = SummaryWriter(args.save_path)

    loss_valid_best = float('inf')

    process_func = FullProcess(cfg['point_path'], init_status=True)
    base_dataset = process_func.dataset
    dataloader_train_mil, dataloader_valid_mil = get_train_mil_dataloader(batch_size_train, batch_size_valid,
                                                                          num_workers, process_func)

    for epoch in range(cfg['epoch']):
        summary_train = train_gan()

        save_train_state_dict_mil_based(args, model_crf, model_mil, summary_train)

        summary_valid = val_gan()

        summary_valid_train = val_train_gan()

        loss_valid_best = save_best_in_valid_mil_based(args, loss_valid_best, summary_valid, summary_train, model_crf,
                                                       model_mil)
        # 记得选择使用哪个loss
        # 记得修改loss名
    summary_writer.close()


if __name__ == '__main__':
    main()
