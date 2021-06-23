import sys
import argparse
import logging
from run_process import *
from train_epoch.train_crf import *
from train_epoch.train_mil import *
from train_epoch.valid_crf import *
from train_epoch.valid_mil import *
from train_epoch.valid_train_crf import *
import torch

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

    process_func = FullProcess(cfg['point_path'], init_status=True)
    base_dataset = process_func.dataset

    summary_train_crf, summary_train_mil, summary_valid_crf, summary_valid_mil, summary_valid_train_crf, summary_valid_train_mil = get_summary()

    summary_writer = SummaryWriter(args.save_path)

    loss_valid_best = float('inf')

    for epoch in range(cfg['epoch']):
        dataloader_train_crf, dataloader_valid_crf = get_train_crf_dataloader(batch_size_train, batch_size_valid,
                                                                              num_workers, process_func)

        dataloader_train_mil, dataloader_valid_mil = get_train_mil_dataloader(batch_size_train, batch_size_valid,
                                                                              num_workers, process_func)

        summary_train_crf = train_epoch_crf(summary_train_crf, summary_writer, cfg, model_crf,
                                            loss_fn_with_sigmoid, loss_fn_without_sigmoid, optimizer_crf,
                                            dataloader_train_crf, base_dataset)

        summary_train_mil = train_epoch_mil(summary_train_mil, summary_writer, cfg, model_mil,
                                            loss_fn_with_sigmoid, loss_fn_without_sigmoid, optimizer_mil,
                                            dataloader_train_mil, base_dataset)

        save_train_state_dict(args, model_crf, model_mil, summary_train_crf, summary_train_mil)

        summary_valid_crf = vaild_crf(base_dataset, cfg, dataloader_valid_crf, loss_fn_with_sigmoid,
                                      loss_fn_without_sigmoid, model_crf, summary_train_crf, summary_valid_crf)

        summary_valid_mil = vaild_mil(base_dataset, cfg, dataloader_valid_mil, loss_fn_with_sigmoid,
                                      loss_fn_without_sigmoid, model_mil, summary_train_mil, summary_valid_mil)

        write_in_summary(summary_train_crf, summary_train_mil, summary_valid_crf, summary_valid_mil, summary_writer)

        valid_train_crf(base_dataset, cfg, dataloader_train_crf, loss_fn_with_sigmoid, loss_fn_without_sigmoid,
                        model_crf, summary_train_crf, summary_valid_train_crf)

        valid_train_mil(base_dataset, cfg, dataloader_train_mil, loss_fn_with_sigmoid, loss_fn_without_sigmoid,
                        model_mil, summary_train_mil, summary_valid_train_mil)

        save_best_in_valid(args, loss_valid_best, summary_valid_mil, summary_train_crf, summary_train_mil, model_crf,
                           model_mil)
    summary_writer.close()


if __name__ == '__main__':
    main()
