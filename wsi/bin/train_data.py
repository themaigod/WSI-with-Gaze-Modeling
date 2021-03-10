import sys
import os
import argparse
import logging
import json
import time

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import SGD

from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from wsi.data.eye_image_producer import GridImageDataset  # noqa
from wsi.model import MODELS  # noqa

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--cfg_path', default="/home/omnisky/ajmq/NCRF/configs/resnet18_crf_data.json", metavar='CFG_PATH',
                    type=str,
                    help='Path to the config file in json format')
parser.add_argument('--save_path', default="/home/omnisky/ajmq/slideclassify/ncrf_save", metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument('--num_workers', default=2, type=int, help='number of'
                                                               ' workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='0,1', type=str, help='comma'
                                                                  ' separated indices of GPU to use, e.g. 0,1 for using GPU_0'
                                                                  ' and GPU_1, default 0.')

loss_fn2 = torch.nn.BCELoss().cuda()
def train_epoch(summary, summary_writer, cfg, model, loss_fn, optimizer,
                dataloader_tumor, dataloader_normal):
    model.train()

    steps = len(dataloader_tumor)
    batch_size = dataloader_tumor.batch_size
    grid_size = dataloader_tumor.dataset._grid_size
    dataiter_tumor = iter(dataloader_tumor)
    dataiter_normal = iter(dataloader_normal)

    time_now = time.time()
    for step in range(steps):
        data_tumor, target_tumor = next(dataiter_tumor)
        data_tumor = Variable(data_tumor.cuda(non_blocking=True))
        target_tumor = Variable(target_tumor.cuda(non_blocking=True))
        tumor_len = len(target_tumor)

        data_normal, target_normal = next(dataiter_normal)
        data_normal = Variable(data_normal.cuda(non_blocking=True))
        target_normal = Variable(target_normal.cuda(non_blocking=True))
        normal_len = len(target_normal)

        idx_rand = Variable(
            torch.randperm(tumor_len + normal_len).cuda(non_blocking=True))

        data = torch.cat([data_tumor, data_normal])[idx_rand]
        target = torch.cat([target_tumor, target_normal])[idx_rand]

        target = target * 0.4

        target[target.mean(dim=1) > 0, :] = target[target.mean(dim=1) > 0, :] + 0.6
        tar = target.clone()
        target = target.mean(dim=1)
        output = model(data)
        if step == 0:
            print(output)

        loss2 = loss_fn(output, tar)
        output = torch.nn.functional.sigmoid(output)
        if step == 0:
            print(output)
        out = output.clone()
        output = output.mean(dim=1)
        loss1 = loss_fn2(output, target)
        loss = 0.2 * loss1 + 0.8 * loss2
        print(loss1)
        print(loss2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = output.sigmoid()
        predicts = (output >= 0.5).type(torch.cuda.FloatTensor)
        print(output)
        print(target)
        print(predicts)

        targets = (target >= 0.5).type(torch.cuda.FloatTensor)
        print(targets)
        acc_data = (predicts == targets).type(
            torch.cuda.FloatTensor).sum().item() * 1.0 / (
                           batch_size * 2)
        loss_data = loss.item()

        time_spent = time.time() - time_now
        time_now = time.time()
        print(step)
        logging.info(
            '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
            'Training Acc : {:.3f}, Run Time : {:.2f}'
                .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1,
                                                    summary['step'] + 1, loss_data, acc_data, time_spent))

        summary['step'] += 1

        if summary['step'] % cfg['log_every'] == 0:
            summary_writer.add_scalar('train/loss', loss_data, summary['step'])
            summary_writer.add_scalar('train/acc', acc_data, summary['step'])
            print(out)
            print(tar)

    summary['epoch'] += 1

    return summary


def valid_epoch(summary, cfg, model, loss_fn,
                dataloader_tumor, dataloader_normal):
    model.eval()

    steps = len(dataloader_tumor)
    batch_size = dataloader_tumor.batch_size
    grid_size = dataloader_tumor.dataset._grid_size
    dataiter_tumor = iter(dataloader_tumor)
    dataiter_normal = iter(dataloader_normal)

    loss_sum = 0
    acc_sum = 0
    for step in range(steps):
        with torch.no_grad():
            data_tumor, target_tumor = next(dataiter_tumor)
            data_tumor = Variable(data_tumor.cuda(non_blocking=True))
            target_tumor = Variable(target_tumor.cuda(non_blocking=True))
            data_normal, target_normal = next(dataiter_normal)
            data_normal = Variable(data_normal.cuda(non_blocking=True))
            target_normal = Variable(target_normal.cuda(non_blocking=True))

            data = torch.cat([data_tumor, data_normal])
            target = torch.cat([target_tumor, target_normal])
            target = target.mean(dim=1) * 0.4
            target[target > 0] = target[target > 0] + 0.6
            output = model(data)
            output = torch.nn.functional.sigmoid(output)
            output = output.mean(dim=1)
            loss = loss_fn2(output, target)

            probs = output.sigmoid()
            predicts = (output >= 0.5).type(torch.cuda.FloatTensor)
            targets = (target >= 0.5).type(torch.cuda.FloatTensor)
            acc_data = (predicts == targets).type(
                torch.cuda.FloatTensor).sum().item() * 1.0 / (
                               batch_size * 2)
            loss_data = loss.item()
            print(output)
            print(target)
            print(acc_data)

            loss_sum += loss_data
            acc_sum += acc_data

    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps

    return summary


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

    patch_per_side = cfg['image_size'] // cfg['patch_size']
    grid_size = patch_per_side * patch_per_side
    model = MODELS[cfg['model']](num_nodes=grid_size, use_crf=cfg['use_crf'])
    model = DataParallel(model, device_ids=None)
    model = model.cuda()
    if cfg['other_dict'] == 1:
        stat_dict = torch.load(cfg['dict_path'])['state_dict']
        model_dict = model.state_dict()
        stat_dict = {k: v for k, v in stat_dict.items() if
                     k in model_dict.keys() and k != 'module.fc.weight' and k != 'module.fc.bias'}
        model_dict.update(stat_dict)
        model.load_state_dict(model_dict)
    loss_fn = BCEWithLogitsLoss().cuda()
    optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])

    dataset_tumor_train = GridImageDataset(cfg['data_path_train'],
                                           cfg['npy_path_train'],
                                           cfg['image_size'],
                                           cfg['patch_size'],
                                           crop_size=cfg['crop_size'])
    dataset_normal_train = GridImageDataset(cfg['data_path_train2'],
                                            cfg['npy_path_train2'],
                                            cfg['image_size'],
                                            cfg['patch_size'],
                                            crop_size=cfg['crop_size'])
    dataset_tumor_valid = GridImageDataset(cfg['data_path_valid'],
                                           cfg['npy_path_valid'],
                                           cfg['image_size'],
                                           cfg['patch_size'],
                                           crop_size=cfg['crop_size'])
    dataset_normal_valid = GridImageDataset(cfg['data_path_valid2'],
                                            cfg['npy_path_valid2'],
                                            cfg['image_size'],
                                            cfg['patch_size'],
                                            crop_size=cfg['crop_size'])

    dataloader_tumor_train = DataLoader(dataset_tumor_train,
                                        batch_size=batch_size_train,
                                        num_workers=num_workers)
    dataloader_normal_train = DataLoader(dataset_normal_train,
                                         batch_size=batch_size_train,
                                         num_workers=num_workers)
    dataloader_tumor_valid = DataLoader(dataset_tumor_valid,
                                        batch_size=batch_size_valid,
                                        num_workers=num_workers)
    dataloader_normal_valid = DataLoader(dataset_normal_valid,
                                         batch_size=batch_size_valid,
                                         num_workers=num_workers)

    summary_train = {'epoch': 0, 'step': 0}
    summary_valid = {'loss': float('inf'), 'acc': 0}
    summary_writer = SummaryWriter(args.save_path)
    loss_valid_best = float('inf')
    for epoch in range(cfg['epoch']):
        summary_train = train_epoch(summary_train, summary_writer, cfg, model,
                                    loss_fn, optimizer,
                                    dataloader_tumor_train,
                                    dataloader_normal_train)
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': model.module.state_dict()},
                   os.path.join(args.save_path, 'train.ckpt'))

        time_now = time.time()
        summary_valid = valid_epoch(summary_valid, cfg, model, loss_fn,
                                    dataloader_tumor_valid,
                                    dataloader_normal_valid)
        time_spent = time.time() - time_now

        logging.info(
            '{}, Epoch : {}, Step : {}, Validation Loss : {:.5f}, '
            'Validation Acc : {:.3f}, Run Time : {:.2f}'
                .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
                summary_train['step'], summary_valid['loss'],
                summary_valid['acc'], time_spent))

        summary_writer.add_scalar(
            'valid/loss', summary_valid['loss'], summary_train['step'])
        summary_writer.add_scalar(
            'valid/acc', summary_valid['acc'], summary_train['step'])

        if summary_valid['loss'] < loss_valid_best:
            loss_valid_best = summary_valid['loss']

            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'state_dict': model.module.state_dict()},
                       os.path.join(args.save_path, 'best.ckpt'))

    summary_writer.close()


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
