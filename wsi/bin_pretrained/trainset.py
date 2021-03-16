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

import torchvision

from pre_model import (resnet_crf, resnet_base, MODELS)



from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from wsi.model.layers import CRF

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from wsi.data.eye_image_producer import GridImageDataset  # noqa
from wsi.model import MODELS  # noqa

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--cfg_path', default="/home/omnisky/ajmq/patch-slide/configs/resnet34_base.json",
                    metavar='CFG_PATH',
                    type=str,
                    help='Path to the config file in json format')
parser.add_argument('--save_path', default="/home/omnisky/ajmq/patch-slide/output", metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument('--num_workers', default=2, type=int, help='number of'
                                                               ' workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='0,1', type=str, help='comma'
                                                                  ' separated indices of GPU to use, e.g. 0,1 for using GPU_0'
                                                                  ' and GPU_1, default 0.')


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

    patch_per_side = cfg['image_size'] // cfg['patch_size']
    grid_size = patch_per_side * patch_per_side
    if cfg['use_crf']:
        model = resnet_crf(cfg['model'], grid_size, pretrained=True)
    else:
        model = resnet_base(cfg['model'], pretrained=True)

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
    loss_fn2 = torch.nn.BCELoss().cuda()
    optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])

    dataset_train = GridImageDataset(cfg['data_path_train'],
                                     cfg['npy_path_train'],
                                     cfg['image_size'],
                                     cfg['patch_size'],
                                     crop_size=cfg['crop_size'])
    dataset_valid = GridImageDataset(cfg['data_path_valid'],
                                     cfg['npy_path_valid'],
                                     cfg['image_size'],
                                     cfg['patch_size'],
                                     crop_size=cfg['crop_size'])

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=batch_size_train,
                                  num_workers=num_workers,
                                  shuffle=True,
                                  drop_last=True)

    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size=batch_size_valid,
                                  num_workers=num_workers,
                                  drop_last=True)

    summary_train = {'epoch': 0, 'step': 0}
    summary_valid = {'epoch': 0, 'loss': float('inf'), 'acc': 0}
    summary_writer = SummaryWriter(args.save_path)
    loss_valid_best = float('inf')
    for epoch in range(cfg['epoch']):
        summary_train = train_epoch(summary_train, summary_writer, cfg, model,
                                    loss_fn, loss_fn2, optimizer,
                                    dataloader_train)
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': model.module.state_dict()},
                   os.path.join(args.save_path, 'train.ckpt'))

        time_now = time.time()
        summary_valid = valid_epoch(summary_valid, cfg, model, loss_fn, loss_fn2,
                                    dataloader_valid)
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


def acc_calculate(out_put, pre_dict):
    acc_data = (pre_dict == out_put).type(
        torch.cuda.FloatTensor).sum().item() * 1.0 / (
                   pre_dict.numel())
    return acc_data


def predict_reform(predict, mode=0, threshold=None):
    if mode == 0:
        if threshold:
            predict = (predict >= threshold).type(torch.cuda.FloatTensor)
        else:
            predict = (predict >= 0.5).type(torch.cuda.FloatTensor)
    elif mode == 1:
        [down, up, value] = threshold
        predict[(up > predict) & (predict >= down)] = value
        predict[predict >= up] = 1
        predict[predict < down] = 0
    return predict


def get_loss(predict, label, loss_fn1, predict2=None, label2=None, loss_fn2=None, ratio=None):
    loss1 = loss_fn1(predict, label)
    if loss_fn2 is not None:
        loss2 = loss_fn2(predict2, label2)
        if ratio is not None:
            loss_total = ratio[0] * loss1 + ratio[1] * loss2
            return loss1, loss2, loss_total
        else:
            return loss1, loss2
    else:
        return loss1


def print_section(string, data, mode="normal", print_function="logging"):
    if print_function == "logging":
        if mode == "normal":
            logging.info(string + ' {}'.format(data))
        elif mode == "double":
            string_total = string
            for i in range(len(data)):
                string_total += ' {}'.format(data[i])
            logging.info(string_total)
        elif mode == "loss_single":
            logging.info("loss name:" + string + ' {}'.format(data))
        elif mode == "all_loss":
            for i in range(len(string)):
                print_section(string[i], data[i], mode="loss_single")
        elif mode == "show_out_pre":  # show output and predict
            for i in range(len(string)):
                print_section(string[i], data[i])
        elif mode == "sample":  # show default result sum
            logging.info(
                '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
                'Training Acc : {:.3f}, Run Time : {:.2f}'.format(time.strftime("%Y-%m-%d %H:%M:%S"), data[0], data[1],
                                                                  data[2], data[3], data[4]))
    else:
        if mode == "normal":
            print(string + ' {}'.format(data))
        elif mode == "double":
            string_total = string
            for i in len(data):
                string_total += ' {}'.format(data[i])
            logging.info(string_total)
        elif mode == "loss_single":
            print("loss name:" + string + ' {}'.format(data))
        elif mode == "all_loss":
            for i in range(len(string)):
                print_section(string[i], data[i], mode="loss_single", print_function="print")
        elif mode == "show_out_pre":  # show output and predict
            for i in range(len(string)):
                print_section(string[i], data[i], print_function="print")
        elif mode == "sample":  # show default result sum
            print(
                '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
                'Training Acc : {:.3f}, Run Time : {:.2f}'.format(time.strftime("%Y-%m-%d %H:%M:%S"), data[0], data[1],
                                                                  data[2], data[3], data[4]))


def change_form(tensor, pre_value=0.5):
    base = 1 - pre_value
    tensor = tensor * base
    if len(tensor.shape) == 1:
        if tensor.mean() > 1:
            tensor[tensor < 1] = pre_value
        else:
            tensor[:] = 0
    else:
        tensor[tensor.mean(dim=1) > 0, :] = tensor[tensor.mean(dim=1) > 0, :] + pre_value
    return tensor


def output_form(output, pre_value=0.5, mode=0, half_size=0.1):
    if mode == 0:
        output = output.mean(dim=1)
        return output
    elif mode == 1:
        output = torch.nn.functional.sigmoid(output)
        output = output_form(output, mode=0)
        return output
    elif mode == 2:
        output2 = output.clone()
        output3 = output.clone()
        output3 = output_form(output3, mode=0)
        output = output_form(output, mode=1)
        predict2 = predict_reform(output)
        predict1 = predict_reform(output2)
        predict3 = predict_reform(output2, mode=1, threshold=[pre_value - half_size, pre_value + half_size, pre_value])
        return output2, output, output3, predict1, predict2, predict3


def train_epoch(summary, summary_writer, cfg, model, loss_fn, loss_fn2, optimizer,
                dataloader):
    model.train()

    time_now = time.time()
    for step, (data, target) in enumerate(dataloader):
        data = Variable(data.cuda(non_blocking=True))
        target = Variable(target.cuda(non_blocking=True))
        target = change_form(target, pre_value=0.6)
        tar = target.clone()
        target = target.mean(dim=1)
        output = model(data)
        output, output2, output3, predict1, predict2, predict3 = output_form(output, pre_value=0.6, mode=2)
        if step == 0:
            print_section("", output, print_function="print")
            print_section("", output2, print_function="print")
        loss1, loss2, loss = get_loss(output, tar, loss_fn, output2, target, loss_fn2, ratio=[0.2, 0.8])

        loss_name = ['loss1', 'loss2', 'loss_total']
        loss_value = [loss1, loss2, loss]
        print_section(loss_name, loss_value, mode="all_loss")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        target_mean_two = predict_reform(target)
        target_two = predict_reform(tar)

        print_section(['output_mean\n', 'target_mean\n', 'output_three\n', 'target_two\n'],
                      [output2, target, predict2, target_mean_two], mode="show_out_pre",
                      print_function="print")

        loss_data = loss.item()
        time_spent = time.time() - time_now
        time_now = time.time()
        print_section("step/step all", [step + 1, len(dataloader)], mode="double")

        acc_data = acc_calculate(predict1, target_two)
        acc_data2 = acc_calculate(predict2, target_mean_two)
        acc_data3 = acc_calculate(predict3, tar)
        print_section(['acc_two: ', 'acc_mean: ', 'acc_three: '], [acc_data, acc_data2, acc_data3], mode="show_out_pre")
        print_section("", [summary['epoch'] + 1, summary['step'] + 1, loss_data, acc_data, time_spent], mode="sample")

        summary['step'] += 1

        if summary['step'] % cfg['log_every'] == 0:
            summary_writer.add_scalar('train/loss', loss_data, summary['step'])
            summary_writer.add_scalar('train/acc', acc_data2, summary['step'])
            print_section("", output3, print_function="print")
            print_section("", tar, print_function="print")

    summary['epoch'] += 1

    return summary


def valid_epoch(summary, cfg, model, loss_fn, loss_fn2,
                dataloader):
    model.eval()

    loss_sum = 0
    acc_sum = 0
    time_now = time.time()
    for step, (data, target) in enumerate(dataloader):
        with torch.no_grad():
            data = Variable(data.cuda(non_blocking=True))
            target = Variable(target.cuda(non_blocking=True))
            target = change_form(target, pre_value=0.6)
            tar = target.clone()
            target = target.mean(dim=1)

            output = model(data)

            output, output2, output3, predict1, predict2, predict3 = output_form(output, pre_value=0.6, mode=2)
            if step == 0:
                print_section("", output, print_function="print")
                print_section("", output2, print_function="print")
            loss1, loss2, loss = get_loss(output, tar, loss_fn, output2, target, loss_fn2, ratio=[0.2, 0.8])

            loss_name = ['loss1', 'loss2', 'loss_total']
            loss_value = [loss1, loss2, loss]
            print_section(loss_name, loss_value, mode="all_loss")

            target_mean_two = predict_reform(target)
            target_two = predict_reform(tar)

            print_section(['output_mean\n', 'target_mean\n', 'output_three\n', 'output_two\n', 'target_two\n'],
                          [output2, target, predict2, predict3, target_mean_two], mode="show_out_pre",
                          print_function="print")

            loss_data = loss.item()
            time_spent = time.time() - time_now
            time_now = time.time()
            print_section(["step/step all"], [step + 1, len(dataloader)], mode="double")

            acc_data = acc_calculate(predict1, target_two)
            acc_data2 = acc_calculate(predict2, target_mean_two)
            acc_data3 = acc_calculate(predict3, tar)
            print_section(['acc_two: ', 'acc_mean: ', 'acc_three: '], [acc_data, acc_data2, acc_data3],
                          mode="show_out_pre")
            print_section("", [summary['epoch'] + 1, summary['step'] + 1, loss_data, acc_data, time_spent],
                          mode="sample")

            if (step + 1) % cfg['log_every'] == 0:
                print_section("", output3, print_function="print")
                print_section("", tar, print_function="print")

            loss_sum += loss_data
            acc_sum += acc_data
    steps = len(dataloader)

    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps
    summary['epoch'] += 1

    return summary


if __name__ == '__main__':
    main()
