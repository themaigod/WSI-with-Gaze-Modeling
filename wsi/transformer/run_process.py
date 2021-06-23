import os
import json
import torch
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import SGD
from pre_model import (ResNetBase, MIL, ResNetTransformer)
from torch.utils.data import DataLoader

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def save_record_key(args, epoch, record_list_key_train, record_list_key_valid):
    path = os.path.join(args.save_path, 'key_result_train{}.json'.format(epoch))
    with open(path, 'w') as f:
        json.dump(record_list_key_train, f)
    path = os.path.join(args.save_path, 'key_result_valid{}.json'.format(epoch))
    with open(path, 'w') as f:
        json.dump(record_list_key_valid, f)


def save_record_list(args, epoch, record_list_total_train, record_list_total_valid):
    path = os.path.join(args.save_path, 'all_result_train{}.json'.format(epoch))
    with open(path, 'w') as f:
        json.dump(record_list_total_train, f)
    path = os.path.join(args.save_path, 'all_result_valid{}.json'.format(epoch))
    with open(path, 'w') as f:
        json.dump(record_list_total_valid, f)


def save_best_in_valid_epoch_based(args, loss_valid_best, summary_valid, summary_train_crf, summary_train_mil,
                                   model_crf,
                                   model_mil):
    if summary_valid['loss'] < loss_valid_best:
        loss_valid_best = summary_valid['loss']

        torch.save({'epoch_crf': summary_train_crf['epoch'],
                    'step_crf': summary_train_crf['step'],
                    'epoch_mil': summary_train_mil['epoch'],
                    'step_mil': summary_train_mil['step'],
                    'state_dict_crf': model_crf.module.state_dict(),
                    'state_dict_mil': model_mil.module.state_dict()},
                   os.path.join(args.save_path, 'best.ckpt'))

    return loss_valid_best


def save_best_in_valid_mil_based(args, loss_valid_best, summary_valid, summary_train,
                                 model_crf, model_mil):
    torch.save({'epoch': summary_train['epoch'],
                'step': summary_train['step'],
                'state_dict_crf': model_crf.module.state_dict(),
                'state_dict_mil': model_mil.module.state_dict()},
               os.path.join(args.save_path, 'train_epoch{}.ckpt'.format(summary_train['epoch'])))
    print("Loss:" + str(summary_valid['loss_mil']))
    print("Loss before:" + str(loss_valid_best))
    if summary_valid['loss_mil'] < loss_valid_best:
        loss_valid_best = summary_valid['loss']

        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict_crf': model_crf.module.state_dict(),
                    'state_dict_mil': model_mil.module.state_dict()},
                   os.path.join(args.save_path, 'best.ckpt'))

    return loss_valid_best


def get_train_crf_dataloader(batch_size_train, batch_size_valid, num_workers, process_func):
    dataset_train_crf, dataset_valid_crf = process_func.inner_get_output_dataset()
    dataloader_train_crf = DataLoader(dataset_train_crf,
                                      batch_size=batch_size_train,
                                      num_workers=num_workers,
                                      shuffle=True,
                                      drop_last=True)
    dataloader_valid_crf = DataLoader(dataset_valid_crf,
                                      batch_size=batch_size_valid,
                                      num_workers=num_workers,
                                      drop_last=True)
    return dataloader_train_crf, dataloader_valid_crf


def get_train_mil_dataloader(batch_size_train, batch_size_valid, num_workers, process_func):
    dataset_train_mil, dataset_valid_mil = process_func.inner_get_output_mil_dataset()
    dataloader_train_mil = DataLoader(dataset_train_mil,
                                      batch_size=batch_size_train,
                                      num_workers=num_workers,
                                      shuffle=True,
                                      drop_last=True)
    dataloader_valid_mil = DataLoader(dataset_valid_mil,
                                      batch_size=batch_size_valid,
                                      num_workers=num_workers,
                                      drop_last=True)
    return dataloader_train_mil, dataloader_valid_mil


def write_in_summary(summary_train_crf, summary_train_mil, summary_valid_crf, summary_valid_mil, summary_writer):
    summary_writer.add_scalar(
        'valid/loss in selector', summary_valid_crf['loss'], summary_train_crf['step'])
    summary_writer.add_scalar(
        'valid/acc in selector', summary_valid_crf['acc'], summary_train_crf['step'])
    summary_writer.add_scalar(
        'valid/loss in mil', summary_valid_mil['loss'], summary_train_mil['step'])
    summary_writer.add_scalar(
        'valid/acc in mil', summary_valid_mil['acc'], summary_train_mil['step'])


def load_dict(model, cfg):
    if cfg['other_dict'] == 1:
        stat_dict = torch.load(cfg['dict_path'])['state_dict']
        model_dict = model.state_dict()
        stat_dict = {k: v for k, v in stat_dict.items() if
                     k in model_dict.keys() and k != 'module.fc.weight' and k != 'module.fc.bias'}
        model_dict.update(stat_dict)
        model.load_state_dict(model_dict)
    # 保留但暂时不考虑


def get_optimzer(cfg, model_crf, model_mil):
    optimizer_crf = SGD(model_crf.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
    optimizer_mil = SGD(model_mil.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
    return optimizer_crf, optimizer_mil


def get_loss_func():
    loss_fn_with_sigmoid = BCEWithLogitsLoss().cuda()
    loss_fn_without_sigmoid = torch.nn.BCELoss().cuda()
    return loss_fn_with_sigmoid, loss_fn_without_sigmoid


def produce_model(cfg):
    # patch_per_side = cfg['image_size'] // cfg['patch_size']
    # grid_size = patch_per_side * patch_per_side
    model_crf = ResNetBase(key=cfg['model_crf'], pretrained=cfg['pretrained_crf'])
    model_mil = MIL(key=cfg['model_mil'], pretrained=True)
    model_crf = DataParallel(model_crf, device_ids=None)
    model_crf = model_crf.cuda()
    model_mil = DataParallel(model_mil, device_ids=None)
    model_mil = model_mil.cuda()
    return model_crf, model_mil


def produce_model_transformer(cfg):
    # patch_per_side = cfg['image_size'] // cfg['patch_size']
    # grid_size = patch_per_side * patch_per_side
    model_crf = ResNetTransformer(key=cfg['model_crf'], pretrained=cfg['pretrained_crf'])
    model_mil = MIL(key=cfg['model_mil'], pretrained=True)
    model_crf = DataParallel(model_crf, device_ids=None)
    model_crf = model_crf.cuda()
    model_mil = DataParallel(model_mil, device_ids=None)
    model_mil = model_mil.cuda()
    return model_crf, model_mil


def produce_model_no_cuda(cfg):
    # patch_per_side = cfg['image_size'] // cfg['patch_size']
    # grid_size = patch_per_side * patch_per_side
    model_crf = ResNetBase(key=cfg['model_crf'], pretrained=cfg['pretrained_crf'])
    model_mil = MIL(key=cfg['model_mil'], pretrained=True)
    return model_crf, model_mil


def cuda_model(model_crf, model_mil):
    model_crf = DataParallel(model_crf, device_ids=None)
    model_crf = model_crf.cuda()
    model_mil = DataParallel(model_mil, device_ids=None)
    model_mil = model_mil.cuda()
    return model_crf, model_mil


def get_run_parameter(args, cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    num_GPU = len(args.device_ids.split(','))
    batch_size_train = cfg['batch_size'] * num_GPU
    batch_size_valid = int(cfg['batch_size'] * num_GPU * 2)
    num_workers = args.num_workers * num_GPU
    if cfg['image_size'] % cfg['patch_size'] != 0:
        raise Exception('Image size / patch size != 0 : {} / {}'.
                        format(cfg['image_size'], cfg['patch_size']))
    return batch_size_train, batch_size_valid, num_workers


def import_config(args):
    with open(args.cfg_path) as f:
        cfg = json.load(f)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:
        json.dump(cfg, f, indent=1)
    return cfg


def save_train_state_dict_epoch_based(args, model_crf, model_mil, summary_train_crf, summary_train_mil):
    torch.save({'epoch_crf': summary_train_crf['epoch'],
                'step_crf': summary_train_crf['step'],
                'epoch_mil': summary_train_mil['epoch'],
                'step_mil': summary_train_mil['step'],
                'state_dict_crf': model_crf.module.state_dict(),
                'state_dict_mil': model_mil.module.state_dict()},
               os.path.join(args.save_path, 'train_epoch.ckpt'))


def save_train_state_dict_mil_based(args, model_crf, model_mil, summary_train):
    torch.save({'epoch': summary_train['epoch'],
                'step': summary_train['step'],
                'state_dict_crf': model_crf.module.state_dict(),
                'state_dict_mil': model_mil.module.state_dict()},
               os.path.join(args.save_path, 'train_epoch.ckpt'))


def get_summary_epoch_based():
    summary_train_crf = {'epoch': 0, 'step': 0}
    summary_valid_crf = {'epoch': 0, 'loss': float('inf'), 'acc': 0}
    summary_valid_train_crf = {'epoch': 0, 'loss': float('inf'), 'acc': 0}
    summary_train_mil = {'epoch': 0, 'step': 0}
    summary_valid_mil = {'epoch': 0, 'loss': float('inf'), 'acc': 0}
    summary_valid_train_mil = {'epoch': 0, 'loss': float('inf'), 'acc': 0}
    return summary_train_crf, summary_train_mil, summary_valid_crf, summary_valid_mil, summary_valid_train_crf, summary_valid_train_mil


def get_summary_mil_based():
    summary_train = {'epoch': 0, 'step': 0}
    summary_valid = {'epoch': 0, 'loss': float(0), 'loss_mil': float(0), 'acc': 0, 'acc_crf': 0, 'fp': 0,
                     "fn": 0, "tp": 0, "tn": 0}
    # summary_valid_train = {'epoch': 0, 'loss_classify': float('inf'), 'loss_selector': float('inf'),
    #                        'loss_mixed': float('inf'), 'acc_classify': 0, 'acc_glaze': 0}
    return summary_train, summary_valid  # summary_valid_train
