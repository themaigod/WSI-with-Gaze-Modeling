import sys
import argparse
import logging
import json
import time

from run_process import *
# from train_gan.train_gan import *
# from train_gan.val_gan import *
# from train_gan.val_train_gan import *
from train_gan.tool import *
import torch

# !!!!!!!!!!!!
# batch_size禁止为1
# （对1没有进行特殊处理，dataloader输出少一维， 会产生各种问题）

from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--cfg_path', default="/home/omnisky/ajmq/patch_slide_relate/save5.28/cfg.json",
                    metavar='CFG_PATH',
                    type=str,
                    help='Path to the config file in json format')
parser.add_argument('--save_path', default="/home/omnisky/ajmq/patch_slide_relate/save5.28/save_test",
                    metavar='SAVE_PATH', type=str,
                    help='Path to the saved test result')
parser.add_argument('--num_workers', default=5, type=int, help='number of'
                                                               ' workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='0,1', type=str, help='comma'
                                                                  ' separated indices of GPU to use, e.g. 0,1 for using GPU_0'
                                                                  ' and GPU_1, default 0.')
parser.add_argument('--weight_path', default='/home/omnisky/ajmq/patch_slide_relate/save5.28/best.ckpt', type=str,
                    help='weight path')
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

    _, batch_size_test, num_workers = get_run_parameter(args, cfg)

    model_crf, model_mil = produce_model_no_cuda(cfg)

    load_weight(args, model_crf, model_mil)

    model_crf, model_mil = cuda_model(model_crf, model_mil)

    _, loss_fn = get_loss_func()

    summary = {'epoch': 0, 'loss': float(0), 'loss_mil': float(0), 'acc': 0, 'acc_crf': 0, 'fpr': 0,
               "fnr": 0}

    summary_writer = SummaryWriter(args.save_path)

    process_func = FullProcess(cfg['point_path'], init_status=False)
    base_dataset = process_func.dataset
    dataset_test_mil = process_func.inner_get_output_mil_dataset(2)
    dataloader_test_mil = DataLoader(dataset_test_mil,
                                     batch_size=batch_size_test,
                                     num_workers=num_workers,
                                     drop_last=True)

    model_crf.eval()
    model_mil.eval()

    time_now = time.time()
    len_data_loader = len(dataloader_test_mil)
    pre_value = 0.8

    time_start = time.time()

    for step, (data_crf, target_crf, data_mil, target_mil, patch, position) in enumerate(dataloader_test_mil):
        with torch.no_grad():
            data_crf, data_mil, target_crf, target_crf_clone, target_mil = transfer2cuda(data_crf, data_mil, target_crf,
                                                                                         target_mil, pre_value)
            output_crf_ori = model_crf(data_crf)
            output_crf_ori, output_crf, predict_crf, predict_crf_mean = output_form(output_crf_ori, pre_value=pre_value,
                                                                                    mode=3)
            output_crf_round = torch_round_with_backward(output_crf)
            data_mil = model_get_data_mil(data_mil, output_crf_round)
            predict_mil = model_mil(data_mil)
            predict_mil = torch.sigmoid(predict_mil)
            record_result(base_dataset, patch, position, predict_mil)
        print("step:" + str(step + 1) + "/" + "total step:" + str(len_data_loader) + "  time spent:" + str(
            time.time() - time_now))
        time_spent = time.time() - time_start
        time_whole = time_spent / (step + 1) * len_data_loader
        time_need = time_whole - time_spent
        print("test inference used :" + str(time_spent // (60 * 60)) + "hour," + str(
            (time_spent // 60) % 60) + "min," + str(time_spent % 60) + "s")
        print("test inference need :" + str(time_need // (60 * 60)) + "hour," + str(
            (time_need // 60) % 60) + "min," + str(time_need % 60) + "s")
        time_now = time.time()
    torch.cuda.empty_cache()
    base_dataset.slide_max(2)
    test_dataset = base_dataset.produce_dataset_test_mil(2, cfg['top_k'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test // 4, num_workers=num_workers, shuffle=True,
                             drop_last=True)
    time_now = time.time()
    record_list = []

    time_start = time.time()
    len_test_loader = len(test_loader)
    for step, (data_crf, target_crf, data_mil, target_mil, patch, position) in enumerate(test_loader):
        record_list.append([])
        with torch.no_grad():
            data_crf, data_mil, target_crf, target_crf_clone, target_mil = transfer2cuda(data_crf, data_mil, target_crf,
                                                                                         target_mil, pre_value)
            output_crf_ori = model_crf(data_crf)
            output_crf_ori, output_crf, predict_crf, predict_crf_mean = output_form(output_crf_ori, pre_value=pre_value,
                                                                                    mode=3)
            output_crf_round = torch_round_with_backward(output_crf)
            data_mil = model_get_data_mil(data_mil, output_crf_round)
            predict_mil = model_mil(data_mil)
            predict_mil = torch.sigmoid(predict_mil)

            loss_crf_mean = loss_fn(output_crf, target_crf)
            loss_crf_ori = loss_fn(output_crf_ori, target_crf_clone)
            loss_crf = 0.2 * loss_crf_ori + 0.8 * loss_crf_mean
            loss_crf_mil = loss_fn(predict_mil, target_mil)
            loss_crf_final = loss_crf * 0.4 + loss_crf_mil * 0.6

            time_now = show_crf(loss_crf, loss_crf_final, loss_crf_mean, loss_crf_mil, loss_crf_ori, output_crf,
                                output_crf_ori, pre_value, predict_crf, predict_crf_mean, predict_mil, step, summary,
                                target_crf, target_mil, target_crf_clone, time_now, test_loader, summary_writer, cfg,
                                record_list)

            print("step:" + str(step + 1) + "/" + "total step:" + str(len_test_loader) + "  time spent:" + str(
                time.time() - time_now))
            time_spent = time.time() - time_start
            time_whole = time_spent / (step + 1) * len_test_loader
            time_need = time_whole - time_spent
            print("test used :" + str(time_spent // (60 * 60)) + "hour," + str(
                (time_spent // 60) % 60) + "min," + str(time_spent % 60) + "s")
            print("test need :" + str(time_need // (60 * 60)) + "hour," + str(
                (time_need // 60) % 60) + "min," + str(time_need % 60) + "s")
            time_now = time.time()

        summary_writer.add_scalar('test/loss', summary['loss'] / len(test_loader), summary['epoch'])
        summary_writer.add_scalar('test/acc', summary['acc'] / len(test_loader), summary['epoch'])
        summary_writer.add_scalar('test/tpr', 1 - (summary['fnr'] / len(test_loader)), summary['epoch'])
        print("Test")
        print("loss: " + str(summary['loss'] / len(test_loader)))
        print("loss_mil: " + str(summary['loss_mil'] / len(test_loader)))
        print("acc: " + str(summary['acc'] / len(test_loader)))
        print("acc_crf: " + str(summary['acc_crf'] / len(test_loader)))
        print("tpr: " + str(1 - (summary['fnr'] / len(test_loader))))
        print("fpr: " + str(summary['fpr'] / len(test_loader)))
        torch.cuda.empty_cache()
        torch.save({"summary": summary, "len": len(test_loader), "loss_final": summary['loss'] / len(test_loader),
                    "loss_mil": summary['loss_mil'] / len(test_loader), "acc": summary['acc'] / len(test_loader),
                    "acc_crf": summary['acc_crf'] / len(test_loader), "tpr": 1 - (summary['fnr'] / len(test_loader)),
                    "fpr": summary['fpr'] / len(test_loader)}, os.path.join(args.save_path, 'test_result.ckpt'))
        path = os.path.join(args.save_path, 'all_result{}.json'.format(epoch))
        with open(path, 'w') as f:
            json.dump(record_list_total, f)


def load_weight(args, model_crf, model_mil):
    weight = torch.load(args.weight_path)
    model_crf = model_crf.load_state_dict(weight['state_dict_crf'])
    model_mil = model_mil.load_state_dict(weight['state_dict_mil'])


def show_crf(loss_crf, loss_crf_final, loss_crf_mean, loss_crf_mil, loss_crf_ori, output_crf, output_crf_ori, pre_value,
             predict_crf, predict_crf_mean, predict_mil, step, summary, target_crf, target_mil, target_crf_clone,
             time_now, train_loader, summary_writer, cfg, record_list):
    if step % 3 == 0:
        print("output crf mean:")
        print(output_crf)
        print("predict crf all:")
        print(predict_crf)
    print("predict crf mean:")
    print(predict_crf_mean)
    target_mean_two = predict_reform(target_crf)
    target_two = predict_reform(target_crf_clone)
    print("target crf mean:")
    print(target_mean_two)
    print_section("num 1 in predict", (predict_crf == 1).sum().item())
    print_section("num 0 in predict", (predict_crf == 0).sum().item())
    print_section("num 1 in target", (target_two == 1).sum().item())
    print_section("num 0 in target", (target_two == 0).sum().item())
    print_section("step/step all", [step + 1, len(train_loader)], mode="double")
    acc_data = acc_calculate(predict_crf, target_two)
    acc_data2 = acc_calculate(predict_crf_mean, target_mean_two)
    predict3 = predict_reform(output_crf_ori, mode=1, threshold=[pre_value - 0.1, pre_value + 0.1, pre_value])
    acc_data3 = acc_calculate(predict3, target_crf_clone)
    predict_mil_to_2 = predict_reform(predict_mil)
    acc_data_mil = acc_calculate(predict_mil_to_2, target_mil)

    print("output mil:")
    print(list(np.array(predict_mil_to_2.cpu())))
    print("target_mil")
    print(list(np.array(target_mil.cpu())))

    err_ori, fpr_ori, fnr_ori = calc_err(predict_crf.cpu(), target_two.cpu())
    err_mean, fpr_mean, fnr_mean = calc_err(predict_crf_mean.cpu(), target_mean_two.cpu())
    err_mil, fpr_mil, fnr_mil = calc_err(predict_mil_to_2.cpu(), target_mil.cpu())

    fp_fn_name = ['False Positive Rate in ori: ', 'False Negative Rate in ori: ', 'False Positive Rate in mean: ',
                  'False Negative Rate in mean: ', 'False Positive Rate in mil: ',
                  'False Negative Rate in mil: ']

    print_section(fp_fn_name, [fpr_ori, fnr_ori, fpr_mean, fnr_mean, fpr_mil, fnr_mil],
                  mode="show_out_pre")

    print_section(['acc_two: ', 'acc_mean: ', 'acc_three: ', 'acc_mil'], [acc_data, acc_data2, acc_data3, acc_data_mil],
                  mode="show_out_pre")
    time_spent = time.time() - time_now
    time_now = time.time()
    loss_name = ['loss_crf_ori: ', 'loss_crf_mean: ', 'loss_crf: ', 'loss_crf_mil: ', 'loss_crf_final: ']
    print_section(loss_name, [loss_crf_ori, loss_crf_mean, loss_crf, loss_crf_mil, loss_crf_final],
                  mode="show_out_pre")
    print("it's test")
    print_section("", [summary['epoch'] + 1, step + 1, loss_crf_final, acc_data, time_spent],
                  mode="sample")

    record_list[-1].append(float(loss_crf_ori.cpu()))
    record_list[-1].append(float(loss_crf_mean.cpu()))
    record_list[-1].append(float(loss_crf.cpu()))
    record_list[-1].append(float(loss_crf_mil.cpu()))
    record_list[-1].append(float(loss_crf_final.cpu()))
    record_list[-1].append(acc_data)
    record_list[-1].append(acc_data2)
    record_list[-1].append(acc_data3)
    record_list[-1].append(acc_data_mil)
    record_list[-1].append(fpr_ori)
    record_list[-1].append(fnr_ori)
    record_list[-1].append(fpr_mean)
    record_list[-1].append(fnr_mean)
    record_list[-1].append(fpr_mil)
    record_list[-1].append(fnr_mil)

    summary['loss'] += float(loss_crf_final.cpu())
    summary['loss_mil'] += float(loss_crf_mil.cpu())
    summary['acc'] += acc_data_mil
    summary['acc_crf'] += acc_data2
    summary['fpr'] = fpr_mil
    summary['fnr'] = fnr_mil

    # if summary['step'] % cfg['log_every'] == 0:
    #     summary_writer.add_scalar('train_epoch/loss in selector', loss_crf_final, summary['step'])
    #     summary_writer.add_scalar('train_epoch/acc in selector', acc_data2, summary['step'])
    #     summary_writer.add_scalar('train fpr/tpr in selector', 1 - fpr_ori, fpr_ori)
    #     print_section("", output_crf_ori, print_function="print")
    #     print_section("", target_crf_clone, print_function="print")

    return time_now


def model_get_data_mil(data_mil, output_crf):
    output_crf = output_crf.clone()
    for i in range(len(data_mil.shape) - 1):
        output_crf = output_crf.unsqueeze(1)
    # output_crf = output_crf.expand((data_mil.shape[0], data_mil.shape[1], data_mil.shape[2], data_mil.shape[3]))
    # 似乎不需要了，pytorch自带广播机制（broadcast）
    data_mil = output_crf * data_mil
    return data_mil


def access_index(output_crf):
    index = []
    for i in range(len(output_crf)):
        if output_crf[i] >= 0.5:
            index.append(i)
    return index


def record_result(dataset, patch, position, predict_mil):
    predict_final = np.array(predict_mil.detach().cpu())
    # dataset.get_index(predict_final, patch, position, 0)
    patch = np.array(patch).tolist()
    position = np.array(position).tolist()
    dataset.record_result_mil(predict_final, patch, position, 2)


def transfer2cuda(data_crf, data_mil, target_crf, target_mil, pre_value):
    data_crf = Variable(data_crf.cuda(non_blocking=True))
    data_mil = Variable(data_mil.cuda(non_blocking=True))
    target_crf = Variable(target_crf.cuda(non_blocking=True))
    target_mil = Variable(target_mil.cuda(non_blocking=True))
    target_crf = change_form(target_crf, pre_value=pre_value)
    target_crf_mean = target_crf.clone().detach().mean(dim=1)
    return data_crf, data_mil, target_crf_mean, target_crf, target_mil


if __name__ == '__main__':
    main()
