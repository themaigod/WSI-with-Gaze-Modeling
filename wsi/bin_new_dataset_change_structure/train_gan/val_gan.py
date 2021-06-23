import torch.nn

from .tool import *


def val_gan(dataloader, model_crf, model_mil, dataset, summary, num_workers, batch_size, loss_fn, top_k, pre_value,
            summary_writer, cfg, record_list_total, record_list_key):
    model_crf.eval()
    model_mil.eval()
    time_now = time.time()
    len_data_loader = len(dataloader)

    time_start = time.time()

    for step, (data_crf, target_crf, data_mil, target_mil, patch, position) in enumerate(dataloader):
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
            record_result(dataset, patch, position, predict_mil)
        print("step:" + str(step + 1) + "/" + "total step:" + str(len_data_loader) + "  time spent:" + str(
            time.time() - time_now))
        time_spent = time.time() - time_start
        time_whole = time_spent / (step + 1) * len_data_loader
        time_need = time_whole - time_spent
        print("val inference used :" + str(time_spent // (60 * 60)) + "hour," + str(
            (time_spent // 60) % 60) + "min," + str(time_spent % 60) + "s")
        print("val inference need :" + str(time_need // (60 * 60)) + "hour," + str(
            (time_need // 60) % 60) + "min," + str(time_need % 60) + "s")
        time_now = time.time()
    torch.cuda.empty_cache()
    dataset.slide_max(1)
    val_dataset = dataset.produce_dataset_test_mil(1, top_k)
    val_loader = DataLoader(val_dataset, batch_size=batch_size // 4, num_workers=num_workers, shuffle=True,
                            drop_last=True)
    time_now = time.time()
    record_list = []
    record_key_dict = []

    time_start = time.time()
    len_val_loader = len(val_loader)

    for step, (data_crf, target_crf, data_mil, target_mil, patch, position) in enumerate(val_loader):
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
            record_key_dict_single = {}

            time_now = show_crf(loss_crf, loss_crf_final, loss_crf_mean, loss_crf_mil, loss_crf_ori, output_crf,
                                output_crf_ori, pre_value, predict_crf, predict_crf_mean, predict_mil, step, summary,
                                target_crf, target_mil, target_crf_clone, time_now, val_loader, summary_writer, cfg,
                                record_list, record_key_dict_single)

            print("step:" + str(step + 1) + "/" + "total step:" + str(len_val_loader) + "  time spent:" + str(
                time.time() - time_now))
            time_spent = time.time() - time_start
            time_whole = time_spent / (step + 1) * len_val_loader
            time_need = time_whole - time_spent
            print("val used :" + str(time_spent // (60 * 60)) + "hour," + str(
                (time_spent // 60) % 60) + "min," + str(time_spent % 60) + "s")
            print("val need :" + str(time_need // (60 * 60)) + "hour," + str(
                (time_need // 60) % 60) + "min," + str(time_need % 60) + "s")
            time_now = time.time()

    record_list_total.append(record_list)
    record_list_key.append(record_key_dict)
    tpr = summary['tp'] / (summary['tp'] + summary['fn'])
    fpr = summary['fp'] / (summary['fp'] + summary['tn'])
    summary_writer.add_scalar('val_epoch/loss', summary['loss'] / len(val_loader), summary['epoch'])
    summary_writer.add_scalar('val_epoch/acc', summary['acc'] / len(val_loader), summary['epoch'])
    summary_writer.add_scalar('val_epoch/tpr', tpr, summary['epoch'])
    summary_writer.add_scalar('val_epoch/fpr', fpr, summary['epoch'])
    print("Val patch")
    print("loss: " + str(summary['loss'] / len(val_loader)))
    print("loss_mil: " + str(summary['loss_mil'] / len(val_loader)))
    print("acc: " + str(summary['acc'] / len(val_loader)))
    print("acc_crf: " + str(summary['acc_crf'] / len(val_loader)))
    print("tpr: " + str(tpr))
    print("fpr: " + str(fpr))

    result_mil, result_label = dataset.calc_result(1)

    err_slide, fpr_slide, fnr_slide = calc_err(result_mil, result_label)
    acc_slide = 1 - err_slide
    summary_writer.add_scalar('val_epoch/acc_slide', acc_slide, summary['epoch'])
    summary_writer.add_scalar('val_epoch/tpr_slide', 1 - fnr_slide, summary['epoch'])
    summary_writer.add_scalar('val_epoch/fpr_slide', fpr_slide, summary['epoch'])
    print("Validation")
    print("acc: " + str(acc_slide))
    print("tpr: " + str(1 - fnr_slide))
    print("fpr: " + str(fpr_slide))
    if acc_slide < 0:
        print(6666)

    torch.cuda.empty_cache()
    summary['epoch'] += 1
    return summary


def show_crf(loss_crf, loss_crf_final, loss_crf_mean, loss_crf_mil, loss_crf_ori, output_crf, output_crf_ori, pre_value,
             predict_crf, predict_crf_mean, predict_mil, step, summary, target_crf, target_mil, target_crf_clone,
             time_now, train_loader, summary_writer, cfg, record_list, record_key_dict_single):
    print_every = 3
    print_complex(print_every, output_crf, predict_crf, step)

    predict3, predict_mil_to_2, target_mean_two, target_two = reform_relate(output_crf_ori, pre_value, predict_mil,
                                                                            target_crf, target_crf_clone)
    print_relate(predict_crf, predict_crf_mean, step, target_mean_two, target_two, train_loader)
    acc_data, acc_data2, acc_data3, acc_data_mil = all_acc_calc(predict3, predict_crf, predict_crf_mean,
                                                                predict_mil_to_2, target_crf_clone, target_mean_two,
                                                                target_mil, target_two)

    print("output mil:")
    print(list(np.array(predict_mil_to_2.cpu())))
    print("target_mil")
    print(list(np.array(target_mil.cpu())))

    fn_mean, fn_mil, fn_ori, fnr_mean, fnr_mil, fnr_ori, fp_mean, fp_mil, fp_ori, fpr_mean, fpr_mil, fpr_ori, tn_mean, tn_mil, tn_ori, tp_mean, tp_mil, tp_ori = calc_relate(
        predict_crf, predict_crf_mean, predict_mil_to_2, target_mean_two, target_mil, target_two)

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
    print("it's val")
    print_section("", [summary['epoch'] + 1, step + 1, loss_crf_final, acc_data, time_spent],
                  mode="sample")

    save2record(acc_data, acc_data2, acc_data3, acc_data_mil, fn_mean, fn_mil, fn_ori, fp_mean, fp_mil, fp_ori,
                loss_crf, loss_crf_final, loss_crf_mean, loss_crf_mil, loss_crf_ori, record_list, tn_mean, tn_mil,
                tn_ori, tp_mean, tp_mil, tp_ori)

    save_data2summary(acc_data2, acc_data_mil, fn_mil, fp_mil, loss_crf_final, loss_crf_mil, summary, tn_mil, tp_mil)

    save_key(acc_data2, acc_data_mil, fn_mean, fn_mil, fn_ori, fp_mean, fp_mil, fp_ori, loss_crf_final, loss_crf_mil,
             record_key_dict_single, summary, tn_mean, tn_mil, tn_ori, tp_mean, tp_mil, tp_ori)

    # if summary['step'] % cfg['log_every'] == 0:
    #     summary_writer.add_scalar('train_epoch/loss in selector', loss_crf_final, summary['step'])
    #     summary_writer.add_scalar('train_epoch/acc in selector', acc_data2, summary['step'])
    #     summary_writer.add_scalar('train fpr/tpr in selector', 1 - fpr_ori, fpr_ori)
    #     print_section("", output_crf_ori, print_function="print")
    #     print_section("", target_crf_clone, print_function="print")

    return time_now


def save2record(acc_data, acc_data2, acc_data3, acc_data_mil, fn_mean, fn_mil, fn_ori, fp_mean, fp_mil, fp_ori,
                loss_crf, loss_crf_final, loss_crf_mean, loss_crf_mil, loss_crf_ori, record_list, tn_mean, tn_mil,
                tn_ori, tp_mean, tp_mil, tp_ori):
    record_list[-1].append(float(loss_crf_ori.cpu()))
    record_list[-1].append(float(loss_crf_mean.cpu()))
    record_list[-1].append(float(loss_crf.cpu()))
    record_list[-1].append(float(loss_crf_mil.cpu()))
    record_list[-1].append(float(loss_crf_final.cpu()))
    record_list[-1].append(acc_data)
    record_list[-1].append(acc_data2)
    record_list[-1].append(acc_data3)
    record_list[-1].append(acc_data_mil)
    record_list[-1].append(fp_ori)
    record_list[-1].append(fn_ori)
    record_list[-1].append(fp_mean)
    record_list[-1].append(fn_mean)
    record_list[-1].append(fp_mil)
    record_list[-1].append(fn_mil)
    record_list[-1].append(tp_ori)
    record_list[-1].append(tn_ori)
    record_list[-1].append(tp_mean)
    record_list[-1].append(tn_mean)
    record_list[-1].append(tp_mil)
    record_list[-1].append(tn_mil)


def calc_relate(predict_crf, predict_crf_mean, predict_mil_to_2, target_mean_two, target_mil, target_two):
    err_ori, fpr_ori, fnr_ori = calc_err(predict_crf.cpu(), target_two.cpu())
    err_mean, fpr_mean, fnr_mean = calc_err(predict_crf_mean.cpu(), target_mean_two.cpu())
    err_mil, fpr_mil, fnr_mil = calc_err(predict_mil_to_2.cpu(), target_mil.cpu())
    fp_ori, tp_ori, fn_ori, tn_ori = calc_num_class(predict_crf.cpu(), target_two.cpu())
    fp_mean, tp_mean, fn_mean, tn_mean = calc_num_class(predict_crf_mean.cpu(), target_mean_two.cpu())
    fp_mil, tp_mil, fn_mil, tn_mil = calc_num_class(predict_mil_to_2.cpu(), target_mil.cpu())
    return fn_mean, fn_mil, fn_ori, fnr_mean, fnr_mil, fnr_ori, fp_mean, fp_mil, fp_ori, fpr_mean, fpr_mil, fpr_ori, tn_mean, tn_mil, tn_ori, tp_mean, tp_mil, tp_ori


def save_key(acc_data2, acc_data_mil, fn_mean, fn_mil, fn_ori, fp_mean, fp_mil, fp_ori, loss_crf_final, loss_crf_mil,
             record_key_dict_single, summary, tn_mean, tn_mil, tn_ori, tp_mean, tp_mil, tp_ori):
    record_key_dict_single['epoch'] = summary['epoch']
    record_key_dict_single['loss'] = float(loss_crf_final.cpu())
    record_key_dict_single['loss_mil'] = float(loss_crf_mil.cpu())
    record_key_dict_single['acc'] = acc_data_mil
    record_key_dict_single['acc_crf'] = acc_data2
    record_key_dict_single['fp'] = fp_mil
    record_key_dict_single['fn'] = fn_mil
    record_key_dict_single['tp'] = tp_mil
    record_key_dict_single['tn'] = tn_mil
    record_key_dict_single['fp_ori'] = fp_ori
    record_key_dict_single['fn_ori'] = fn_ori
    record_key_dict_single['tp_ori'] = tp_ori
    record_key_dict_single['tn_ori'] = tn_ori
    record_key_dict_single['fp_mean'] = fp_mean
    record_key_dict_single['fn_mean'] = fn_mean
    record_key_dict_single['tp_mean'] = tp_mean
    record_key_dict_single['tn_mean'] = tn_mean


def save_data2summary(acc_data2, acc_data_mil, fn_mil, fp_mil, loss_crf_final, loss_crf_mil, summary, tn_mil, tp_mil):
    summary['loss'] += float(loss_crf_final.cpu())
    summary['loss_mil'] += float(loss_crf_mil.cpu())
    summary['acc'] += acc_data_mil
    summary['acc_crf'] += acc_data2
    summary['fp'] += fp_mil
    summary['fn'] += fn_mil
    summary['tp'] += tp_mil
    summary['tn'] += tn_mil


def all_acc_calc(predict3, predict_crf, predict_crf_mean, predict_mil_to_2, target_crf_clone, target_mean_two,
                 target_mil, target_two):
    acc_data3 = acc_calculate(predict3, target_crf_clone)
    acc_data = acc_calculate(predict_crf, target_two)
    acc_data2 = acc_calculate(predict_crf_mean, target_mean_two)
    acc_data_mil = acc_calculate(predict_mil_to_2, target_mil)
    return acc_data, acc_data2, acc_data3, acc_data_mil


def reform_relate(output_crf_ori, pre_value, predict_mil, target_crf, target_crf_clone):
    target_mean_two = predict_reform(target_crf)
    target_two = predict_reform(target_crf_clone)
    predict3 = predict_reform(output_crf_ori, mode=1, threshold=[pre_value - 0.1, pre_value + 0.1, pre_value])
    predict_mil_to_2 = predict_reform(predict_mil)
    return predict3, predict_mil_to_2, target_mean_two, target_two


def print_relate(predict_crf, predict_crf_mean, step, target_mean_two, target_two, train_loader):
    print("predict crf mean:")
    print(predict_crf_mean)
    print("target crf mean:")
    print(target_mean_two)
    print_section("num 1 in predict", (predict_crf == 1).sum().item())
    print_section("num 0 in predict", (predict_crf == 0).sum().item())
    print_section("num 1 in target", (target_two == 1).sum().item())
    print_section("num 0 in target", (target_two == 0).sum().item())
    print_section("step/step all", [step + 1, len(train_loader)], mode="double")


def print_complex(print_every, output_crf, predict_crf, step):
    if step % print_every == 0:
        print("output crf mean:")
        print(output_crf)
        print("predict crf all:")
        print(predict_crf)


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
    dataset.record_result_mil(predict_final, patch, position, 1)


def transfer2cuda(data_crf, data_mil, target_crf, target_mil, pre_value):
    data_crf = Variable(data_crf.cuda(non_blocking=True))
    data_mil = Variable(data_mil.cuda(non_blocking=True))
    target_crf = Variable(target_crf.cuda(non_blocking=True))
    target_mil = Variable(target_mil.cuda(non_blocking=True))
    target_crf = change_form(target_crf, pre_value=pre_value)
    target_crf_mean = target_crf.clone().detach().mean(dim=1)
    return data_crf, data_mil, target_crf_mean, target_crf, target_mil
