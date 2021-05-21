import torch.nn

from .tool import *


def train_gan(dataloader, model_crf, model_mil, dataset, summary, num_workers, batch_size, loss_fn, optimizer_crf,
              optimizer_mil, top_k, pre_value, summary_writer, cfg, record_list_total):
    model_crf.eval()
    model_mil.eval()
    time_now = time.time()
    for step, (data_crf, target_crf, data_mil, target_mil, patch, position) in enumerate(dataloader):
        # predict_final = np.zeros((len(data)))
        # if step != 0 or summary['epoch'] != 0:
        #     output_crf = model_crf(data)
        #     index = []
        #     for i in range(len(output_crf)):
        #         if output_crf[i] >= 0.5:
        #             index.append(i)
        #     data_mil = data_mil[index]
        # else:
        #     index = list(range(len(data)))
        #     data_mil = data_mil
        data_mil = Variable(data_mil.cuda(non_blocking=True))
        predict_mil = model_mil(data_mil)
        predict_mil = torch.sigmoid(predict_mil)
        record_result(dataset, patch, position, predict_mil)
    dataset.slide_max(0)
    train_dataset = dataset.produce_dataset_mil(0, top_k)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    model_mil.train()
    model_crf.train()
    record_list = []
    for step, (data_crf, target_crf, data_mil, target_mil, patch, position) in enumerate(train_loader):
        record_list.append([])
        data_crf, data_mil, target_crf, target_crf_clone, target_mil = transfer2cuda(data_crf, data_mil, target_crf,
                                                                                     target_mil)
        output_crf_ori = model_crf(data_crf)
        output_crf_ori, output_crf, predict_crf, predict_crf_mean = output_form(output_crf_ori, pre_value=0.8, mode=3)
        index = access_index(output_crf)
        output_crf = torch_round_with_backward(output_crf)
        data_mil = model_get_data_mil(data_mil, output_crf)
        predict_mil = model_mil(data_mil)
        predict_mil = torch.sigmoid(predict_mil)

        loss_crf_mean = loss_fn(output_crf, target_crf)
        loss_crf_ori = loss_fn(output_crf_ori, target_crf_clone)
        loss_crf = 0.2 * loss_crf_ori + 0.8 * loss_crf_mean
        loss_crf_mil = loss_fn(predict_mil, target_mil)
        loss_crf_final = loss_crf * 0.4 + loss_crf_mil * 0.6

        loss_mil = None
        if len(index) != 0:
            loss_mil = loss_fn(predict_mil[index], target_mil[index])

        optimizer_crf.zero_grad()
        loss_crf_final.backward(retain_graph=True)
        optimizer_crf.step()

        optimizer_mil.zero_grad()
        if len(index) != 0:
            loss_mil.backward()
            optimizer_mil.step()

        time_now = show_crf(loss_crf, loss_crf_final, loss_crf_mean, loss_crf_mil, loss_crf_ori, output_crf,
                            output_crf_ori, pre_value, predict_crf, predict_crf_mean, predict_mil, step, summary,
                            target_crf, target_mil, target_crf_clone, time_now, train_loader, summary_writer, cfg,
                            record_list)

        show_mil(index, loss_mil, predict_mil[index], step, summary, target_mil[index], time_now, summary_writer, cfg,
                 record_list)

        summary['step'] += 1

    record_list_total.append(record_list)


def show_mil(index, loss_mil, predict_mil, step, summary, target_mil, time_now, summary_writer, cfg, record_list):
    #         record 添加 loss_mil acc_mil fpr_mil fnr_mil
    if len(index) != 0:
        if step % 3 == 0:
            print("output mil:")
            print(predict_mil)
        predict_mil_to_2 = predict_reform(predict_mil)
        print("output mil:")
        print(predict_mil_to_2)
        print("target_mil")
        print(target_mil)
        print_section("num 1 in predict", (predict_mil_to_2 == 1).sum().item())
        print_section("num 0 in predict", (predict_mil_to_2 == 0).sum().item())
        print_section("num 1 in target", (target_mil == 1).sum().item())
        print_section("num 0 in target", (target_mil == 0).sum().item())
        print("loss_mil: ", str(loss_mil))
        acc_mil = acc_calculate(predict_mil_to_2, target_mil)
        print("acc_mil:", acc_mil)
        err_mil, fpr_mil, fnr_mil = calc_err(predict_mil_to_2, target_mil)
        fp_fn_name = ['False Positive Rate in mil: ', 'False Negative Rate in mil: ']

        print_section(fp_fn_name, [fpr_mil, fnr_mil], mode="show_out_pre")

        time_spent = time.time() - time_now

        print_section("", [summary['epoch'] + 1, step + 1, loss_mil, acc_mil, time_spent],
                      mode="sample")

        record_list[-1].append(loss_mil)
        record_list[-1].append(acc_mil)
        record_list[-1].append(fpr_mil)
        record_list[-1].append(fnr_mil)

        if summary['step'] % cfg['log_every'] == 0:
            summary_writer.add_scalar('train_epoch/loss in mil', loss_mil, summary['step'])
            summary_writer.add_scalar('train_epoch/acc in mil', acc_mil, summary['step'])
            summary_writer.add_scalar('train fpr/tpr in mil', 1 - fpr_mil, fpr_mil)


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

    err_ori, fpr_ori, fnr_ori = calc_err(predict_crf, target_two)
    err_mean, fpr_mean, fnr_mean = calc_err(predict_crf_mean, target_mean_two)
    err_mil, fpr_mil, fnr_mil = calc_err(predict_mil_to_2, target_mil)

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
    print_section("", [summary['epoch'] + 1, step + 1, loss_crf_final, acc_data, time_spent],
                  mode="sample")

    record_list[-1].append(loss_crf_ori)
    record_list[-1].append(loss_crf_mean)
    record_list[-1].append(loss_crf)
    record_list[-1].append(loss_crf_mil)
    record_list[-1].append(loss_crf_final)
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

    if summary['step'] % cfg['log_every'] == 0:
        summary_writer.add_scalar('train_epoch/loss in selector', loss_crf_final, summary['step'])
        summary_writer.add_scalar('train_epoch/acc in selector', acc_data2, summary['step'])
        summary_writer.add_scalar('train_epoch/acc in total', acc_data_mil, summary['step'])
        summary_writer.add_scalar('train fpr/tpr in selector', 1 - fpr_ori, fpr_ori)
        summary_writer.add_scalar('train fpr/tpr in total', 1 - fpr_mil, fpr_mil)
        print_section("", output_crf_ori, print_function="print")
        print_section("", target_crf_clone, print_function="print")

    return time_now


def model_get_data_mil(data_mil, output_crf):
    output_crf = output_crf.clone()
    for i in range(len(data_mil.shape)):
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
    predict_final = np.array(predict_mil.cpu())
    # dataset.get_index(predict_final, patch, position, 0)
    patch = np.array(patch).tolist()
    position = np.array(position).tolist()
    dataset.record_result_mil(predict_final, patch, position, 0)


def transfer2cuda(data_crf, data_mil, target_crf, target_mil):
    data_crf = Variable(data_crf.cuda(non_blocking=True))
    data_mil = Variable(data_mil.cuda(non_blocking=True))
    target_crf_clone = target_crf.cuda(non_blocking=True).clone().detach()
    target_crf = Variable(target_crf.cuda(non_blocking=True))
    target_mil = Variable(target_mil.cuda(non_blocking=True))
    target_crf = change_form(target_crf, pre_value=0.8)
    return data_crf, data_mil, target_crf, target_crf_clone, target_mil
