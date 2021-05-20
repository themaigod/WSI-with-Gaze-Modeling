from .tool import *


def vaild_crf(base_dataset, cfg, dataloader_valid_crf, loss_fn_with_sigmoid, loss_fn_without_sigmoid, model_crf,
              summary_train_crf, summary_valid_crf):
    time_now = time.time()
    summary_valid_crf = valid_epoch_crf(summary_valid_crf, cfg, model_crf, loss_fn_with_sigmoid,
                                        loss_fn_without_sigmoid,
                                        dataloader_valid_crf, base_dataset)
    time_spent = time.time() - time_now
    logging.info(
        '{}, Epoch : {}, Step : {}, Validation Loss : {:.5f}, '
        'Validation Acc : {:.3f}, Run Time : {:.2f}'
            .format(
            time.strftime("%Y-%m-%d %H:%M:%S"), summary_train_crf['epoch'],
            summary_train_crf['step'], summary_valid_crf['loss'],
            summary_valid_crf['acc'], time_spent))
    return summary_valid_crf


def valid_epoch_crf(summary, cfg, model, loss_fn, loss_fn2,
                    dataloader, base_dataset):
    model.eval()

    loss_sum = 0
    acc_sum = 0
    acc_sum2 = 0
    time_now = time.time()
    for step, (data, target, patch, position) in enumerate(dataloader):
        with torch.no_grad():
            data = Variable(data.cuda(non_blocking=True))
            target = Variable(target.cuda(non_blocking=True))
            target = change_form(target, pre_value=0.9)
            tar = target.clone()
            target = target.mean(dim=1)

            output = model(data)

            output, output2, output3, predict1, predict2, predict3 = output_form(output, pre_value=0.9, mode=2)
            if step == 0:
                print_section("", output, print_function="print")
                print_section("", output2, print_function="print")
            loss1, loss2, loss = get_loss(output, tar, loss_fn, output2, target, loss_fn2, ratio=[0.8, 0.2])

            loss_name = ['loss1', 'loss2', 'loss_total']
            loss_value = [loss1, loss2, loss]
            print_section(loss_name, loss_value, mode="all_loss")

            target_mean_two = predict_reform(target)
            target_two = predict_reform(tar)

            print_section(['output_mean\n', 'target_mean\n', 'output_three\n', 'target_two\n'],
                          [output2, target, predict2, target_mean_two], mode="show_out_pre",
                          print_function="print")

            loss_data = loss.item()
            time_spent = time.time() - time_now
            time_now = time.time()
            print_section("step/step all", [step + 1, len(dataloader)], mode="double")

            print_section("num 1 in predict", (predict1 == 1).sum().item())
            print_section("num 0 in predict", (predict1 == 0).sum().item())

            print_section("num all", predict1.numel())

            print_section("num 1 in target", (target_two == 1).sum().item())
            print_section("num 0 in target", (target_two == 0).sum().item())

            acc_data = acc_calculate(predict1, target_two)
            acc_data2 = acc_calculate(predict2, target_mean_two)
            acc_data3 = acc_calculate(predict3, tar)

            result_patch = np.array((predict2 == target_mean_two).cpu())
            patch = np.array(patch).tolist()
            position = np.array(position).tolist()
            base_dataset.get_index(result_patch, patch, position, 1)

            print_section(['acc_two: ', 'acc_mean: ', 'acc_three: '], [acc_data, acc_data2, acc_data3],
                          mode="show_out_pre")
            print_section("", [summary['epoch'] + 1, step + 1, loss_data, acc_data, time_spent],
                          mode="sample")

            if (step + 1) % cfg['log_every'] == 0:
                print_section("", output3, print_function="print")
                print_section("", tar, print_function="print")

            loss_sum += loss_data
            acc_sum += acc_data
            acc_sum2 += acc_data2
    steps = len(dataloader)

    print_section("val acc2: ", acc_sum2 / steps)

    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps
    summary['epoch'] += 1

    return summary

