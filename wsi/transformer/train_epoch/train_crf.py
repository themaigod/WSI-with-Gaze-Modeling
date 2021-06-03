from .tool import *





def train_epoch_crf(summary, summary_writer, cfg, model, loss_fn, loss_fn2, optimizer,
                    dataloader, base_dataset):
    model.train()

    time_now = time.time()
    for step, (data, target, patch, position) in enumerate(dataloader):
        data = Variable(data.cuda(non_blocking=True))
        target = Variable(target.cuda(non_blocking=True))
        target = change_form(target, pre_value=0.9)
        tar = target.clone()
        arget = target.mean(dim=1)
        output = model(data)
        output, output2, output3, predict1, predict2, predict3 = output_form(output, pre_value=0.9, mode=2)
        if step == 0:
            print_section("", output, print_function="print")
            print_section("", output2, print_function="print")
        loss1, loss2, loss = get_loss(output, tar, loss_fn, output2, target, loss_fn2, ratio=[0.8, 0.2])

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

        print_section("num 1 in predict", (predict1 == 1).sum().item())
        print_section("num 0 in predict", (predict1 == 0).sum().item())

        print_section("num all", predict1.numel())

        print_section("num 1 in target", (target_two == 1).sum().item())
        print_section("num 0 in target", (target_two == 0).sum().item())

        acc_data = acc_calculate(predict1, target_two)
        acc_data2 = acc_calculate(predict2, target_mean_two)
        acc_data3 = acc_calculate(predict3, tar)
        print_section(['acc_two: ', 'acc_mean: ', 'acc_three: '], [acc_data, acc_data2, acc_data3], mode="show_out_pre")
        print_section("", [summary['epoch'] + 1, summary['step'] + 1, loss_data, acc_data, time_spent], mode="sample")

        result_patch = np.array((predict2 == target_mean_two).cpu())
        patch = np.array(patch).tolist()
        position = np.array(position).tolist()
        base_dataset.get_index(result_patch, patch, position, 0)

        summary['step'] += 1

        if summary['step'] % cfg['log_every'] == 0:
            summary_writer.add_scalar('train_epoch/loss', loss_data, summary['step'])
            summary_writer.add_scalar('train_epoch/acc', acc_data2, summary['step'])
            print_section("", output3, print_function="print")
            print_section("", tar, print_function="print")

    summary['epoch'] += 1

    return summary