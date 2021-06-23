import time
import logging
import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Function


def calc_err(pred, real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum()) / pred.size
    fpr = float(np.logical_and(pred == 1, neq).sum()) / (real == 0).sum()
    fnr = float(np.logical_and(pred == 0, neq).sum()) / (real == 1).sum()
    return err, fpr, fnr


def calc_num_class(pred, real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum()) / pred.size
    fp = float(np.logical_and(pred == 1, neq).sum())
    tp = (real == 0).sum() - fp
    fn = float(np.logical_and(pred == 0, neq).sum())
    tn = (real == 1).sum() - fn
    return fp, tp, fn, tn


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
            for i in range(len(data)):
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
    if pre_value != 1:
        base = 1 - pre_value
        tensor = tensor * base
        if len(tensor.shape) == 1:
            if tensor.mean() > 0:
                tensor[tensor < 1] = pre_value
            else:
                tensor[:] = 0
        else:
            tensor[tensor.mean(dim=1) > 0, :] = tensor[tensor.mean(dim=1) > 0, :] + pre_value
    else:
        if len(tensor.shape) == 1:
            if tensor.mean() > 0:
                tensor[tensor < 1] = 1
            else:
                tensor[:] = 0
        else:
            tensor[tensor.mean(dim=1) > 0, :] = 1
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
    elif mode == 3:
        output = torch.sigmoid(output)
        output_clone = output.clone()
        output_mean = output.mean(dim=1)
        predict = predict_reform(output_clone)
        predict_mean = predict_reform(output_mean)
        return output_clone, output_mean, predict, predict_mean


class TorchRound(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# torch_round_with_backward = TorchRound.apply


def torch_round_with_backward(output_crf):
    output_crf_round = output_crf.clone()
    output_crf_round[output_crf_round >= 0.5] = 1.0
    output_crf_round[output_crf_round < 0.5] = 0
    return output_crf_round
