from .tool import *


def valid_train_mil(base_dataset, cfg, dataloader_train_mil, loss_fn_with_sigmoid, loss_fn_without_sigmoid, model_mil,
                    summary_train_mil, summary_valid_train_mil):
    time_now = time.time()
    summary_valid_train_mil = valid_epoch_mil(summary_valid_train_mil, cfg, model_mil, loss_fn_with_sigmoid,
                                              loss_fn_without_sigmoid,
                                              dataloader_train_mil, base_dataset)
    time_spent = time.time() - time_now
    logging.info(
        '{}, Epoch : {}, Step : {}, Validation Loss : {:.5f}, '
        'Validation Acc : {:.3f}, Run Time : {:.2f}'
            .format(
            time.strftime("%Y-%m-%d %H:%M:%S"), summary_train_mil['epoch'],
            summary_train_mil['step'], summary_valid_train_mil['loss'],
            summary_valid_train_mil['acc'], time_spent))
