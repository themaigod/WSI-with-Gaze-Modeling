import torch.nn

from .tool import *


def train_gan(dataloader, model_crf, model_mil, dataset, summary, num_workers, batch_size, loss_fn, optimizer_crf,
              optimizer_mil):
    model_crf.eval()
    model_mil.eval()
    for step, (data, data_mil, target, patch, position) in enumerate(dataloader):
        predict_final = np.zeros((len(data)))
        if step != 0 or summary['epoch'] != 0:
            predict_crf = model_crf(data)
            index = []
            for i in range(len(predict_crf)):
                if predict_crf[i] >= 0.5:
                    index.append(i)
            data_mil = data[index]
        else:
            index = list(range(len(data)))
            data_mil = data
        predict_mil = model_mil(data_mil)
        predict_final[index] = predict_mil
        dataset.get_index(predict_final, patch, position, 0)
    train_dataset = dataset.group_max()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    model_mil.train()
    model_crf.train()
    for step, (data, data_mil, target_crf, target_mil, patch, position) in enumerate(train_loader):
        predict_crf = model_crf(data)
        predict_crf = torch.sigmoid(predict_crf)
        index = []
        for i in range(len(predict_crf)):
            if predict_crf[i] >= 0.5:
                index.append(i)
        predict_crf = torch_round_with_backward(predict_crf)
        for i in range(len(data.shape)):
            predict_crf = predict_crf.unsqueeze(1)
        # predict_crf = predict_crf.expand((data_mil.shape[0], data_mil.shape[1], data_mil.shape[2], data_mil.shape[3]))
        # 似乎不需要了，pytorch自带广播机制（broadcast）
        data_mil = predict_crf * data_mil
        predict_mil = model_mil(data_mil)
        loss_crf = loss_fn(predict_crf, target_crf)
        loss_crf_mil = loss_fn(predict_mil, target_mil)
        loss_mil = loss_fn(predict_mil[index], target_mil[index])

        loss_crf_final = loss_crf * 0.4 + loss_crf_mil * 0.6

        optimizer_crf.zero_grad()
        loss_crf_final.backward(retain_graph=True)
        optimizer_crf.step()

        optimizer_mil.zero_grad()
        loss_mil.backward()
        optimizer_mil.step()
