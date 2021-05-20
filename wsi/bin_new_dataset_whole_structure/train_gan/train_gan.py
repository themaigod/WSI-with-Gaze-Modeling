import torch.nn

from .tool import *


def train_gan(dataloader, model_crf, model_mil, dataset, summary, num_workers, batch_size):
    model_crf.eval()
    model_mil.eval()
    for step, (data, target, patch, position) in enumerate(dataloader):
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
        predict_mil = model_mil(data)
        predict_final[index] = predict_mil
        dataset.get_index(predict_final, patch, position, 0)
    train_dataset = dataset.group_max()
    train_loader = DataLoader(train_dataset,batch_size=batch_size, num_workers=num_workers, drop_last=True)
    model_mil.train()
    model_crf.train()
    for step, (data, data_mil, target, patch, position) in enumerate(train_loader):
        predict_crf = model_crf(data)
        predict_crf = torch.sigmoid(predict_crf)
        predict_crf = torch_round_with_backward(predict_crf)
        data_mil = predict_crf * data_mil




