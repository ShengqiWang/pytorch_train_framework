import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 128),
                                # nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Linear(128, 1))
    def forward(self, x):
        y = self.net(x.unsqueeze(1)).squeeze(1)
        return y

# class MyLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bce_loss = nn.BCEWithLogitsLoss()
#     def forward(self, pred, label):
#         loss_msg = {}
#         bce_loss_value = self.bce_loss(pred, label)
#         loss = bce_loss_value

#         loss_msg['bceloss'] = bce_loss_value
#         loss_msg['loss_total'] = loss
#         return loss, loss_msg

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    def forward(self, pred, label):
        loss_msg = {}
        mse_loss_value = self.mse_loss(pred, label)
        loss = mse_loss_value
        loss_msg['mseloss'] = mse_loss_value
        loss_msg['loss_total'] = loss
        return loss, loss_msg


if __name__=='__main__':
    net = Net()
    data = torch.zeros(100, 2)
    y = net(data)
    print(y.shape)

