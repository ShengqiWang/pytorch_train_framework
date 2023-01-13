import torch
import numpy as np
# import pickle
# import os


class YourDataset(torch.utils.data.Dataset):
    def __init__(self, data_cls='trian'):
        self.num = 10000
        data = torch.linspace(1, 2, steps=self.num)
        label = data**2
        self.data, self.label = data, label

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def collate_fn(self, samples):
        data_list = []
        label_list = []

        for batch_no, sample in enumerate(samples):
            data, label = sample
            data_list.append(data.unsqueeze(0))
            label_list.append(label.unsqueeze(0))
        data = torch.cat(data_list, dim=0)
        label = torch.cat(label_list, dim=0)
        return data, label



if __name__ == '__main__':
    dataset = YourDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1,
                                                drop_last=True, collate_fn = dataset.collate_fn)
    for i, (data, label) in enumerate(train_loader):
        print(data.shape, label.shape)




