import torch
from torch import nn
from dataset.yourdataset import *
from model.yourmodel import *
from train_parse import *
import os

from utils.train_utils import *
from tqdm import tqdm

def get_dataloader(dataset, is_multigpu=False):
    if is_multigpu:
        sampler = DistributedSampler(dataset)
        batch_size_train = int(args.batch_size / get_world_size())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train,
                                                 num_workers=8,
                                                 collate_fn=dataset.collate_fn,
                                                 sampler=sampler)
        return dataloader, sampler
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 num_workers=8,
                                                 shuffle=True,
                                                 collate_fn=dataset.collate_fn)
        return dataloader, None

def get_model(net, is_gpu=False, is_multigpu=False):
    if is_gpu or is_multigpu:
        net.cuda()
    if is_multigpu:
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DDP(net, device_ids=[GPUNO], find_unused_parameters=False)
    return net

def train_one_epoch(epoch, net, trloader, trsampler, optimizer, lossfun, recorder, is_gpu=False, is_multigpu=False):
    net.train()
    for iter, (data, label) in enumerate(trloader):
        if is_gpu or is_multigpu:
            data = data.cuda()
            label = label.cuda()
        if is_multigpu:
            trsampler.set_epoch(epoch)
        optimizer.zero_grad()
        pred = net(data)
        loss, loss_msg = lossfun(pred, label)
        loss.backward()
        
        optimizer.step()
        recorder.record(loss_msg)

        if (not is_multigpu or GPUNO == 0) and iter%args.iter_display==0:
            recorder.display(iter)
    return net

def val(net, tsloader, is_gpu=False, is_multigpu=False):
    # if not is_multigpu or gpu_no==0:
    # if 1:
    net.eval()
    label_list = []
    pred_list = []
    for iter, (data, label) in enumerate(tqdm(tsloader)):
        if is_gpu or is_multigpu:
            data = data.cuda()
            label = label.cuda()
        
        pred = net(data)
        label_list.append(label.cpu())
        pred_list.append(pred.cpu())
    label = torch.cat(label_list, dim=0)
    pred = torch.cat(pred_list, dim=0)
    acc = torch.sum((label-pred)**2)
    acc = torch.tensor([acc]).cuda()
    num = torch.tensor([pred.shape[0]]).cuda()
    acc = sum_tensor(acc)
    num = sum_tensor(num)
    if not is_multigpu or GPUNO==0:
        # print('papa', num)
        acc = (acc/num).item()
        print('acc:', acc)

def train():
    trloader, trsampler = get_dataloader(YourDataset(), MULTIGPU)
    tsloader, _ = get_dataloader(YourDataset(), MULTIGPU)
    net = Net()
    # net.module.load_state_dict(torch.load('para.pkl'))
    net = get_model(net, is_gpu=GPU, is_multigpu=MULTIGPU)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fun = MyLoss()
    recorder = LossRecorder()
    for epoch_no in range(args.epoch_num):
        net = train_one_epoch(epoch_no, net, trloader, trsampler,\
                              optimizer, loss_fun, recorder, \
                              is_gpu=GPU, is_multigpu=MULTIGPU)
        val(net, tsloader, is_gpu=GPU, is_multigpu=MULTIGPU)
        
        # torch.save(net.module.state_dict(), 'para.pkl')


MULTIGPU = 1   # use multiple gpu or not
GPU = 1    # use single gpu or not
GPUNO = 0  # single gpu no

args = parse_args()

if __name__ == '__main__':
    if MULTIGPU:
        GPUNO = int(os.environ["LOCAL_RANK"])
        device_ids = range(torch.cuda.device_count())
        torch.distributed.init_process_group(backend="nccl")

    torch.cuda.set_device(GPUNO)
    print("gpu_no:", GPUNO)
    train()