import torch
from torch import nn
from torch import optim
from dataset.yourdataset import *
from model.yourmodel import *
from train_parse import *
import os

from utils.multi_gpu import *
from utils.loss_rec import *
from tqdm import tqdm

from val import *

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

def train_one_epoch(epoch, net, trloader, trsampler, optimizer, lr_sche, lossfun, recorder, is_gpu=False, is_multigpu=False):
    net.train()
    if not is_multigpu or GPUNO == 0:
        print('Epoch:', epoch+1)
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
    lr_sche.step()   # lr_sche by epoch, if you want to update by iter, please put this line in the loop
    return net

def train():
    trloader, trsampler = get_dataloader(YourDataset(), MULTIGPU)
    tsloader, _ = get_dataloader(YourDataset(), MULTIGPU)
    net = Net()
    # net.module.load_state_dict(torch.load('para.pkl'))
    net = get_model(net, is_gpu=GPU, is_multigpu=MULTIGPU)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    lr_sche = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestone, gamma=0.1)
    loss_fun = MyLoss()
    recorder = LossRecorder(is_tb=args.use_tensorboard, process_id=GPUNO)
    for epoch_no in range(args.epoch_num):
        net = train_one_epoch(epoch_no, net, trloader, trsampler,\
                              optimizer, lr_sche, loss_fun, recorder, \
                              is_gpu=GPU, is_multigpu=MULTIGPU)
        val(net, tsloader, gpu_no=GPUNO, is_gpu=GPU, is_multigpu=MULTIGPU)
        
        # torch.save(net.module.state_dict(), 'para.pkl')


args = parse_args()

MULTIGPU = args.is_multigpu   # use multiple gpu or not
GPU = args.is_gpu    # use single gpu or not
GPUNO = args.gpu_no  # single gpu no

if __name__ == '__main__':
    if MULTIGPU:
        GPUNO = int(os.environ["LOCAL_RANK"])
        device_ids = range(torch.cuda.device_count())
        torch.distributed.init_process_group(backend="nccl")

    torch.cuda.set_device(GPUNO)
    print("gpu_no:", GPUNO)
    train()