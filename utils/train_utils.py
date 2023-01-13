import torch

import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp.clone()
        dist.reduce(reduced_inp, dst=0)
        # dist.all_reduce(reduced_inp, op=dist.reduce_op.SUM)
        reduced_inp = reduced_inp/world_size
    return reduced_inp


def sum_tensor(inp):
    """
    Reduce the loss from all processes so that process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp.clone()
        dist.reduce(reduced_inp, dst=0)
        # dist.all_reduce(reduced_inp, op=dist.reduce_op.SUM)
        reduced_inp = reduced_inp
    return reduced_inp

def gather_tensor(inp):
    world_size = get_world_size()
    if world_size < 2:
        return [inp]
    with torch.no_grad():
        var_list = []
        gather_inp = inp.clone()
        dist.reduce(gather_inp, var_list)
    return var_list

class LossRecorder():
    def __init__(self):
        super().__init__()
        self.is_start = True
        self.loss_num = 0
        
    def record(self, lossmsg):
        with torch.no_grad():
            if self.is_start:
                self.lossmsg = {}
                for key in lossmsg.keys():
                    self.lossmsg[key] = reduce_tensor(lossmsg[key]).item()
            else:
                for key in lossmsg.keys():
                    self.lossmsg[key] += reduce_tensor(lossmsg[key]).item()
        self.loss_num += 1

    def display(self, iter):
        print("iter", iter, " ||| ", end = "")
        for key in self.lossmsg.keys():
            print(key+": " , end="")
            print("%.3f" % (self.lossmsg[key]/self.loss_num), end="")
            print(" ||| ", end="")
        print("\n")
        self.loss_num = 0
        self.is_start = True