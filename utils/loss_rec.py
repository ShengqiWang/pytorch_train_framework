import torch
import os 
from torch.utils.tensorboard import SummaryWriter
from .multi_gpu import *

class LossRecorder():
    def __init__(self, process_id, is_tb=False, record_path='./loss_rec', record_name='record'):
        super().__init__()
        self.reset()
        self.total_iter = 0
        self.is_tb = is_tb
        self.process_id = process_id
        if self.is_tb and self.process_id:
            os.makedirs(record_path, exist_ok=True)
            self.tbwriter = SummaryWriter(log_dir=record_path)
        
            
    def reset(self):
        self.loss_num = 0
        self.is_start = True
        self.lossmsg = {}
        
    def record(self, lossmsg):
        reduce_loss={}
        
        with torch.no_grad():
            for key in lossmsg.keys():
                reduce_loss[key] = reduce_tensor(lossmsg[key]).item()
                
        if self.is_start:
            for key in reduce_loss.keys():
                self.lossmsg[key] = reduce_loss[key]
            self.is_start = False
        else:
            for key in reduce_loss.keys():
                self.lossmsg[key] += reduce_loss[key]
        
        if self.is_tb and self.process_id:
            for key in reduce_loss.keys():
                self.tbwriter.add_scalar(key, reduce_loss[key], self.total_iter)
                
        self.loss_num += 1
        self.total_iter += 1

    def display(self, iter):
        print("iter", iter, " || | ", end = "")
        for key in self.lossmsg.keys():
            print(key+": " , end="")
            print("%.3f" % (self.lossmsg[key]/self.loss_num), end="")
            print(" | ", end="")
        print("||")
        self.reset()
        
    