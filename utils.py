import csv
import random
from functools import partialmethod

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path,'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


def save_checkpoint(save_file_path, epoch, model, optimizer):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        #'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)



def Hamming_Loss(y_pred, y_true):
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    temp=0
    for i in range(y_true.shape[0]):
        temp += (y_true[i] == y_pred[i]).size()[0] - torch.count_nonzero(y_true[i] == y_pred[i])
    return (temp/(y_true.shape[0] * y_true.shape[1])).item()

# def Accuracy(y_pred, y_true):
#     y_pred[y_pred>=0.5] = 1
#     y_pred[y_pred<0.5] = 0
    
#     temp = 0
#     for i in range(y_true.shape[0]):
#         label_pred = torch.argwhere(y_pred[i]==1).reshape(-1).numpy()
#         label_true = torch.argwhere(y_true[i]==1).reshape(-1).numpy()
#         temp += len(set(label_pred)&set(label_true))/len(set(label_pred).union(set(label_true)))
    
#     return temp / y_true.shape[0]

def Accuracy(y_pred, y_true):

    temp = 0
    for i in range(y_true.shape[0]):
        label_pred = torch.argmax(y_pred, 1)
        label_true = torch.argwhere(y_true[i]==1).reshape(-1).numpy()
        if label_pred == label_true:
            temp += 1
    return temp / y_true.shape[0]

def Precision(y_pred, y_true):
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    
    temp = 0
    for i in range(y_true.shape[0]):
        label_pred = torch.argwhere(y_pred[i]==1).reshape(-1).numpy()
        label_true = torch.argwhere(y_true[i]==1).reshape(-1).numpy()
        if len(set(label_pred)) == 0:
            temp += 0
        else:
            temp += len(set(label_pred)&set(label_true))/len(set(label_pred))
    
    return temp / y_true.shape[0]

def Recall(y_pred, y_true):
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    
    temp = 0
    for i in range(y_true.shape[0]):
        label_pred = torch.argwhere(y_pred[i]==1).reshape(-1).numpy()
        label_true = torch.argwhere(y_true[i]==1).reshape(-1).numpy()
        temp += len(set(label_pred)&set(label_true))/len(set(label_true))
    
    return temp / y_true.shape[0]
