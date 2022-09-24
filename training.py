import time
import torch
import os
import sys
from utils import AverageMeter, Hamming_Loss, Accuracy, Precision, Recall

def train_epoch(model, epoch, train_dataloader, criterion, optimizer, batch_logger, epoch_logger):

    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to()
        labels = labels.to()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = Accuracy(outputs, labels)
        precision = Precision(outputs, labels)
        recall = Recall(outputs, labels)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        precisions.update(precision, inputs.size(0))
        recalls.update(recall, inputs.size(0))

        # print("=============after update===========")
        # for name, parms in model.named_parameters():	
        #     print('-->name:', name)
        #     print('-->para:', parms)
        #     print('-->grad_requirs:',parms.requires_grad)
        #     print('-->grad_value:',parms.grad)
        #     print("===")


        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        if batch_logger is not None:
                batch_logger.log({
                        'epoch': epoch,
                        'batch': i + 1,
                        'iter': (epoch - 1) * len(train_dataloader) + (i + 1),
                        'loss': losses.val,
                        'Accuracy': accuracies.val,
                        'Precision':precisions.val,
                        'Recall':recalls.val,
                        'lr': current_lr
                    })

        # print('Epoch: [{0}][{1}/{2}]\t'
        #     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #     'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'
        #     'Precision {precision.val:.3f} ({precision.avg:.3f})\t'
        #     'Recall {recall.val:.3f} ({recall.avg:.3f})'.format(epoch,
        #                                                 i + 1,
        #                                                 len(train_dataloader),
        #                                                 loss=losses,
        #                                                 acc=accuracies,
        #                                                 precision=precisions,
        #                                                 recall=recalls))
    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'Accuracy': accuracies.avg,
            'Precision': precisions.avg,
            'Recall': recalls.avg,
            'lr': current_lr
        })

