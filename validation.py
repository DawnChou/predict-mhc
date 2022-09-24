import time
import torch
import os
import sys
from utils import AverageMeter, Hamming_Loss, Accuracy, Precision, Recall

def validate_model(model, epoch, validation_dataloader, criterion, logger):
    
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_dataloader):
            inputs = inputs.to()
            labels = labels.to()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = Accuracy(outputs, labels)
            precision = Precision(outputs, labels)
            recall = Recall(outputs, labels)
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            precisions.update(precision, inputs.size(0))
            recalls.update(recall, inputs.size(0))

            print('Validation: [{0}][{1}/{2}]\t'
                'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'
                'Precision {precision.val:.3f} ({precision.avg:.3f})\t'
                'Recall {recall.val:.3f} ({recall.avg:.3f})'.format(epoch,
                                                                    i + 1,
                                                                    len(validation_dataloader),
                                                                    loss=losses,
                                                                    acc=accuracies,
                                                                    precision=precisions,
                                                                    recall=recalls))
    
    if logger is not None:
        logger.log({'epoch': epoch, 'loss': losses.avg, 'Accuracy': accuracies.avg, 'Precision': precisions.avg,
                    'Recall': recalls.avg})
    
    return losses.avg, accuracies.avg