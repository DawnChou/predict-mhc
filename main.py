import os
import numpy as np
import torch
import pandas as pd
import pickle
import matplotlib
import argparse
from collections import defaultdict, Counter
from models.Text1DCNN_model import TextCNN1d
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from utils import Logger, AverageMeter, save_checkpoint
from build_dataset import build_dataset
from training import train_epoch
from validation import validate_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="datasets/HLA_CDR/merge_dataset_3k.pickle", type=str)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epoches", default=100, type=int)
    parser.add_argument("--n_classes", default=100, type=int)
    parser.add_argument("--plateau_patience", default=5, type=int)
    parser.add_argument("--train_logger_path", default="log/train_logger.log", type=str)
    parser.add_argument("--train_batch_logger_path", default="log/train_batch_logger.log", type=str)
    parser.add_argument("--validation_logger_path", default="log/validation_logger.log", type=str)
    parser.add_argument("--result_path", default="log/save_models", type=str)

    args = parser.parse_args()

    with open(args.data_path, "rb") as f:
        data = pickle.load(f)
    combine_labels = []
    for sample_id, values in data.items():
        combine_labels.extend(data[sample_id]['HLA_label'])
    labels_count = sorted(Counter(combine_labels).items(),key=lambda item:item[0])
    pos_weight = torch.ones(args.n_classes)
    for label, count in labels_count:
        pos_weight[label] = int(900/count)

    train_logger = Logger(args.train_logger_path, ['epoch','loss', 'Accuracy', 'Precision', 'Recall','lr'])
    train_batch_logger_path = Logger(args.train_batch_logger_path, ['epoch','batch','iter','loss','Accuracy','Precision', 'Recall', 'lr'])
    validation_logger = Logger(args.validation_logger_path, ['epoch','loss', 'Accuracy', 'Precision', 'Recall'])

    train_dataset = build_dataset(args.data_path, 'training')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)
    validation_dataset = build_dataset(args.data_path, 'validation')
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, pin_memory=True)


    model = TextCNN1d(n_filters=32, filter_sizes=[2,3,4,5,6,7], embedding_dim=1, n_classes=args.n_classes)
    model.to()
    optimizer = Adam(model.parameters(), lr= args.learning_rate)
    #criterion= BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion= BCEWithLogitsLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.plateau_patience)

    for epoch in np.arange(args.epoches):
        train_epoch(model, epoch, train_dataloader, criterion, optimizer, batch_logger=train_batch_logger_path, epoch_logger=train_logger)
        if epoch % 10 == 0:
            save_file_path = os.path.join(args.result_path, 'save_{}.pth'.format(epoch))
            save_checkpoint(save_file_path, epoch, model, optimizer)
        val_loss, accuracy = validate_model(model, epoch, validation_dataloader, criterion, logger=validation_logger)
        scheduler.step(val_loss)