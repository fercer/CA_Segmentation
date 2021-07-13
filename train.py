import sys
import os
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torch.optim as optim

from sklearn import metrics

from models.unet import UNet
from data.angiodb import AngioDB, AngioTransform, ang_patches_collate
from functools import partial

def train_step(model, loader, criterion, optimizer, scheduler=None):
    logger = logging.getLogger('training')

    model.train()
    for i, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.squeeze(dim=1), target.cuda())
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        prob = out.sigmoid().detach().cpu().numpy().flatten()
        target = target.long().numpy().flatten()
        roc = metrics.roc_auc_score(target, prob, labels=[0, 1])

        pred = (prob > 0.5)
        prec = metrics.precision_score(target, pred, labels=[0, 1])
        rec = metrics.recall_score(target, pred, labels=[0, 1])
        acc = metrics.accuracy_score(target, pred)
        f1 = metrics.f1_score(target, pred, labels=[0, 1])

        logger.log(level=logging.INFO, msg='[Train step] [{}, {}] Loss: {}, Acc: {}, Prec: {}, Recall: {}, F1: {}, AUC_ROC: {}'.format(i, data.size(0), loss.item(), acc, prec, rec, f1, roc))


def valid_step(model, loader, criterion, scheduler=None):
    logger = logging.getLogger('training')

    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            out = model(data)
            loss = criterion(out.squeeze(dim=1), target.cuda())

            prob = out.sigmoid().detach().cpu().numpy().flatten()
            target = target.long().numpy().flatten()
            roc = metrics.roc_auc_score(target, prob, labels=[0, 1])

            pred = (prob > 0.5)
            prec = metrics.precision_score(target, pred, labels=[0, 1])
            rec = metrics.recall_score(target, pred, labels=[0, 1])
            acc = metrics.accuracy_score(target, pred)
            f1 = metrics.f1_score(target, pred, labels=[0, 1])

            if scheduler is not None:
                scheduler.step(roc)

            logger.log(level=logging.INFO, msg='[Valid step] [{}, {}] Loss: {}, Acc: {}, Prec: {}, Recall: {}, F1: {}, AUC_ROC: {}'.format(i, data.size(0), loss.item(), acc, prec, rec, f1, roc))


def main(args):
    logger = logging.getLogger('training')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Define the dataset and loader
    trn_dataset = AngioDB(args.data, mode='train', transform=AngioTransform('train' if args.augment_data else 'valid'), size=args.dataset_size)
    val_dataset = AngioDB(args.data, mode='valid', transform=AngioTransform('valid'), size=-1)

    trn_loader = DataLoader(trn_dataset, batch_size=5, shuffle=True, pin_memory=True, collate_fn=partial(ang_patches_collate, patch_size=32, patch_stride=32, min_labeled_area=50, max_batch_size=args.batch_size))
    val_loader = DataLoader(val_dataset, batch_size=5, pin_memory=True, collate_fn=partial(ang_patches_collate, patch_size=32, patch_stride=32, min_labeled_area=50, max_batch_size=args.batch_size))

    # Define the model
    model = UNet(1, 1, hidden_dim=64, multiplier=2, depth=3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([args.pos_weight]).cuda())

    # Send the model and criterion to GPU
    model, criterion = nn.DataParallel(model).cuda(), criterion.cuda()

    # Define the optimizer and the learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(args.beta0, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, cooldown=2, mode='max')

    for e in range(args.epochs):
        logger.info('Epoch {}'.format(e))
        train_step(model, trn_loader, criterion, optimizer, None)
        valid_step(model, val_loader, criterion, scheduler)

        torch.save(model.state_dict(), os.path.join(args.save, 'checkpoint.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Stenosis classification")
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--dataset_size', type=int, default=100, help='training dataset size')
    parser.add_argument('--augment_data', action='store_true', default=False, help='apply transformation to augment data')
    parser.add_argument('--pos_weight', type=float, default=1, help='positive weight')

    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--beta0', type=float, default=0.9, help='ADAM beta 0 parameter')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')

    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = np.random.randint(0, 10000)

    args.save = '{}_lr{:.0e}_bs{}_{}ds{}_pw{:.0e}'.format(args.save, args.learning_rate, args.batch_size, 'da_' if args.augment_data else '', args.dataset_size, args.pos_weight)
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    logger = logging.getLogger('training')
    fh = logging.FileHandler(os.path.join(args.save, 'experiment.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    # logger.removeHandler(logger.handlers[0])

    main(args)