import sys
import os
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn import metrics

from models.unet import UNet
from data.angiodb import AngioDB, AngioTransform, ang_patches_collate
from functools import partial


def valid_step(model, loader, criterion):
    logger = logging.getLogger('testing')

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

            logger.log(level=logging.INFO, msg='[Test step] [{}, {}] Loss: {}, Acc: {}, Prec: {}, Recall: {}, F1: {}, AUC_ROC: {}'.format(i, data.size(0), loss.item(), acc, prec, rec, f1, roc))


def main(args):
    logger = logging.getLogger('testing')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Define the dataset and loader
    tst_dataset = AngioDB(args.data, mode='test', transform=AngioTransform('test'), size=-1)
    tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, pin_memory=True)

    # Define the model
    model = UNet(1, 1, hidden_dim=64, multiplier=2, depth=3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([args.pos_weight]).cuda())

    # Send the model and criterion to GPU
    model, criterion = nn.DataParallel(model).cuda(), criterion.cuda()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)

    valid_step(model, tst_loader, criterion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Stenosis classification")
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--pos_weight', type=float, default=1, help='positive weight')

    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')

    parser.add_argument('--model_path', type=str, help='path to the checkpoint model')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = np.random.randint(0, 10000)

    if args.model_path is None:
        args.model_path = os.path.join(args.save, 'checkpoint.pth')

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    logger = logging.getLogger('testing')
    fh = logging.FileHandler(os.path.join(args.save, 'evaluation.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    # logger.removeHandler(logger.handlers[0])

    main(args)