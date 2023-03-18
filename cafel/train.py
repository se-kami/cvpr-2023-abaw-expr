#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train the model from scratch.

Example:
    python train.py
    python train.py --runs-directory runs
"""

import os
from argparse import ArgumentParser
from time import time

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import data
from loss import AnchorLoss, CenterLoss, DistLoss
from metrics import Meter
from models.anchors import Anchors
from models.head import ClassificationHead
from models.poster import get_poster
from utils import normalized_entropy


torch.autograd.set_detect_anomaly(True)

def train(config):
    # training config
    val_every = config['val_every']
    num_iters = config['num_iters']
    test_set_correction = config['test_set_correction']
    # data
    data_maker = config['data_maker']
    loader_dev = config['loader_dev']
    batch_size = config['batch_size']
    # model
    device = config['device']
    vit = config['vit']
    classifier = config['classifier']
    anchors = config['anchors']
    vit = vit.to(device)
    classifier = classifier.to(device)
    anchors = anchors.to(device)
    # loss
    loss_fn_mu = config['loss_fn_mu']
    loss_fn_center = config['loss_fn_center']

    lambda_ce = config['lambda_ce']
    lambda_mu = config['lambda_mu']
    lambda_center = config['lambda_center']
    temperature = config['temperature']
    delta = config['delta']

    optimizer = config['optimizer']([
        {'params': vit.parameters(), 'lr': config['lr_vit']},
        {'params': classifier.parameters(), 'lr': config['lr_classifier']},
        {'params': anchors.parameters(), 'lr': config['lr_anchors']},
        ])

    # functions
    confidence_fn = config['confidence_fn']

    # logs
    dir_log = config['dir_run']
    log_file = config['log_file']
    csv_file = config['csv_file']
    writer = config['writer']
    meter_train = Meter()
    meter_train_no_corr = Meter()

    # best metrics
    metric_best = 0
    acc_best = 0

    loader = data_maker.get_dataloader(batch_size)
    iter_loader = iter(loader)
    weights = loader.dataset.get_weights().to(device).float()
    loss_fn_ce = config['loss_fn_ce'](weight=weights)
    bar = tqdm(range(num_iters))
    for n_iter in bar:
        # get data
        try:
            x, y = next(iter_loader)
        except StopIteration:
            loader = data_maker.get_dataloader(batch_size)
            weights = loader.dataset.get_weights().to(device).float()
            loss_fn_ce = config['loss_fn_ce'](weight=weights)
            iter_loader = iter(loader)
            x, y = next(iter_loader)

        # TRAINING
        # models to train mode
        vit.train()
        classifier.train()
        anchors.train()

        # forward pass
        x = x.to(device)
        y = y.to(device)
        embeddings = vit(x)  # [batch, emb]
        logits = classifier(embeddings)  # [batch, n_classes]
        prob_dist = torch.softmax(logits / temperature, -1)  # label prob distribution
        # label correction
        confidence = confidence_fn(prob_dist).view(-1, 1)  # [batch, 1]
        distances = anchors(embeddings)  # [batch, n_classes, n_anchors]
        similarities = torch.softmax(-distances.view(distances.shape[0], -1) / delta, -1)  # [batch, n_classes * n_anchors]
        similarities = similarities.view_as(distances)  # [batch, n_classes, n_anchors]
        similarities = similarities.sum(-1)  # [batch, n_classes]
        dist_correction = similarities * confidence  # [batch, n_classes]
        trust = n_iter / num_iters
        dist_correction = confidence * prob_dist + (1 - confidence) * dist_correction
        prob_dist_corrected = (1 - trust) * prob_dist + trust * dist_correction
        # loss
        loss_ce = loss_fn_ce(prob_dist_corrected, y)  # cross entropy with temperature
        loss_mu = loss_fn_mu(anchors.get_anchors())  # keep anchors apart
        loss_center = loss_fn_center(distances, y, confidence)  # keep embeddings in the right cluster
        loss = lambda_ce * loss_ce + lambda_mu * loss_mu * trust + lambda_center * loss_center * trust  # total loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        # to meter
        meter_train.add(torch.argmax(prob_dist_corrected, 1), y)
        meter_train_no_corr.add(torch.argmax(prob_dist, 1), y)  # acc without correction
        meter_train.log('loss_ce', loss_ce.item())
        meter_train.log('loss_mu', loss_mu.item())
        meter_train.log('loss_center', loss_center.item())

        # to cli
        msg = f"[TRAIN]"
        msg += f"[ITER: {n_iter:04d}]"
        msg += meter_train.get_message(loss=True)

        # to file
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
        print(msg)

        # to tensorboard
        metrics_ema = meter_train.get_metrics_ema()
        metrics_iter = meter_train.get_metrics_last()
        log_ema = meter_train.get_log_ema()
        log_iter = meter_train.get_log_last()

        writer.add_scalar('Accuracy/Train_iter', metrics_iter['accuracy'], n_iter)
        writer.add_scalar('Accuracy/Train_ema', metrics_ema['accuracy'], n_iter)
        writer.add_scalar('F1/Train_iter', metrics_iter['f1'], n_iter)
        writer.add_scalar('F1/Train_ema', metrics_ema['f1'], n_iter)

        writer.add_scalar('Loss/CE_iter', log_iter['loss_ce'], n_iter)
        writer.add_scalar('Loss/CE_ema', log_ema['loss_ce'], n_iter)
        writer.add_scalar('Loss/Mu_iter', log_iter['loss_mu'], n_iter)
        writer.add_scalar('Loss/Mu_ema', log_ema['loss_mu'], n_iter)
        writer.add_scalar('Loss/Center_iter', log_iter['loss_center'], n_iter)
        writer.add_scalar('Loss/Center_ema', log_ema['loss_center'], n_iter)

        # EVAL
        if n_iter % val_every == 0:
            # models to eval
            vit.eval()
            classifier.eval()
            anchors.eval()
            # logging
            meter_dev = Meter()
            meter_dev_no_corr = Meter()
            with torch.no_grad():
                bar_dev = tqdm(loader_dev)
                for x, y in bar_dev:
                    # forward pass
                    x = x.to(device)
                    y = y.to(device)
                    embeddings = vit(x)  # [batch, emb]
                    logits = classifier(embeddings)  # [batch, n_classes]
                    prob_dist = torch.softmax(logits / temperature, -1)  # label prob distribution
                    # label correction
                    if test_set_correction:
                        confidence = confidence_fn(prob_dist).view(-1, 1)  # [batch, 1]
                        distances = anchors(embeddings)  # [batch, n_classes, n_anchors]
                        similarities = torch.softmax(-distances.view(distances.shape[0], -1) / delta)  # [batch, n_classes * n_anchors]
                        similarities = similarities.view_as(distances)  # [batch, n_classes, n_anchors]
                        similarities = similarities.sum(-1)  # [batch, n_classes]
                        dist_correction = similarities * confidence  # [batch, n_classes]
                        trust = n_iter / num_iters
                        dist_correction = confidence * prob_dist + (1 - confidence) * dist_correction
                        prob_dist_corrected = (1 - trust) * prob_dist + trust * dist_correction
                    else:
                        prob_dist_corrected = prob_dist

                    # logging
                    # to meter
                    meter_dev.add(torch.argmax(prob_dist_corrected, 1), y)
                    meter_dev_no_corr.add(torch.argmax(prob_dist, 1), y)

            # to cli
            msg = f"[VALID]"
            msg += f"[ITER: {n_iter:04d}]"
            msg += meter_dev.get_message(loss=False)

            # to tensorboard
            metrics_total = meter_dev.get_metrics_total()

            writer.add_scalar('Accuracy/Dev', metrics_total['accuracy'], n_iter)
            writer.add_scalar('F1/Dev', metrics_total['f1'], n_iter)

            # check for best
            if metrics_total['f1'] > metric_best:
                metric_best = metrics_total['f1']
                acc_best = metrics_total['accuracy']
                # save best models
                torch.save(vit.state_dict(),
                           os.path.join(dir_log, 'vit.best.{epoch:04d}.pth'))
                torch.save(classifier.state_dict(),
                           os.path.join(dir_log, 'classifier.best.{epoch:04d}.pth'))
                torch.save(anchors.state_dict(),
                           os.path.join(dir_log, 'anchors.best.{epoch:04d}.pth'))

            msg += f"[ACC_BEST: {acc_best:.1f}]"
            msg += f"[F1_BEST: {metric_best:.1f}]"
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
            print(msg)


    # save logs
    df = {}
    df.update(meter_train.get_log_all())
    df.update(meter_train.get_metrics_all())
    df = pd.DataFrame(df)
    df.to_csv(csv_file)


def main():
    description = __doc__.split("\n")[0]
    root = '/'.join(__file__.split('/')[:-1])  # location of this file
    parser = ArgumentParser(description=description)

    parser.add_argument(
            '--runs-directory',
            type=str,
            default='runs',
            help="root directory for experiment logs"
            )
    parser.add_argument(
            '--experiment-name',
            type=str,
            default=None,
            help='directory name for current run'
            )
    parser.add_argument(
            '--tensorboard-directory',
            type=str,
            default='runs/board',
            help='root directory for tensorboard logs'
            )
    parser.add_argument(
            '--path-mobilefacenet',
            type=str,
            default='models/pretrained/mobilefacenet.pth',
            help='location of state dict for pretrained mobilefacenet'
            )
    parser.add_argument(
            '--path-ir50',
            type=str,
            default='models/pretrained/ir50.pth',
            help='location of state dict for pretrained ir50'
            )
    parser.add_argument(
            '--size-img',
            type=str,
            default=224,
            help='input image size'
            )
    parser.add_argument(
            '--dim-emb',
            type=str,
            default=768,
            help='dimension of embedding'
            )
    parser.add_argument(
            '--num-anchors',
            type=str,
            default=1,
            help='number of anchors per class'
            )
    parser.add_argument(
            '--batch-size',
            type=str,
            default=96,
            help='batch size'
            )
    parser.add_argument(
            '--num-classes',
            type=int,
            default=8,
            help='number of classes in dataset'
            )
    parser.add_argument(
            '--val-every',
            type=int,
            default=32,
            help='run testing on dev set every every <val-every> iters'
            )
    parser.add_argument(
            '--hidden-sizes',
            type=str,
            default='384,192,96',
            help='sizes of layers in MLP classification head, separate with comma',
            )
    parser.add_argument(
            '--hyperparam-delta',
            type=float,
            default=1.0,
            help='value of hyperparam delta',
            )
    parser.add_argument(
            '--hyperparam-temp',
            type=float,
            default=1.0,
            help='value of hyperparam temperature',
            )
    parser.add_argument(
            '--hyperparam-lr-vit',
            type=float,
            default=3e-6,
            help='learning rate for ViT',
            )
    parser.add_argument(
            '--hyperparam-lr-head',
            type=float,
            default=3e-6,
            help='learning rate for classification head',
            )
    parser.add_argument(
            '--hyperparam-lr-anchor',
            type=float,
            default=3e-7,
            help='learning rate for anchors',
            )
    parser.add_argument(
            '--hyperparam-lambda-ce',
            type=float,
            default=1.0,
            help='CE loss multiplier',
            )
    parser.add_argument(
            '--hyperparam-lambda-mu',
            type=float,
            default=0.001,
            help='Anchor Center Loss loss multiplier',
            )
    parser.add_argument(
            '--hyperparam-lambda-center',
            type=float,
            default=0.001,
            help='Anchor Embedding Loss loss multiplier',
            )
    parser.add_argument(
            '--num-iters',
            type=int,
            default=10000,
            help='number of training iterations',
            )
    parser.add_argument(
            '--test-set-correction',
            type=bool,
            default=False,
            help='use label correction on test set',
            )

    args = parser.parse_args()

    # PREPARE LOG DIRECTORIES
    # root expriment directory
    dir_runs = args.runs_directory
    if not os.path.exists(dir_runs):
        os.mkdir(dir_runs)
    # current experiment directory
    dir_run = args.experiment_name
    if dir_run is None:  # name it with timestamp
        dir_run = str(int(time()))
    experiment_name = dir_run
    dir_run = os.path.join(dir_runs, dir_run)
    if not os.path.exists(dir_run):
        os.mkdir(dir_run)
    # tensorboard directory
    dir_board = args.tensorboard_directory
    if not os.path.exists(dir_board):
        os.mkdir(dir_board)
    dir_board = os.path.join(dir_board, experiment_name)

    # training config
    config = dict()
    config['dir_run'] = dir_run
    config['log_file'] = os.path.join(dir_run, 'log.txt')
    config['csv_file'] = os.path.join(dir_run, 'log.csv')
    config['writer'] = SummaryWriter(log_dir=dir_board)
    config['val_every'] = args.val_every
    config['test_set_correction'] = args.test_set_correction
    config['batch_size'] = args.batch_size
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['vit'] = get_poster(path_landmark=args.path_mobilefacenet,
                               path_ir=args.path_ir50,
                               dim_emb=args.dim_emb,)
    config['classifier'] = ClassificationHead(
            size_in=args.dim_emb,
            size_out=args.num_classes,
            size_hidden=[int(i) for i in args.hidden_sizes.split(',')],)
    config['anchors'] = Anchors(size_emb=args.dim_emb,
                                n_classes=args.num_classes,
                                n_anchors=args.num_anchors,
                                )
    config['confidence_fn'] = lambda x: 1 - normalized_entropy(x)
    config['loss_fn_ce'] = DistLoss
    config['loss_fn_mu'] = AnchorLoss()
    config['loss_fn_center'] = CenterLoss()
    config['lambda_ce'] = args.hyperparam_lambda_ce
    config['lambda_mu'] = args.hyperparam_lambda_mu
    config['lambda_center'] = args.hyperparam_lambda_center
    config['temperature'] = args.hyperparam_temp
    config['delta'] = args.hyperparam_temp
    config['lr_vit'] = args.hyperparam_lr_vit
    config['lr_classifier'] = args.hyperparam_lr_head
    config['lr_anchors'] = args.hyperparam_lr_anchor
    config['optimizer'] = torch.optim.Adam

    config['num_iters'] = args.num_iters
    config['loader_train'] = data.get_dataloader_train(
            batch_size=args.batch_size, image_size=args.size_img,)
    config['loader_dev'] = data.get_dataloader_dev(
            batch_size=args.batch_size * 2, image_size=args.size_img,)
    config['data_maker'] = data.get_datamaker()


    train(config)


if __name__ == '__main__':
    main()
