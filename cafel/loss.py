#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
from torch import nn

class AnchorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, anchors):
        n_classes, k, dim_emb = anchors.shape
        anchors = anchors.view(n_classes, -1)
        factor = math.sqrt(dim_emb)
        distances = (anchors.unsqueeze(0) - anchors.unsqueeze(1)) ** 2
        distances = distances / factor
        loss = -distances.sum()
        return loss


class CenterLoss(nn.Module):
    def __init__(self, n_classes=8, reduction='mean'):
        super().__init__()
        self.n_classes = n_classes
        self.reduction = reduction

    def forward(self, distances, labels, confidence):
        distances = distances[range(len(labels)), labels]
        distances = torch.min(distances, 1)
        loss = distances * confidence.view(-1)
        if self.reduction == 'mean':
            return loss.sum() / len(loss)
        return loss.sum()


class DistLoss(nn.Module):
    """
    NLL Loss applied to probability distribution.
    """
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight)

    def forward(self, x, l):
        return self.loss(torch.log(x), l)
