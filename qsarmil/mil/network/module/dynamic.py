import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Softmax
from qsarmil.mil.network.module.base import BaseNetwork, BaseRegressor, FeatureExtractor


class MarginLoss(nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, alpha=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.alpha = alpha

    def forward(self, lengths, labels):
        left = F.relu(self.m_pos - lengths, inplace=True) ** 2
        right = F.relu(lengths - self.m_neg, inplace=True) ** 2
        margin_loss = labels * left + self.alpha * (1. - labels) * right
        return margin_loss.mean()


class Norm(nn.Module):
    def forward(self, inputs):
        lengths = torch.norm(inputs, p=2, dim=1, keepdim=True)
        return lengths


class Squash(nn.Module):
    def forward(self, inputs):
        norm = torch.norm(inputs, p=2, dim=2, keepdim=True)
        scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)
        return scale * inputs


class DynamicPooling(nn.Module):
    def __init__(self, n_iter=3):
        super().__init__()
        self.n_iter = n_iter

    def forward(self, x, m):
        x = m * x
        b = torch.zeros(x.shape[0], x.shape[1], 1).to(x.device)
        for t in range(self.n_iter):
            w = Softmax(dim=1)(m * b)
            w = torch.transpose(w, 2, 1)
            sigma = torch.bmm(w, x)
            s = Squash()(sigma)

            b_new = torch.sum(s * x, dim=2)
            b_new = b_new.reshape(b_new.shape[0], b_new.shape[1], 1)
            b = b + b_new

        w = Softmax(dim=1)(b)

        s = s.view(s.shape[0], s.shape[-1])
        w = w.view(w.shape[0], w.shape[1])
        return w, s


class DynamicPoolingNetwork(BaseNetwork):
    def __init__(self, **kwarhs):
        super().__init__(**kwarhs)

    def _initialize(self, input_layer_size, hidden_layer_sizes):
        self.extractor = FeatureExtractor((input_layer_size, *hidden_layer_sizes))
        self.pooling = DynamicPooling()
        self.estimator = Norm()

    def forward(self, x, m):
        x = self.extractor(x)
        w, s = self.pooling(x, m)
        bag_score = self.estimator(s)
        return w, bag_score

    def predict(self, x):
        y_pred = super().predict(x)
        if isinstance(self, BaseRegressor):
            y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        return y_pred
