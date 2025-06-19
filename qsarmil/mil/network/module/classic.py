from torch import nn
from torch.nn import Sequential, Linear, Softmax, Sigmoid
from qsarmil.mil.network.module.base import BaseNetwork, BaseClassifier, FeatureExtractor


class BagNetwork(BaseNetwork):
    def __init__(self, pool='mean', **kwargs):
        super().__init__(**kwargs)
        self.pool = pool

    def _initialize(self, input_layer_size, hidden_layer_sizes):
        self.extractor = FeatureExtractor((input_layer_size, *hidden_layer_sizes))
        self.pooling = Pooling(self.pool)
        self.estimator = Linear(hidden_layer_sizes[-1], 1)

        if self.init_cuda:
            self.extractor.cuda()
            self.estimator.cuda()

    def forward(self, x, m):
        out = self.extractor(x)
        out = self.pooling(out, m)
        out = self.estimator(out)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        return None, out


class InstanceNetwork(BaseNetwork):

    def __init__(self, pool='mean', **kwargs):
        super().__init__(**kwargs)
        self.pool = pool

    def _initialize(self, input_layer_size, hidden_layer_sizes):
        self.extractor = Sequential(FeatureExtractor((input_layer_size, *hidden_layer_sizes)), Linear(hidden_layer_sizes[-1], 1))
        self.pooling = Pooling(self.pool)

        if self.init_cuda:
            self.extractor.cuda()

    def forward(self, x, m):
        out = self.extractor(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = self.pooling(out, m)
        return None, out


class Pooling(nn.Module):
    def __init__(self, pool='mean'):
        super().__init__()
        self.pool = pool

    def forward(self, x, m):
        x = m * x
        if self.pool == 'mean':
            out = x.sum(axis=1) / m.sum(axis=1)
        elif self.pool == 'max':
            out = x.max(dim=1)[0]
        elif self.pool == 'lse':
            out = x.exp().sum(dim=1).log()
        return out

    def extra_repr(self):
        return 'Pooling(out_dim=1)'
