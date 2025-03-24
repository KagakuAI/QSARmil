import torch
from torch.nn import Sequential, Linear, Sigmoid, ReLU
from qsarmil.mil.network.module.base import BaseNetwork, BaseClassifier, FeatureExtractor
from qsarmil.mil.network.module.classic import Pooling


class GaussianPoolingNetwork(BaseNetwork):  # TODO does not work when y is big (e.g. 345)
    def __init__(self, pool='lse', **kwargs):
        super().__init__(**kwargs)
        self.pool = pool

    def _initialize(self, input_layer_size, hidden_layer_sizes):

        det_ndim = (128,)
        self.extractor = FeatureExtractor((input_layer_size, *hidden_layer_sizes))
        self.detector = Sequential(Linear(hidden_layer_sizes[-1], det_ndim[0]), ReLU(), Linear(det_ndim[0], 1))
        self.estimator = Linear(hidden_layer_sizes[-1], 1)
        self.m = torch.nn.Parameter(torch.Tensor([0.]))
        self.s = torch.nn.Parameter(torch.Tensor([1.]))

        if self.init_cuda:
            self.extractor.cuda()
            self.detector.cuda()
            self.estimator.cuda()

    def forward(self, x, m):
        x = self.extractor(x)

        out = self.estimator(x)

        x = self.detector(x)

        w = self.gaussian_weighting(x, self.m, self.s)

        out = w * out
        out = self.pooling(out, m)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        w = w.reshape(w.shape[0], w.shape[2], w.shape[1])
        return w, out

    def gaussian_weighting(self, x, m, s):
        m = m.to(x.device)
        s = s.to(x.device)

        z = (x - m) / s
        w = torch.exp(-(z ** 2))
        return w

    def pooling(self, out, m):
        out = Pooling(pool=self.pool)(out, m)
        return out
