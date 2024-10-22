import torch
from torch.nn import Sequential, Linear, Sigmoid, ReLU
from miprop.mil.networks.modules.base import BaseNetwork, BaseClassifier, Pooling
from miprop.mil.networks.modules.base import MainNetwork


class GaussianPoolingNetwork(BaseNetwork):
    def __init__(self, pool='lse', **kwargs):
        super().__init__(**kwargs)
        self.pool = pool

    def _initialize(self, input_layer_size, hidden_layer_sizes, init_cuda):

        det_ndim = (128,)
        self.main_net = MainNetwork((input_layer_size, *hidden_layer_sizes))
        self.detector = Sequential(Linear(hidden_layer_sizes[-1], det_ndim[0]), ReLU(), Linear(det_ndim[0], 1))
        self.estimator = Linear(hidden_layer_sizes[-1], 1)
        self.m = torch.nn.Parameter(torch.Tensor([0.]))
        self.s = torch.nn.Parameter(torch.Tensor([1.]))

        if init_cuda:
            self.main_net.cuda()
            self.detector.cuda()
            self.estimator.cuda()

    def forward(self, x, m):
        x = self.main_net(x)
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
