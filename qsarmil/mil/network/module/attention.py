import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid, Softmax, Tanh
from qsarmil.mil.network.module.base import BaseNetwork, FeatureExtractor
from qsarmil.mil.network.module.utils import SelfAttention


class AttentionNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, input_layer_size, hidden_layer_sizes):

        self.extractor = FeatureExtractor((input_layer_size, *hidden_layer_sizes))
        self.estimator = Linear(hidden_layer_sizes[-1], 1)
        self.attention = Sequential(
            Linear(hidden_layer_sizes[-1], hidden_layer_sizes[-1]),
            Tanh(),
            Linear(hidden_layer_sizes[-1], 1)
        )

        if self.init_cuda:
            self.extractor.cuda()
            self.attention.cuda()
            self.estimator.cuda()

    def forward(self, x, m):

        x = self.extractor(x)
        x_det = torch.transpose(m * self.attention(x), 2, 1)

        w = Softmax(dim=2)(x_det)

        x = torch.bmm(w, x)

        out = self.estimator(x)

        out = self.get_score(out)

        return w, out


class GatedAttentionNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, input_layer_size, hidden_layer_sizes):

        det_ndim = (128,)

        self.main_net = FeatureExtractor((input_layer_size, *hidden_layer_sizes))

        self.attention_V = Sequential(Linear(hidden_layer_sizes[-1], det_ndim[0]), Tanh())

        self.attention_U = Sequential(Linear(hidden_layer_sizes[-1], det_ndim[0]), Sigmoid())

        self.detector = Linear(det_ndim[0], 1)

        self.estimator = Linear(hidden_layer_sizes[-1], 1)

        if self.init_cuda:
            self.main_net.cuda()
            self.attention_V.cuda()
            self.attention_U.cuda()
            self.detector.cuda()
            self.estimator.cuda()

    def forward(self, x, m):

        x = self.main_net(x)

        w_v = self.attention_V(x)

        w_u = self.attention_U(x)

        x_det = torch.transpose(m * self.detector(w_v * w_u), 2, 1)

        w = Softmax(dim=2)(x_det)

        x = torch.bmm(w, x)

        out = self.estimator(x)

        out = self.get_score(out)

        return w, out


class SelfAttentionNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, input_layer_size, hidden_layer_sizes):

        det_ndim = (128,)
        self.main_net = FeatureExtractor((input_layer_size, *hidden_layer_sizes))
        self.attention = SelfAttention(hidden_layer_sizes[-1], hidden_layer_sizes[-1])
        self.detector = Sequential(Linear(hidden_layer_sizes[-1], det_ndim[0]), Tanh(), Linear(det_ndim[0], 1))
        self.estimator = Linear(hidden_layer_sizes[-1], 1)

        if self.init_cuda:
            self.main_net.cuda()
            self.attention.cuda()
            self.detector.cuda()
            self.estimator.cuda()

    def forward(self, x, m):
        x = self.main_net(x)
        x = self.attention(x)

        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = Softmax(dim=2)(x_det)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        out = self.get_score(out)
        return w, out


class TemperatureAttentionNetwork(AttentionNetwork):

    def __init__(self, tau=1, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def forward(self, x, m):

        x = self.extractor(x)

        x_det = torch.transpose(m * self.attention(x), 2, 1)

        w = Softmax(dim=2)(x_det / self.tau)

        x = torch.bmm(w, x)

        out = self.estimator(x)

        out = self.get_score(out)

        return w, out



