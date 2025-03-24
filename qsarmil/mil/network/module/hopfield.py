from torch.nn import Linear, Sigmoid
from qsarmil.mil.network.module.base import BaseNetwork, BaseClassifier, FeatureExtractor


class HopfieldNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, input_layer_size, hidden_layer_sizes, init_cuda):
        self.main_net = FeatureExtractor((input_layer_size, *hidden_layer_sizes))
        self.estimator = Linear(hidden_layer_sizes[-1], 1)
        #
        input_dim = hidden_layer_sizes[-1]
        self.pooling = HopfieldPooling(input_size=input_dim, hidden_size=input_dim, output_size=input_dim, num_heads=4)

        if init_cuda:
            self.main_net.cuda()
            self.pooling.cuda()
            self.estimator.cuda()

    def forward(self, x, m):
        x = self.main_net(x)

        x = self.pooling(x)

        x = x.view(-1, 1, x.shape[-1])

        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        w = None
        return w, out


