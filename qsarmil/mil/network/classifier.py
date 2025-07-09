from qsarmil.mil.network.module.attention import AttentionNetwork, MultiHeadAttentionNetwork, SelfAttentionNetwork, GatedAttentionNetwork, \
    TempAttentionNetwork, HopfieldAttentionNetwork
from qsarmil.mil.network.module.base import BaseClassifier
from qsarmil.mil.network.module.dynamic import DynamicPoolingNetwork, MarginLoss
from qsarmil.mil.network.module.classic import BagNetwork, InstanceNetwork


class AttentionNetworkClassifier(AttentionNetwork, BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MultiHeadAttentionNetworkClassifier(MultiHeadAttentionNetwork, BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SelfAttentionNetworkClassifier(SelfAttentionNetwork, BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class GatedAttentionNetworkClassifier(GatedAttentionNetwork, BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class TempAttentionNetworkClassifier(TempAttentionNetwork, BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class HopfieldAttentionNetworkClassifier(HopfieldAttentionNetwork, BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BagNetworkClassifier(BagNetwork, BaseClassifier):

    def __init__(self, pool='mean', **kwargs):
        super().__init__(pool=pool, **kwargs)

class InstanceNetworkClassifier(InstanceNetwork, BaseClassifier):
    def __init__(self, pool='mean', **kwargs):
        super().__init__(pool=pool, **kwargs)

class DynamicPoolingNetworkClassifier(DynamicPoolingNetwork, BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def loss(self, y_pred, y_true):
        margin_loss = MarginLoss()
        loss = margin_loss(y_pred, y_true.reshape(-1, 1))
        return loss
