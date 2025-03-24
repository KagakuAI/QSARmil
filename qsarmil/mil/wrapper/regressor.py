import numpy as np

from qsarmil.mil.wrapper.base import InstanceWrapper, BagWrapper


class InstanceWrapperRegressor(InstanceWrapper):

    def __init__(self, estimator, pool='mean'):
        super().__init__(estimator=estimator, pool=pool)

    def predict(self, bags):
        preds = [self.apply_pool(self.estimator.predict(bag.reshape(-1, bag.shape[-1]))) for bag in bags]
        return np.asarray(preds)


class BagWrapperRegressor(BagWrapper):

    def __init__(self, estimator, pool='mean'):
        super().__init__(estimator=estimator, pool=pool)
