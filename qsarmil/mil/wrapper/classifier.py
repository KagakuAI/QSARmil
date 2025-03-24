import numpy as np
from sklearn.svm import SVC
from qsarmil.mil.wrapper.base import BagWrapper, InstanceWrapper


class BagWrapperClassifier(BagWrapper):

    def __init__(self, estimator, pool='mean'):
        super().__init__(estimator=estimator, pool=pool)


class InstanceWrapperClassifier(InstanceWrapper):

    def __init__(self, estimator, pool='mean'):
        super().__init__(estimator=estimator, pool=pool)

    def predict(self, bags):
        preds = self.predict_proba(bags)
        preds = np.where(preds > 0.5, 1, 0)  # TODO temp solution
        return preds

    def predict_proba(self, bags):
        preds = []
        for bag in bags:
            bag = bag.reshape(-1, bag.shape[-1])
            y_pred = self.apply_pool(self.estimator.predict_proba(bag))
            preds.append(y_pred)
        return np.asarray(preds)
