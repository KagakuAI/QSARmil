import numpy as np
from sklearn.base import BaseEstimator


def probs_to_class(probs):
    """
    Convert probability predictions to class labels.

    Handles different shapes of probability arrays:
    - 1D array: threshold at 0.5
    - 2D array with 1 or 2 columns: threshold or argmax
    - Multi-class: argmax over classes
    """
    if probs.ndim == 1:
        return (probs > 0.5).astype(int)
    elif probs.shape[1] == 1:
        return (probs[:, 0] > 0.5).astype(int)
    elif probs.shape[1] == 2:
        return (probs[:, 1] > 0.5).astype(int)
    else:
        return np.argmax(probs, axis=1)


class BagWrapper(BaseEstimator):
    """
    Wrapper for bag-level multi-instance learning.

    Aggregates instance features within each bag using a pooling strategy,
    then fits/predicts using a standard sklearn estimator on the pooled features.
    """

    VALID_POOLS = {'mean', 'max', 'min', 'extreme'}

    def __init__(self, estimator, pool='mean'):
        if not hasattr(estimator, "fit") or not (hasattr(estimator, "predict") or hasattr(estimator, "predict_proba")):
            raise ValueError("Estimator must have a 'fit' and 'predict' or 'predict_proba' method.")
        if not (pool in self.VALID_POOLS or callable(pool)):
            raise ValueError(f"Pooling strategy '{pool}' is not supported.")
        self.estimator = estimator
        self.pool = pool
        self.is_classifier = None  # Set during fit()

    def __repr__(self):
        pool_name = self.pool.__name__ if callable(self.pool) else self.pool.title()
        return f'{self.__class__.__name__}|{self.estimator.__class__.__name__}|{pool_name}Pooling'

    def apply_pool(self, bags):
        if not isinstance(bags, (list, np.ndarray)):
            raise ValueError("Input 'bags' should be a list or ndarray of arrays.")

        if callable(self.pool):
            return np.asarray([self.pool(bag) for bag in bags])

        if self.pool == 'mean':
            return np.asarray([np.mean(bag, axis=0) for bag in bags])
        elif self.pool == 'max':
            return np.asarray([np.max(bag, axis=0) for bag in bags])
        elif self.pool == 'min':
            return np.asarray([np.min(bag, axis=0) for bag in bags])
        elif self.pool == 'extreme':
            bags_max = np.asarray([np.max(bag, axis=0) for bag in bags])
            bags_min = np.asarray([np.min(bag, axis=0) for bag in bags])
            return np.concatenate((bags_max, bags_min), axis=1)

        raise RuntimeError("Unknown pooling strategy.")

    def fit(self, bags, labels):
        self.is_classifier = hasattr(self.estimator, 'predict_proba')
        bags_transformed = self.apply_pool(bags)
        self.estimator.fit(bags_transformed, labels)
        return self

    def predict_proba(self, bags):
        if not self.is_classifier:
            raise NotImplementedError("predict_proba is only available for classifiers.")
        bags_transformed = self.apply_pool(bags)
        return self.estimator.predict_proba(bags_transformed)

    def predict(self, bags):
        if self.is_classifier:
            bag_probs = self.predict_proba(bags)
            return probs_to_class(bag_probs)
        else:
            bags_transformed = self.apply_pool(bags)
            return self.estimator.predict(bags_transformed)


class InstanceWrapper(BaseEstimator):
    """
    Wrapper for instance-level multi-instance learning.

    Flattens all instances from all bags to train/predict an instance-level estimator,
    then aggregates instance-level predictions into bag-level predictions using pooling.
    """

    VALID_POOLS = {'mean', 'max', 'min'}

    def __init__(self, estimator, pool='mean'):
        if not hasattr(estimator, "fit") or not (hasattr(estimator, "predict") or hasattr(estimator, "predict_proba")):
            raise ValueError("Estimator must have a 'fit' and 'predict' or 'predict_proba' method.")
        if not (pool in self.VALID_POOLS or callable(pool)):
            raise ValueError(f"Pooling strategy '{pool}' is not supported.")
        self.estimator = estimator
        self.pool = pool
        self.is_classifier = None  # Set during fit()

    def __repr__(self):
        pool_name = self.pool.__name__ if callable(self.pool) else self.pool.title()
        return f'{self.__class__.__name__}|{self.estimator.__class__.__name__}|{pool_name}Pooling'

    def apply_pool(self, instance_preds):
        if callable(self.pool):
            return self.pool(instance_preds)
        elif self.pool == 'mean':
            return np.mean(instance_preds, axis=0)
        elif self.pool == 'max':
            return np.max(instance_preds, axis=0)
        elif self.pool == 'min':
            return np.min(instance_preds, axis=0)
        else:
            raise ValueError(f"Pooling strategy '{self.pool}' is not recognized.")

    def fit(self, bags, labels):
        self.is_classifier = hasattr(self.estimator, 'predict_proba')
        bags = np.asarray(bags, dtype=object)
        bags_transformed = np.vstack(bags)
        labels_transformed = np.hstack([np.full(len(bag), lb) for bag, lb in zip(bags, labels)])
        self.estimator.fit(bags_transformed, labels_transformed)
        return self

    def predict_proba(self, bags):
        if not self.is_classifier:
            raise NotImplementedError("predict_proba is only available for classifiers.")
        bag_preds = []
        for bag in bags:
            bag = bag.reshape(-1, bag.shape[-1])
            instance_preds = self.estimator.predict_proba(bag)
            bag_pred = self.apply_pool(instance_preds)
            bag_preds.append(bag_pred)
        return np.asarray(bag_preds)

    def predict(self, bags):
        if self.is_classifier:
            bag_probs = self.predict_proba(bags)
            return probs_to_class(bag_probs)
        else:
            bag_preds = []
            for bag in bags:
                bag = bag.reshape(-1, bag.shape[-1])
                instance_preds = self.estimator.predict(bag)
                bag_pred = self.apply_pool(instance_preds)
                bag_preds.append(bag_pred)
            return np.asarray(bag_preds)
