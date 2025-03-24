import numpy as np


class BagWrapper:

    def __init__(self, estimator, pool='mean'):
        self.estimator = estimator
        self.pool = pool

    def __repr__(self):
        return f'{self.__class__.__name__}|' \
               f'{self.estimator.__class__.__name__}|' \
               f'{self.pool.title()}Pooling'

    def apply_pool(self, bags):
        if self.pool == 'mean':
            bags_modified = np.asarray([np.mean(bag, axis=0) for bag in bags])
        elif self.pool == 'extreme':
            bags_max = np.asarray([np.amax(bag, axis=0) for bag in bags])
            bags_min = np.asarray([np.amin(bag, axis=0) for bag in bags])
            bags_modified = np.concatenate((bags_max, bags_min), axis=1)
        elif self.pool == 'max':
            bags_modified = np.asarray([np.amax(bag, axis=0) for bag in bags])
        elif self.pool == 'min':
            bags_modified = np.asarray([np.amin(bag, axis=0) for bag in bags])
        return bags_modified

    def fit(self, bags, labels):
        bags_modified = self.apply_pool(bags)
        self.estimator.fit(bags_modified, labels)
        return self.estimator

    def predict(self, bags):
        bags_modified = self.apply_pool(bags)
        preds = self.estimator.predict(bags_modified)
        return preds

    def predict_proba(self, bags):
        bags_modified = self.apply_pool(bags)
        preds = self.estimator.predict_proba(bags_modified)
        return preds


class InstanceWrapper:

    def __init__(self, estimator, pool='mean'):
        self.estimator = estimator
        self.pool = pool

    def __repr__(self):
        return f'{self.__class__.__name__}|' \
               f'{self.estimator.__class__.__name__}|' \
               f'{self.pool.title()}Pooling'

    def apply_pool(self, preds):
        if self.pool == 'mean':
            return np.mean(preds)
        elif self.pool == 'max':
            return np.max(preds)
        elif self.pool == 'min':
            return np.min(preds)
        else:
            print(f'Poolling {self.pool} is not known')
        return preds

    def fit(self, bags, labels):
        bags = np.asarray(bags, dtype="object")
        bags_modified = np.vstack(bags)
        labels_modified = np.hstack([float(lb) * np.array(np.ones(len(bag))) for bag, lb in zip(bags, labels)])
        self.estimator.fit(bags_modified, labels_modified)
        return self.estimator
