
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame


class BagScaler:
    pass


class BagMinMaxScaler(BagScaler):
    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler()

    def fit(self, x):
        if isinstance(x, DataFrame):
            x = df_to_list_of_bags(x)
        self.scaler.fit(np.vstack(x))

    def transform(self, x):
        if isinstance(x, DataFrame):
            x = df_to_list_of_bags(x)

        x_scaled = x.copy()
        for i, bag in enumerate(x):
            x_scaled[i] = self.scaler.transform(bag)

        return x_scaled

