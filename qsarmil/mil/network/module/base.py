import torch
import numpy as np
from torch import nn
import torch_optimizer as optim
from torch.nn import Sigmoid, Linear, ReLU, Sequential
from sklearn.model_selection import train_test_split
from qsarmil.mil.network.module.utils import add_padding, get_mini_batches, set_seed

set_seed(42)  # TODO change later

class BaseClassifier:
    def loss(self, y_pred, y_true):
        total_loss = nn.BCELoss(reduction='mean')(y_pred, y_true.reshape(-1, 1))
        return total_loss

    def get_pred(self, out):
        out = Sigmoid()(out)
        out = out.view(-1, 1)
        return out


class BaseRegressor:
    def loss(self, y_pred, y_true):
        total_loss = nn.MSELoss(reduction='mean')(y_pred, y_true.reshape(-1, 1))
        return total_loss

    def get_pred(self, out):
        out = out.view(-1, 1)
        return out


class FeatureExtractor:
    def __new__(cls, hidden_layer_sizes):
        inp_dim = hidden_layer_sizes[0]
        net = []
        for dim in hidden_layer_sizes[1:]:
            net.append(Linear(inp_dim, dim))
            net.append(ReLU())
            inp_dim = dim
        net = Sequential(*net)
        return net


class BaseNetwork(nn.Module):
    def __init__(self,
                 hidden_layer_sizes=(256, 128, 64),
                 num_epoch=500,
                 batch_size=128,
                 learning_rate=0.001,
                 weight_decay=0.001,
                 instance_weight_dropout=0,
                 verbose=False,
                 init_cuda=False):

        super().__init__()

        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.instance_weight_dropout = instance_weight_dropout
        self.batch_size = batch_size
        self.init_cuda = init_cuda
        self.verbose = verbose

    def _initialize(self, input_layer_size, hidden_layer_sizes):
        pass

    def _train_val_split(self, x, y, val_size=0.2, random_state=42):
        x, y = np.asarray(x, dtype="object"), np.asarray(y, dtype="object")
        x, m = add_padding(x)

        x_train, x_val, y_train, y_val, m_train, m_val = train_test_split(x, y, m, test_size=val_size,
                                                                          random_state=random_state)
        x_train, y_train, m_train = self._array_to_tensor(x_train, y_train, m_train)
        x_val, y_val, m_val = self._array_to_tensor(x_val, y_val, m_val)

        return x_train, x_val, y_train, y_val, m_train, m_val

    def _array_to_tensor(self, x, y, m):

        x = torch.from_numpy(x.astype('float32'))
        y = torch.from_numpy(y.astype('float32'))
        m = torch.from_numpy(m.astype('float32'))
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if self.init_cuda:
            x, y, m = x.cuda(), y.cuda(), m.cuda()
        return x, y, m

    def _loss_batch(self, x_mb, y_mb, m_mb, optimizer=None):
        w_out, y_out = self.forward(x_mb, m_mb)
        total_loss = self.loss(y_out, y_mb)
        if optimizer is not None:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return total_loss.item()

    def _reset_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # Add other layer types and initializations as needed
        self.apply(_init_weights)

    def fit(self, x, y):

        self._reset_weights()

        input_layer_size = x[0].shape[-1] # TODO make consistent: x.shape[-1]
        self._initialize(input_layer_size=input_layer_size,
                         hidden_layer_sizes=self.hidden_layer_sizes)

        x_train, x_val, y_train, y_val, m_train, m_val = self._train_val_split(x, y)
        optimizer = optim.Yogi(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        val_loss = []
        for epoch in range(self.num_epoch):

            mb = get_mini_batches(x_train, y_train, m_train, batch_size=self.batch_size)

            self.train()
            for x_mb, y_mb, m_mb in mb:
                loss = self._loss_batch(x_mb, y_mb, m_mb, optimizer=optimizer)

            self.eval()
            with torch.no_grad():
                loss = self._loss_batch(x_val, y_val, m_val, optimizer=None)
                val_loss.append(loss)

            min_loss_idx = val_loss.index(min(val_loss))
            if min_loss_idx == epoch:
                best_parameters = self.state_dict()
                if self.verbose:
                    print(epoch, loss)

        self.load_state_dict(best_parameters, strict=True)

        return self

    def predict(self, x):
        x, m = add_padding(np.asarray(x, dtype="object"))
        x = torch.from_numpy(x.astype('float32'))
        m = torch.from_numpy(m.astype('float32'))

        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                x, m = x.cuda(), m.cuda()
            w, y_pred = self.forward(x, m)
        return np.asarray(y_pred.cpu()).flatten()

    def get_instance_weights(self, x):
        x, m = add_padding(np.asarray(x, dtype="object"))
        x = torch.from_numpy(x.astype('float32'))
        m = torch.from_numpy(m.astype('float32'))

        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                x, m = x.cuda(), m.cuda()
            w, y_pred = self.forward(x, m)

        w = w.view(w.shape[0], w.shape[-1]).cpu()
        m = m.cpu()
        w = [np.asarray(i[j.bool().flatten()]) for i, j in zip(w, m)]
        return w



