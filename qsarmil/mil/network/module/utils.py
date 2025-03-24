import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset


def add_padding(x):
    bag_size = max(len(i) for i in x)
    mask = np.ones((len(x), bag_size, 1))

    out = []
    for i, bag in enumerate(x):
        bag = np.asarray(bag)
        if len(bag) < bag_size:
            mask[i][len(bag):] = 0
            padding = np.zeros((bag_size - bag.shape[0], bag.shape[1]))
            bag = np.vstack((bag, padding))
        out.append(bag)
    out_bags = np.asarray(out)
    return out_bags, mask


def get_mini_batches(x, y, m, batch_size=16):
    data = MBSplitter(x, y, m)
    mb = DataLoader(data, batch_size=batch_size, shuffle=True)
    return mb


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class MBSplitter(Dataset):
    def __init__(self, x, y, m):
        super(MBSplitter, self).__init__()
        self.x = x
        self.y = y
        self.m = m

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.m[i]

    def __len__(self):
        return len(self.y)


class SelfAttention(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()

        self.w_query = nn.Linear(inp_dim, out_dim)
        self.w_key = nn.Linear(inp_dim, out_dim)
        self.w_value = nn.Linear(inp_dim, out_dim)

    def forward(self, x):
        keys = self.w_key(x)
        querys = self.w_query(x)
        values = self.w_value(x)

        att_weights = softmax(querys @ torch.transpose(keys, 2, 1), dim=-1)
        weighted_values = values[:, :, None] * torch.transpose(att_weights, 2, 1)[:, :, :, None]
        outputs = weighted_values.sum(dim=1)

        return outputs
