import torch
import MinkowskiEngine as ME
from math import log10
import color_space
import numpy as np
from data_utils import isin, istopk
from functional import bound
criterion = torch.nn.BCEWithLogitsLoss()
mse_criterion = torch.nn.MSELoss()
mse = torch.nn.MSELoss()


def rgb2yuv(data, groud_truth):
    device = data.device
    data_ = data.permute(1, 0)
    groud_truth_ = groud_truth.permute(1, 0)
    A = torch.tensor([[0.299, 0.587, 0.114],
                      [-0.169, -0.331, 0.5],
                      [0.5, -0.419, -0.081]]).to(device)
    B = torch.tensor([[0], [128], [128]]).to(device)
    data = torch.add(torch.matmul(A, data_), B).permute(1, 0)
    groud_truth = torch.add(torch.matmul(A, groud_truth_), B).permute(1, 0)
    data = bound(data, 0.0, 255.0)
    groud_truth = bound(groud_truth, 0.0, 255.0)
    return data, groud_truth

def get_mse(data, groud_truth):
    sum_mse = mse(data, groud_truth)
    return sum_mse



class get_mse_g(torch.nn.Module):
    def __init__(self):
        super(get_mse_g, self).__init__()
        self.SumPooling = ME.MinkowskiSumPooling(kernel_size=1, stride=1, dilation=1, dimension=3)

    def forward(self, input, out):
        input = self.SumPooling(input, out.C)
        sum_mse = torch.nn.functional.mse_loss(input.F, out.F)

        return sum_mse


def get_bce_mse(data,ground_truth):
    mask = isin(data.C, ground_truth.C)
    mse = mse_criterion(data.F, (mask.type(data.F.dtype).unsqueeze(1).expand(data.F.shape[0], data.F.shape[1])))
    return mse

def get_bce(data, groud_truth):
    """ Input data and ground_truth are sparse tensor.
    """
    mask = isin(data.C, groud_truth.C)

    bce = criterion(data.F, (mask.type(data.F.dtype).unsqueeze(1).expand(data.F.shape[0], data.F.shape[1])))

    bce /= torch.log(torch.tensor(2.0)).to(bce.device)

    return bce

