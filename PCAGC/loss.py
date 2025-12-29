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

def yuv2rgb(data, groud_truth):
    device = data.device
    data_ = data.permute(1, 0)
    groud_truth_ = groud_truth.permute(1, 0)
    A = torch.tensor([[0.299, 0.587, 0.114],
                      [-0.169, -0.331, 0.5],
                      [0.5, -0.419, -0.081]]).to(device)
    A = torch.linalg.inv(A)
    B = torch.tensor([[0], [128], [128]]).to(device)
    data = torch.matmul(A, torch.sub(data_, B)).permute(1, 0)
    groud_truth = torch.matmul(A, torch.sub(groud_truth_, B)).permute(1, 0)
    data = bound(data, 0.0, 255.0)
    groud_truth = bound(groud_truth, 0.0, 255.0)
    return data, groud_truth

def get_mse(data, groud_truth):
    sum_mse = mse(data, groud_truth)
    return sum_mse

def get_mse_F(data, groud_truth):
    sum_mse = mse(data.F, groud_truth.F)
    return sum_mse

class get_mse_g(torch.nn.Module):
    def __init__(self):
        super(get_mse_g, self).__init__()
        self.SumPooling = ME.MinkowskiSumPooling(kernel_size=1, stride=1, dilation=1, dimension=3)

    def forward(self, input, out):
        input = self.SumPooling(input, out.C)
        sum_mse = torch.nn.functional.mse_loss(input.F, out.F)

        return sum_mse

def psnr(imgs1, imgs2):
    mse = torch.nn.functional.mse_loss(imgs1.F, imgs2.F)
    psnr = 10 * log10(255*255/mse)
    return psnr

def yuv_psnr(data, groud_truth):
    data = color_space.rgb_to_yuv(data)
    groud_truth = color_space.rgb_to_yuv(groud_truth)
    mse = np.mean(np.square((groud_truth - data) / 255.), axis=0)
    psnr = -10 * np.log10(mse)
    # mse1 = np.mean(np.square(groud_truth - data), axis=0)
    # psnr1 = 10 * np.log10(255*255/mse1)
    return psnr

def get_bits(likelihood):
    bits = -torch.sum(torch.log2(likelihood))

    return bits


def get_metrics(data, groud_truth):
    mask_real = isin(data.C, groud_truth.C)
    nums = [len(C) for C in groud_truth.decomposed_coordinates]
    mask_pred = istopk(data, nums, rho=1.0)
    metrics = get_cls_metrics(mask_pred, mask_real)

    return metrics[0]


def get_cls_metrics(pred, real):
    TP = (pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FN = (~pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FP = (pred * ~real).cpu().nonzero(as_tuple=False).shape[0]
    TN = (~pred * ~real).cpu().nonzero(as_tuple=False).shape[0]

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    IoU = TP / (TP + FP + FN + 1e-7)

    return [round(precision, 4), round(recall, 4), round(IoU, 4)]

def get_bce_mse(data,ground_truth):
    mask = isin(data.C, ground_truth.C)
    mse = mse_criterion(data.F, (mask.type(data.F.dtype).unsqueeze(1).expand(data.F.shape[0], data.F.shape[1])))
    return mse

def get_bce(data, groud_truth):
    """ Input data and ground_truth are sparse tensor.
    """
    mask = isin(data.C, groud_truth.C)

    # bce = criterion(data.F.squeeze(), mask.type(data.F.dtype))

    bce = criterion(data.F, (mask.type(data.F.dtype).unsqueeze(1).expand(data.F.shape[0], data.F.shape[1])))

    bce /= torch.log(torch.tensor(2.0)).to(bce.device)

    return bce

