import os
import numpy as np
import h5py
import open3d as o3d

import torch.nn as nn
import torch.nn.functional as F


def read_h5_geo(filedir):
    pc = h5py.File(filedir, 'r')['data'][:]
    coords = pc[:, 0:3].astype('int')
    attri = pc[:, 3:6].astype('int')

    return coords, attri


def write_h5_geo(filedir, coords):
    data = coords.astype('uint8')
    with h5py.File(filedir, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)

    return


def read_ply_ascii_geo(filename):
    '''
    load data from ply file.
    '''
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)
    attribute = np.asarray(pcd.colors)
    attribute = np.around(attribute * 254)
    points = points.astype(np.int32)  # np.uint8
    attribute = attribute.astype(np.int32)  # np.uint8

    return points, attribute


def read_ply_ascii_geo_test(filedir):
    files = open(filedir)
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError:
            continue
        data.append(line_values)
    data = np.array(data)
    coords = data[:, 0:3].astype('int')
    return coords


def write_ply_ascii_geo(filedir, coords):
    if os.path.exists(filedir): os.system('rm ' + filedir)
    f = open(filedir, 'a+')
    f.writelines(['ply\n', 'format ascii 1.0\n'])
    f.write('element vertex ' + str(coords.shape[0]) + '\n')
    f.writelines(['property float x\n', 'property float y\n', 'property float z\n'])
    f.write('end_header\n')
    coords = coords.astype('int')
    for p in coords:
        f.writelines([str(p[0]), ' ', str(p[1]), ' ', str(p[2]), '\n'])
    f.close()

    return


def write_ply_ascii_attri(filedir, coords, feature):
    if os.path.exists(filedir): os.system('rm ' + filedir)
    f = open(filedir, 'a+')
    f.writelines(['ply\n', 'format ascii 1.0\n'])
    f.write('element vertex ' + str(coords.shape[0]) + '\n')
    f.writelines(['property float x\n', 'property float y\n', 'property float z\n', 'property uchar red\n',
                  'property uchar green\n', 'property uchar blue\n'])
    f.write('end_header\n')
    coords = coords.astype('int')
    feature = feature.astype('int')
    for p, c in zip(coords, feature):
        f.writelines([str(p[0]), ' ', str(p[1]), ' ', str(p[2]), ' ', str(c[0]), ' ', str(c[1]), ' ', str(c[2]), '\n'])
    f.close()

    return


###########################################################################################################

import torch
import MinkowskiEngine as ME


def sort_spare_tensor2(sparse_tensor):
    """ Sort points in sparse tensor according to their coordinates.
    """

    indices_sort = np.argsort(array2vector2(sparse_tensor.C.cpu(),
                                            sparse_tensor.C.cpu().max() + 1))
    sparse_tensor_sort = ME.SparseTensor(features=sparse_tensor.F[indices_sort],
                                         coordinates=sparse_tensor.C[indices_sort],
                                         tensor_stride=sparse_tensor.tensor_stride[0],
                                         device=sparse_tensor.device)
    return sparse_tensor_sort


# def array2vector(array, step):
#     """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
#     """
#     array, step = array.long().cpu(), step.long().cpu()
#     vector = sum([array[:, i] * (step ** i) for i in range(array.shape[-1])])
#
#     return vector

def array2vector(array, step):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long(), step.long()
    vector = sum([array[:, i] * (step ** i) for i in range(array.shape[-1])])

    return vector


def array2vector2(array, step):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long().cpu(), step.long().cpu()
    vector = sum([array[:, 3 - i] * (step ** i) for i in range(array.shape[-1])])

    return vector


# def isin(data, ground_truth):
#     """ Input data and ground_truth are torch tensor of shape [N, D].
#     Returns a boolean vector of the same length as `data` that is True
#     where an element of `data` is in `ground_truth` and False otherwise.
#     """
#     device = data.device
#     data, ground_truth = data.cpu(), ground_truth.cpu()
#     step = torch.max(data.max(), ground_truth.max()) + 1
#     data = array2vector(data, step)
#     ground_truth = array2vector(ground_truth, step)
#     mask = np.isin(data.cpu().numpy(), ground_truth.cpu().numpy())
#
#     return torch.Tensor(mask).bool().to(device)

def isin(data, ground_truth):
    """ Input data and ground_truth are torch tensor of shape [N, D].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is in `ground_truth` and False otherwise.
    """
    device = data.device
    if len(ground_truth)==0:
        return torch.zeros([len(data)]).bool().to(device)
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = torch.isin(data.to(device), ground_truth.to(device))

    return mask
"""Ori
def istopk(data, nums, rho=1.000):
    #Input data is sparse tensor and nums is a list of shape [batch_size].
    #Returns a boolean vector of the same length as `data` that is True
    #where an element of `data` is the top k (=nums*rho) value and False otherwise.

    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N * rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)  # must CPU.
        mask[row_indices[indices]] = True

    return mask.bool().to(data.device)
"""


# def istopk(data, nums, rho=1.000):
#     mask = torch.zeros(len(data), dtype=torch.bool)
#     row_indices_per_batch = data._batchwise_row_indices
#     for row_indices, N in zip(row_indices_per_batch, nums):
#         k = int(min(len(row_indices), N * rho))
#         _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)  # must CPU.
#         mask[row_indices[indices]] = True
#     topk_add_1_indices = get_max_index2(data.F)
#     mask[topk_add_1_indices] = True
#     return mask.bool().to(data.device)

def istopk(data, nums, rho=1.0):
    '''
    Input data is sparse tensor and nums is a list of shape [batch_size].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is the top k (=nums*rho) value and False otherwise.
    '''
    mask = torch.zeros(len(data), dtype=torch.bool, device=data.device)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N*rho))
        _, indices = torch.topk(data.F[row_indices].squeeze(), k)
        mask[row_indices[indices]] = True
        topk_add_1_indices = get_max_index2(data.F)
        mask[topk_add_1_indices] = True
    return mask

def istopk1(data, nums, rho=1.000):
    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N * rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)  # must CPU.
        mask[row_indices[indices]] = True
    topk_add_1_indices = get_max_index2(data.F)
    mask[topk_add_1_indices] = True
    return mask.bool().to(data.device)


def get_max_index2(features):
    num_points = len(features)
    reshaped_features = features.view(-1, 8)
    sorted_indices = torch.argsort(reshaped_features, descending=True)
    max_indices = sorted_indices[:, 0]
    total_indices = max_indices + torch.arange(0, num_points, 8, device='cuda')

    return total_indices.tolist()


def istopk_ori(data, nums, rho=1.000):
    """ Input data is sparse tensor and nums is a list of shape [batch_size].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is the top k (=nums*rho) value and False otherwise.
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N * rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)  # must CPU.
        mask[row_indices[indices]] = True

    return mask.bool().to(data.device)


def sort_by_coor_sum(f, stride=None):
    if stride is None:
        stride = f.tensor_stride[0]
    xyz, feature = f.C, f.F
    maximum = xyz.max() + 1
    xyz, maximum = xyz.long(), maximum.long()
    coor_sum = xyz[:, 0] * maximum * maximum * maximum \
               + xyz[:, 1] * maximum * maximum \
               + xyz[:, 2] * maximum \
               + xyz[:, 3]
    _, idx = coor_sum.sort()
    xyz_, feature_ = xyz[idx].to(torch.float32), feature[idx]
    f_ = ME.SparseTensor(feature_, coordinates=xyz_, tensor_stride=stride, device=f.device)
    return f_


# def sort_spare_tensor(sparse_tensor):
#     """ Sort points in sparse tensor according to their coordinates.
#     """
#     indices_sort = np.argsort(array2vector(sparse_tensor.C.cpu(),
#                                            sparse_tensor.C.cpu().max() + 1))
#     sparse_tensor_sort = ME.SparseTensor(features=sparse_tensor.F[indices_sort],
#                                          coordinates=sparse_tensor.C[indices_sort],
#                                          tensor_stride=sparse_tensor.tensor_stride[0],
#                                          device=sparse_tensor.device)
#
#     return sparse_tensor_sort

# def sort_spare_tensor(sparse_tensor):
#     """ Sort points in sparse tensor according to their coordinates.
#     """
#     indices_sort = np.argsort(array2vector(sparse_tensor.C.cpu(),
#                                            sparse_tensor.C.cpu().max() + 1))
#     sparse_tensor_sort = ME.SparseTensor(features=sparse_tensor.F[indices_sort],
#                                          coordinates=sparse_tensor.C[indices_sort],
#                                          tensor_stride=sparse_tensor.tensor_stride[0],
#                                          device=sparse_tensor.device)
#
#     return sparse_tensor_sort, indices_sort

def sort_spare_tensor(sparse_tensor):
    """ Sort points in sparse tensor according to their coordinates.
    """
    indices_sort = np.argsort(array2vector(sparse_tensor.C.cpu(),
                                           sparse_tensor.C.cpu().max() + 1))
    sparse_tensor_sort = ME.SparseTensor(features=sparse_tensor.F[indices_sort],
                                         coordinates=sparse_tensor.C[indices_sort],
                                         tensor_stride=sparse_tensor.tensor_stride[0],
                                         device=sparse_tensor.device)

    return sparse_tensor_sort


# def load_sparse_tensor(filedir, device):
#     coords = torch.tensor(read_ply_ascii_geo(filedir)).int()
#     feats = torch.ones((len(coords), 1)).float()
#     # coords, feats = ME.utils.sparse_quantize(coordinates=coords, features=feats, quantization_size=1)
#     coords, feats = ME.utils.sparse_collate([coords], [feats])
#     x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)
#
#     return x

def load_sparse_tensor(filedir, device):
    # coords = torch.tensor(read_ply_ascii_geo(filedir)).int()
    coords, feats = torch.tensor(read_ply_ascii_geo(filedir)).int()
    # feats = torch.ones((len(coords), 1)).float()
    # coords, feats = ME.utils.sparse_quantize(coordinates=coords, features=feats, quantization_size=1)
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)

    return x


def load_sparse_tensor_attri(filedir, device):
    coords, attri = torch.tensor(read_ply_ascii_attri(filedir)).int()
    # prob = torch.ones((len(coords), 1)).float()
    # new_feats = torch.cat((prob, attri/255), dim=1)
    # prob = torch.ones((len(coords), 1)).float() * 255
    # new_feats = torch.cat((prob, attri), dim=1)
    new_feats = attri
    coords, feats = ME.utils.sparse_collate([coords], [new_feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)

    return x


def load_sparse_tensor_test(filedir, device):
    coords = torch.tensor(read_ply_ascii_geo_test(filedir)).int()
    feats = torch.ones((len(coords), 1)).float()
    # coords, feats = ME.utils.sparse_quantize(coordinates=coords, features=feats, quantization_size=1)
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)

    return x


def scale_sparse_tensor(x, factor):
    coords = (x.C[:, 1:] * factor).round().int()
    feats = torch.ones((len(coords), 1)).float()
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=x.device)

    return x


def read_h5_attri(filedir):
    pc = h5py.File(filedir, 'r')['data'][:]
    coords = pc[:, 0:3].astype('int')
    attri = pc[:, 3:6].astype('int')

    return coords, attri


def read_ply_ascii_attri(filename):
    '''
    load data from ply file.
    '''
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)
    attribute = np.asarray(pcd.colors)
    attribute = np.around(attribute * 254)
    points = points.astype(np.int32)  # np.uint8
    attribute = attribute.astype(np.int32)  # np.uint8
    return points, attribute