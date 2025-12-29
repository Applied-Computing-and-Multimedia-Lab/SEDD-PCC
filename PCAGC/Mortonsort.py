import numpy as np
import point_cloud_utils as pcu
import torch
import MinkowskiEngine as ME


def Mortonsort(input):
    batch_size = torch.max(input.C[:, 0])
    for i in range(batch_size + 1):
        mask = input.C[:, 0] == i
        input_C = input.C[mask]
        point_c = input_C[:, 1:]
        point_c = (point_c // input.tensor_stride[0]).cpu().numpy()
        input_F = input.F[mask]
        eps = 1.0 / 128.0
        pts_quantized = (point_c / eps).astype(np.int32)
        morton_codes = pcu.morton_encode(pts_quantized)
        indices_sort = [np.argsort(morton_codes)]
        input_C_sort = input_C[indices_sort]
        input_F_sort = input_F[indices_sort]
        if i == 0:
            input_feature = input_F_sort
            input_corr = input_C_sort
        else:
            input_feature = torch.cat((input_feature, input_F_sort), dim=0)
            input_corr = torch.cat((input_corr, input_C_sort), dim=0)
    sparse_tensor_sort = ME.SparseTensor(features=input_feature,
                                         coordinates=input_corr,
                                         tensor_stride=input.tensor_stride[0],
                                         device=input.device)
    return sparse_tensor_sort

def Mortonsort_C(input_C):
    eps = 1.0 / 128.0
    input_CC = input_C.cpu().numpy()[:, 1:]
    pts_quantized = (input_CC / eps).astype(np.int32)
    morton_codes = pcu.morton_encode(pts_quantized)
    indices_sort = [np.argsort(morton_codes)]
    input_C_sort = input_C[indices_sort]

    return input_C_sort

if __name__ == '__main__':
    device = "cuda:0"
    # device = "cpu"
    t = torch.rand(1, 1, 128, 128, 128)
    t *= t.abs().lt(0.02)
    input = ME.to_sparse(t, device=device)

    input2 = Mortonsort(input)

    print("finish")
