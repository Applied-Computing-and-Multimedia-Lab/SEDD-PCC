from loss import get_bce_mse, get_bce, get_bits, get_metrics, get_mse, rgb2yuv
import torch
from scipy.spatial import cKDTree
import MinkowskiEngine as ME
import numpy as np
import color_space

def geo_loss (input, out_set):
    bce = 0
    # bce
    for out_cls, ground_truth in zip(out_set['out_cls_list_G'], out_set['ground_truth_list']):
        curr_bce = get_bce(out_cls, ground_truth)
        bce += curr_bce
    # bce_hat
    out_cls = ME.SparseTensor(features=out_set['out_g'].F,
                              coordinate_map_key=out_set['out_g'].coordinate_map_key,
                              coordinate_manager=out_set['out_g'].coordinate_manager,
                              device=out_set['out_g'].device)
    ground_truth = ME.SparseTensor(features=input.F,
                                   coordinate_map_key=input.coordinate_map_key,
                                   coordinate_manager=input.coordinate_manager,
                                   device=input.device)
    bce_hat = get_bce(out_cls, ground_truth)

    bce_mse = get_bce_mse(out_cls, ground_truth)

    bce_list = [bce, bce_hat, bce_mse]

    bce_list1 = torch.stack([bce, bce_hat, bce_mse], dim=0).tolist()

    return bce_list, bce_list1

def mse_loss(ori_pts, dist_pts):
    ori_colors = ori_pts.F
    dist_colors = dist_pts.F
    dist_colors, ori_colors = rgb2yuv(dist_colors, ori_colors)
    y_mse = torch.nn.functional.mse_loss(ori_colors[:, 0], dist_colors[:, 0])
    u_mse = torch.nn.functional.mse_loss(ori_colors[:, 1], dist_colors[:, 1])
    v_mse = torch.nn.functional.mse_loss(ori_colors[:, 2], dist_colors[:, 2])
    mse = torch.stack([y_mse, u_mse, v_mse], dim=0)
    return mse

def yuv_mse(ori_pts, dist_pts):
    ori_geo = ori_pts.C[:, 1:].cpu().numpy()
    ori_col = ori_pts.F
    dist_geo = dist_pts.C[:, 1:].cpu().numpy()
    dist_col = dist_pts.F

    ################################     batch sort   ########################################
    batch_size = torch.max(dist_pts.C[:, 0])
    for i in range(batch_size + 1):
        mask = dist_pts.C[:, 0] == i
        dist_c = dist_pts.C[mask][:, 1:].cpu().numpy()
        dist_f = dist_pts.F[mask]

        mask = ori_pts.C[:, 0] == i
        ori_c = ori_pts.C[mask][:, 1:].cpu().numpy()
        ori_f = ori_pts.F[mask]

        fwd_tree = cKDTree(dist_c, balanced_tree=False)
        _, fwd_idx = fwd_tree.query(ori_c)
        fwd_colors = dist_f[fwd_idx]

        bwd_tree = cKDTree(ori_c, balanced_tree=False)
        _, bwd_idx = bwd_tree.query(dist_c)
        bwd_colors = ori_f[bwd_idx]

        if i == 0:
            fwd_col = fwd_colors
            bwd_col = bwd_colors
        else:
            fwd_col = torch.cat((fwd_col, fwd_colors), dim=0)
            bwd_col = torch.cat((bwd_col, bwd_colors), dim=0)

    dist_colors, ori_colors = rgb2yuv(fwd_col, ori_col)
    y_mse = torch.nn.functional.mse_loss(ori_colors[:, 0], dist_colors[:, 0])
    u_mse = torch.nn.functional.mse_loss(ori_colors[:, 1], dist_colors[:, 1])
    v_mse = torch.nn.functional.mse_loss(ori_colors[:, 2], dist_colors[:, 2])
    mse_f = torch.stack([y_mse, u_mse, v_mse], dim=0)

    dist_colors, ori_colors = rgb2yuv(dist_col, bwd_col)
    y_mse = torch.nn.functional.mse_loss(ori_colors[:, 0], dist_colors[:, 0])
    u_mse = torch.nn.functional.mse_loss(ori_colors[:, 1], dist_colors[:, 1])
    v_mse = torch.nn.functional.mse_loss(ori_colors[:, 2], dist_colors[:, 2])
    mse_b = torch.stack([y_mse, u_mse, v_mse], dim=0)

    mse_list = torch.stack([mse_f, mse_b], dim=0).tolist()
    # mse = (mse_f + mse_b) / 2
    mse = torch.max(mse_f, mse_b)
    return mse, mse_list


def teach_loss_G_mse (out_set, out_set_t):
    loss = 0
    loss = torch.nn.functional.mse_loss(out_set['SM_G'].F, out_set_t['TM_G'].F)
    return loss

def teach_loss_G_bce (out_set, out_set_t):

    bce_hat = 0
    # bce_hat
    out_cls = ME.SparseTensor(features=out_set['out_g'].F,
                              coordinate_map_key=out_set['out_g'].coordinate_map_key,
                              coordinate_manager=out_set['out_g'].coordinate_manager,
                              device=out_set['out_g'].device)
    ground_truth = ME.SparseTensor(features=out_set_t['out'].F,
                                   coordinate_map_key=out_set_t['out'].coordinate_map_key,
                                   coordinate_manager=out_set_t['out'].coordinate_manager,
                                   device=out_set_t['out'].device)
    bce_hat = get_bce(out_cls, ground_truth)
    return bce_hat

def teach_loss_A (out_set, out_set_t):
    loss = 0

    loss = torch.nn.functional.mse_loss(out_set['SM_A'].F, out_set_t['TM_A'].F)
        
    return loss

def YUV_loss (diff_scale, diff_scale1, input, out_set):# YUV loss
    mse, mse1, mse_list , all_mses_list, all_mses_list_1 = 0, 0, [], [], []

    #ori and decode output
    out1, ground1 = rgb2yuv(out_set['out_cls_list'][2].F, out_set['ground_truth_list'][2].F)
    for out_cls, ground_truth in zip([out1[:, 0], out1[:, 1], out1[:, 2]], [ground1[:, 0], ground1[:, 1], ground1[:, 2]]):
        curr_mse = get_mse(out_cls, ground_truth)
        all_mses_list_1.append(curr_mse)
        all_mses_list.append(curr_mse.item())
    mse = diff_scale[0] * all_mses_list_1[0] + diff_scale[1] * all_mses_list_1[1] + diff_scale[2] * all_mses_list_1[2]

    out2, ground2= rgb2yuv(out_set['out'].F, input.F)
    for out_cls, ground_truth in zip([out2[:, 0], out2[:, 1], out2[:, 2]], [ground2[:, 0], ground2[:, 1], ground2[:, 2]]):
        curr_mse = get_mse(out_cls, ground_truth)
        all_mses_list_1.append(curr_mse)
        all_mses_list.append(curr_mse.item())
    mse1 = diff_scale1[0] * all_mses_list_1[3] + diff_scale1[1] * all_mses_list_1[4] + diff_scale1[2] * all_mses_list_1[5]
    mse_list = [mse, mse1]
    return mse_list, all_mses_list

