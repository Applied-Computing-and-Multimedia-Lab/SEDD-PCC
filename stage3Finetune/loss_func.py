from loss import get_bce_mse, get_bce, get_mse, rgb2yuv
import torch
from scipy.spatial import cKDTree
import MinkowskiEngine as ME


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

# def yuv_mse(ori_pts, dist_pts):
#
#     ori_col = ori_pts.F
#     dist_col = dist_pts.F
#
#     ################################     batch sort   ########################################
#     batch_size = torch.max(dist_pts.C[:, 0])
#     for i in range(batch_size + 1):
#         mask = dist_pts.C[:, 0] == i
#         dist_c = dist_pts.C[mask][:, 1:].cpu().numpy()
#         dist_f = dist_pts.F[mask]
#
#         mask = ori_pts.C[:, 0] == i
#         ori_c = ori_pts.C[mask][:, 1:].cpu().numpy()
#         ori_f = ori_pts.F[mask]
#
#         fwd_tree = cKDTree(dist_c, balanced_tree=False)
#         _, fwd_idx = fwd_tree.query(ori_c)
#         fwd_colors = dist_f[fwd_idx]
#
#         bwd_tree = cKDTree(ori_c, balanced_tree=False)
#         _, bwd_idx = bwd_tree.query(dist_c)
#         bwd_colors = ori_f[bwd_idx]
#
#         if i == 0:
#             fwd_col = fwd_colors
#             bwd_col = bwd_colors
#         else:
#             fwd_col = torch.cat((fwd_col, fwd_colors), dim=0)
#             bwd_col = torch.cat((bwd_col, bwd_colors), dim=0)
#
#     dist_colors, ori_colors = rgb2yuv(fwd_col, ori_col)
#     y_mse = torch.nn.functional.mse_loss(ori_colors[:, 0], dist_colors[:, 0])
#     u_mse = torch.nn.functional.mse_loss(ori_colors[:, 1], dist_colors[:, 1])
#     v_mse = torch.nn.functional.mse_loss(ori_colors[:, 2], dist_colors[:, 2])
#     mse_f = torch.stack([y_mse, u_mse, v_mse], dim=0)
#
#     dist_colors, ori_colors = rgb2yuv(dist_col, bwd_col)
#     y_mse = torch.nn.functional.mse_loss(ori_colors[:, 0], dist_colors[:, 0])
#     u_mse = torch.nn.functional.mse_loss(ori_colors[:, 1], dist_colors[:, 1])
#     v_mse = torch.nn.functional.mse_loss(ori_colors[:, 2], dist_colors[:, 2])
#     mse_b = torch.stack([y_mse, u_mse, v_mse], dim=0)
#
#     mse_list = torch.stack([mse_f, mse_b], dim=0).tolist()
#     # mse = (mse_f + mse_b) / 2
#     mse = torch.max(mse_f, mse_b)
#     return mse, mse_list

import pytorch3d.ops
def yuv_mse(ori_pts, dist_pts):
    ori_col = ori_pts.F
    dist_col = dist_pts.F
    ################################     batch sort   ########################################
    batch_size = torch.max(dist_pts.C[:, 0])
    for i in range(batch_size + 1):
        mask = dist_pts.C[:, 0] == i
        dist_c = dist_pts.C[mask][:, 1:].float()  # 保持為 tensor
        dist_f = dist_pts.F[mask]
        mask = ori_pts.C[:, 0] == i
        ori_c = ori_pts.C[mask][:, 1:].float()  # 保持為 tensor
        ori_f = ori_pts.F[mask]

        # 使用 PyTorch3D KNN 進行前向查詢 (ori -> dist)
        # 為每個 ori 點找到最近的 dist 點
        ori_c_batch = ori_c.unsqueeze(0)  # [1, N, 3]
        dist_c_batch = dist_c.unsqueeze(0)  # [1, M, 3]

        fwd_nn = pytorch3d.ops.knn_points(ori_c_batch, dist_c_batch, K=1)
        fwd_idx = fwd_nn.idx.squeeze(0).squeeze(-1)  # [N]
        fwd_colors = dist_f[fwd_idx]

        # 使用 PyTorch3D KNN 進行反向查詢 (dist -> ori)
        # 為每個 dist 點找到最近的 ori 點
        bwd_nn = pytorch3d.ops.knn_points(dist_c_batch, ori_c_batch, K=1)
        bwd_idx = bwd_nn.idx.squeeze(0).squeeze(-1)  # [M]
        bwd_colors = ori_f[bwd_idx]

        if i == 0:
            fwd_col = fwd_colors
            bwd_col = bwd_colors
        else:
            fwd_col = torch.cat((fwd_col, fwd_colors), dim=0)
            bwd_col = torch.cat((bwd_col, bwd_colors), dim=0)

    # 計算前向 MSE
    dist_colors, ori_colors = rgb2yuv(fwd_col, ori_col)
    y_mse = torch.nn.functional.mse_loss(ori_colors[:, 0], dist_colors[:, 0])
    u_mse = torch.nn.functional.mse_loss(ori_colors[:, 1], dist_colors[:, 1])
    v_mse = torch.nn.functional.mse_loss(ori_colors[:, 2], dist_colors[:, 2])
    mse_f = torch.stack([y_mse, u_mse, v_mse], dim=0)

    # 計算反向 MSE
    dist_colors, ori_colors = rgb2yuv(dist_col, bwd_col)
    y_mse = torch.nn.functional.mse_loss(ori_colors[:, 0], dist_colors[:, 0])
    u_mse = torch.nn.functional.mse_loss(ori_colors[:, 1], dist_colors[:, 1])
    v_mse = torch.nn.functional.mse_loss(ori_colors[:, 2], dist_colors[:, 2])
    mse_b = torch.stack([y_mse, u_mse, v_mse], dim=0)

    mse_list = torch.stack([mse_f, mse_b], dim=0).tolist()
    mse = torch.max(mse_f, mse_b)
    return mse, mse_list


def attri_loss (input, out_set):
    loss = 0

    out = out_set['out']
    GT = input
    out = ME.SparseTensor(features=out.F,
                        coordinate_map_key=out.coordinate_map_key,
                        coordinate_manager=out.coordinate_manager,
                        device=out.device)
    GT = ME.SparseTensor(features=GT.F,
                        coordinate_map_key=GT.coordinate_map_key,
                        coordinate_manager=GT.coordinate_manager,
                        device=GT.device)
    loss = sum(mse_loss(out, GT))

    return loss

def attri_loss_1 (input, out_set):
    loss = 0

    out3 = out_set['out_A']
    GT3 = input

    out3 = ME.SparseTensor(features=out3.F,
                        coordinate_map_key=out3.coordinate_map_key,
                        coordinate_manager=out3.coordinate_manager,
                        device=out3.device)
    GT3 = ME.SparseTensor(features=GT3.F,
                        coordinate_map_key=GT3.coordinate_map_key,
                        coordinate_manager=GT3.coordinate_manager,
                        device=GT3.device)
    mse, mse_list = yuv_mse(out3, GT3)

    loss = sum(mse)

    outscale1 = out_set['out_cls_list_A'][2]
    ground = out_set['ground_truth_list'][2]
    outscale1 = ME.SparseTensor(features=outscale1.F,
                           coordinate_map_key=outscale1.coordinate_map_key,
                           coordinate_manager=outscale1.coordinate_manager,
                           device=outscale1.device)

    ground = ME.SparseTensor(features=ground.F,
                          coordinate_map_key=ground.coordinate_map_key,
                          coordinate_manager=ground.coordinate_manager,
                          device=ground.device)

    mse1, mse_list1 = yuv_mse(outscale1, ground)

    loss1 = sum(mse1)

    total_loss = loss + loss1


    return total_loss, loss,loss1

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
