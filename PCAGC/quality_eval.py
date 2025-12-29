import color_space
from scipy.spatial import cKDTree
import numpy as np
import torch


def color(ori_colors, dist_colors):
    ori_colors = color_space.rgb_to_yuv(ori_colors)
    dist_colors = color_space.rgb_to_yuv(dist_colors)
    mae = np.mean(np.abs((ori_colors - dist_colors) / 255.), axis=0)
    mse = np.mean(np.square((ori_colors - dist_colors) / 255.), axis=0)
    psnr = -10 * np.log10(mse)

    return mse, psnr, mae


def color_with_geo(ori_pts, dist_pts):
    # ori_geo = ori_pts[:, :3]
    # dist_geo = dist_pts[:, :3]
    # ori_col = ori_pts[:, 3:]
    # dist_col = dist_pts[:, 3:]
    ori_geo = ori_pts.C[:, 1:].cpu().numpy()
    ori_col = ori_pts.F.cpu().numpy()
    dist_geo = dist_pts.C[:, 1:].cpu().numpy()
    dist_col = dist_pts.F.cpu().numpy()

    fwd_tree = cKDTree(dist_geo, balanced_tree=False)
    _, fwd_idx = fwd_tree.query(ori_geo)
    fwd_colors = dist_col[fwd_idx]
    fwd_metrics = color(fwd_colors, ori_col)

    bwd_tree = cKDTree(ori_geo, balanced_tree=False)
    _, bwd_idx = bwd_tree.query(dist_geo)
    bwd_colors = ori_col[bwd_idx]
    bwd_metrics = color(bwd_colors, dist_col)

    assert len(fwd_metrics) == len(bwd_metrics),\
        f'found len(fwd_metrics) = {len(fwd_metrics)} != len(bwd_metrics) = {len(bwd_metrics)}'

    final_metrics = tuple([np.min((fwd_metrics[i], bwd_metrics[i]), axis=0) for i in range(len(fwd_metrics))])

    return final_metrics, fwd_metrics, bwd_metrics

def color_with_geo_1(ori_pts, dist_pts):
    ori_col = ori_pts.F.cpu().numpy()
    dist_col = dist_pts.F.cpu().numpy()

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


    fwd_metrics = color(fwd_col.cpu().numpy(), ori_col)
    bwd_metrics = color(bwd_col.cpu().numpy(), dist_col)

    assert len(fwd_metrics) == len(bwd_metrics),\
        f'found len(fwd_metrics) = {len(fwd_metrics)} != len(bwd_metrics) = {len(bwd_metrics)}'

    final_metrics = tuple([np.min((fwd_metrics[i], bwd_metrics[i]), axis=0) for i in range(len(fwd_metrics))])

    return final_metrics, fwd_metrics, bwd_metrics
