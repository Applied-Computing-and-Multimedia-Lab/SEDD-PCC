from loss import get_mse,  rgb2yuv


def YUV_loss (diff_scale, diff_scale1, input, out_set):# YUV loss
    mse, mse1, mse_list , all_mses_list, all_mses_list_1 = 0, 0, [], [], []

    #ori and decode output
    out1, ground1 = rgb2yuv(out_set['out_cls_list_A'][2].F, out_set['ground_truth_list'][2].F)
    for out_cls, ground_truth in zip([out1[:, 0], out1[:, 1], out1[:, 2]], [ground1[:, 0], ground1[:, 1], ground1[:, 2]]):
        curr_mse = get_mse(out_cls, ground_truth)
        all_mses_list_1.append(curr_mse)
        all_mses_list.append(curr_mse.item())
    mse = diff_scale[0] * all_mses_list_1[0] + diff_scale[1] * all_mses_list_1[1] + diff_scale[2] * all_mses_list_1[2]

    out2, ground2= rgb2yuv(out_set['out_A'].F, input.F)
    for out_cls, ground_truth in zip([out2[:, 0], out2[:, 1], out2[:, 2]], [ground2[:, 0], ground2[:, 1], ground2[:, 2]]):
        curr_mse = get_mse(out_cls, ground_truth)
        all_mses_list_1.append(curr_mse)
        all_mses_list.append(curr_mse.item())
    mse1 = diff_scale1[0] * all_mses_list_1[3] + diff_scale1[1] * all_mses_list_1[4] + diff_scale1[2] * all_mses_list_1[5]
    # mse_list = [mse, mse1]
    return mse,mse1

