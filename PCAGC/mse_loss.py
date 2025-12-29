from loss import get_mse, get_mse_F, get_bits, psnr, yuv_psnr, rgb2yuv

def all_scale_loss (diff_scale, diff_scale1, diff_scale2, input, out_set):# all scale loss
    mse, mse1, mse2, mse3, mse_list , all_mses_list, all_mses_list_1= 0, 0, 0, 0, [], [], []
    for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
        curr_mse = get_mse_F(out_cls, ground_truth)
        all_mses_list_1.append(curr_mse)
        all_mses_list.append(curr_mse.item())
    mse = diff_scale[0] * all_mses_list_1[0] + diff_scale[1] * all_mses_list_1[1] + diff_scale[2] * all_mses_list_1[2]
    for out_cls, ground_truth in zip(out_set['out_cls_list2'], out_set['ground_truth_list2']):
        curr_mse = get_mse_F(out_cls, ground_truth)
        all_mses_list_1.append(curr_mse)
        all_mses_list.append(curr_mse.item())
    mse1 = diff_scale1[0] * all_mses_list_1[3] + diff_scale1[1] * all_mses_list_1[4] + diff_scale1[2] * all_mses_list_1[5]
    for out_cls, ground_truth in zip(out_set['out_cls_list3'], out_set['ground_truth_list']):
        curr_mse = get_mse_F(out_cls, ground_truth)
        all_mses_list_1.append(curr_mse)
        all_mses_list.append(curr_mse.item())
    mse2 = diff_scale2[0] * all_mses_list_1[6] + diff_scale2[1] * all_mses_list_1[7] + diff_scale2[2] * all_mses_list_1[8]
    for out_cls, ground_truth in zip([out_set['out'].F[:, 0],
                                      out_set['out'].F[:, 1],
                                      out_set['out'].F[:, 2]],
                                     [input.F[:, 0],
                                      input.F[:, 1],
                                      input.F[:, 2]]):
        curr_mse = get_mse(out_cls, ground_truth)
        mse3 += curr_mse
        all_mses_list.append(curr_mse.item())
    mse_list = [mse, mse1, mse2, mse3]
    return mse_list, all_mses_list

def RGB_loss (input, out_set):# all scale loss
    mse, mse1, mse2, mse3, mse_list , all_mses_list = 0, 0, 0, 0, [], []
    for out_cls, ground_truth in zip([out_set['out_cls_list'][2].F[:, 0],
                                      out_set['out_cls_list'][2].F[:, 1],
                                      out_set['out_cls_list'][2].F[:, 2]],
                                     [out_set['ground_truth_list'][2].F[:, 0],
                                      out_set['ground_truth_list'][2].F[:, 1],
                                      out_set['ground_truth_list'][2].F[:, 2]]):
        curr_mse = get_mse(out_cls, ground_truth)
        mse += curr_mse
        all_mses_list.append(curr_mse.item())
    for out_cls, ground_truth in zip([out_set['out_cls_list2'][2].F[:, 0],
                                      out_set['out_cls_list2'][2].F[:, 1],
                                      out_set['out_cls_list2'][2].F[:, 2]],
                                     [out_set['ground_truth_list2'][2].F[:, 0],
                                      out_set['ground_truth_list2'][2].F[:, 1],
                                      out_set['ground_truth_list2'][2].F[:, 2]]):
        curr_mse = get_mse(out_cls, ground_truth)
        mse1 += curr_mse
        all_mses_list.append(curr_mse.item())
    for out_cls, ground_truth in zip([out_set['out_cls_list3'][2].F[:, 0],
                                      out_set['out_cls_list3'][2].F[:, 1],
                                      out_set['out_cls_list3'][2].F[:, 2]],
                                     [out_set['ground_truth_list'][2].F[:, 0],
                                      out_set['ground_truth_list'][2].F[:, 1],
                                      out_set['ground_truth_list'][2].F[:, 2]]):
        curr_mse = get_mse(out_cls, ground_truth)
        mse2 += curr_mse
        all_mses_list.append(curr_mse.item())
    for out_cls, ground_truth in zip([out_set['out'].F[:, 0],
                                      out_set['out'].F[:, 1],
                                      out_set['out'].F[:, 2]],
                                     [input.F[:, 0],
                                      input.F[:, 1],
                                      input.F[:, 2]]):
        curr_mse = get_mse(out_cls, ground_truth)
        mse3 += curr_mse
        all_mses_list.append(curr_mse.item())
    mse_list = [mse, mse1, mse2, mse3]
    return mse_list, all_mses_list




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
    mse_list = [mse, mse1]
    return mse_list, all_mses_list


def YUV2_loss (diff_scale1, input, out_set):# YUV2 loss
    mse, mse1, mse2, mse3, mse_list , all_mses_list, all_mses_list_1 = 0, 0, 0, 0, [], [], []
    out1, ground1 = rgb2yuv(out_set['out_cls_list'][2].F, out_set['ground_truth_list'][2].F)
    for out_cls, ground_truth in zip([out1[:, 0], out1[:, 1], out1[:, 2]], [ground1[:, 0], ground1[:, 1], ground1[:, 2]]):
        curr_mse = get_mse(out_cls, ground_truth)
        mse += curr_mse
        all_mses_list.append(curr_mse.item())
    for out_cls, ground_truth in zip(out_set['out_cls_list2'], out_set['ground_truth_list2']):
        curr_mse = get_mse_F(out_cls, ground_truth)
        all_mses_list_1.append(curr_mse)
        all_mses_list.append(curr_mse.item())
    mse1 = diff_scale1[0] * all_mses_list_1[0] + diff_scale1[1] * all_mses_list_1[1] + diff_scale1[2] * all_mses_list_1[2]
    out3, ground3 = rgb2yuv(out_set['out_cls_list3'][2].F, out_set['ground_truth_list'][2].F)
    for out_cls, ground_truth in zip([out3[:, 0], out3[:, 1], out3[:, 2]], [ground3[:, 0], ground3[:, 1], ground3[:, 2]]):
        curr_mse = get_mse(out_cls, ground_truth)
        mse2 += curr_mse
        all_mses_list.append(curr_mse.item())
    out4, ground4 = rgb2yuv(out_set['out'].F, input.F)
    for out_cls, ground_truth in zip([out4[:, 0], out4[:, 1], out4[:, 2]], [ground4[:, 0], ground4[:, 1], ground4[:, 2]]):
        curr_mse = get_mse(out_cls, ground_truth)
        mse3 += curr_mse
        all_mses_list.append(curr_mse.item())
    mse_list = [mse, mse1, mse2, mse3]
    return mse_list, all_mses_list