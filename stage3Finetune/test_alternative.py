import torch
import numpy as np
import os
from pcc_model_alternative import UModel
from ICoder import ICoder
from data_utils import scale_sparse_tensor,load_sparse_tensor_attri
from data_utils import  write_ply_ascii_attri
from pc_error import pc_error


rootdir = os.path.split(__file__)[0]
import time
import pandas as pd

timestr = time.strftime("%m%d-%H%M")

os.environ["OMP_NUM_THREADS"] = "24"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def test(filedir, ckptdir_list,Tstage, outdir, resultdir,step='0',scaling_factor=1.0,
         rho=1.0, res=1024):
    # load data
    start_time = time.time()
    x = load_sparse_tensor_attri(filedir, device)
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')

    x = x.float()


    print('=' * 10, os.path.split(filedir)[-1].split('.')[0], '=' * 10)
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')

    # output filename
    if not os.path.exists(outdir): os.makedirs(outdir)
    filename = os.path.join(outdir, os.path.split(filedir)[-1].split('.')[0])
    print('output filename:\t', filename)

    # load model
    model = UModel()


    for idx, ckptdir in enumerate(ckptdir_list):
        print('=' * 10, idx + 1, '=' * 10)
        # load checkpoints
        assert os.path.exists(ckptdir)
        ckpt = torch.load(ckptdir)
        # model = UModel().to(device)
        model.load_state_dict(ckpt['model'])
        # model.update(force=True)
        print('load checkpoint from \t', ckptdir)


        if (step=='0'):
            coder = ICoder(model=model, filename=filename)


        # postfix: rate index
        postfix_idx = '_r' + str(idx + 1)
        # down-scale
        if scaling_factor != 1:
            x_in = scale_sparse_tensor(x, factor=scaling_factor)
        else:
            x_in = x

        # get_coordinate
        _, main_coord = model.get_coordinate(x_in)

        with torch.no_grad():
            # encode
            start_time = time.time()

            if (step=='0'):
                _ = coder.encode(Tstage, x_in, postfix=postfix_idx)



            print('Enc Time:\t', round(time.time() - start_time, 3), 's')
            time_enc = round(time.time() - start_time, 3)
            torch.cuda.empty_cache()  # empty cache.
            # decode
            start_time = time.time()

            if(step=='0'):
                x_dec = coder.decode(Tstage, postfix=postfix_idx, rho=rho,y_key=main_coord.coordinate_map_key,y_manager=main_coord.coordinate_manager)

            print('Dec Time:\t', round(time.time() - start_time, 3), 's')
            time_dec = round(time.time() - start_time, 3)
            torch.cuda.empty_cache()  # empty cache.

        # up-scale
        if scaling_factor != 1:
            x_dec = scale_sparse_tensor(x_dec, factor=1.0 / scaling_factor)

        if Tstage == 1:
            bits = np.array([os.path.getsize(filename + postfix_idx + postfix) * 8 \
                             for postfix in ['_F.bin', '_H.bin']])
        # bitrate
        else:
            bits = np.array([os.path.getsize(filename + postfix_idx + postfix) * 8 \
                         for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])


        bpps = (bits / len(x)).round(3)

        print('bits:\t', sum(bits), '\nbpps:\t', sum(bpps).round(3))


        # distortion
        write_ply_ascii_attri(filename + postfix_idx + '_dec_attri.ply', x_dec.C.detach().cpu().numpy()[:, 1:],
                              x_dec.F.detach().cpu().numpy())
        print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

        start_time = time.time()
        pc_error_metrics = pc_error(filedir, filename + postfix_idx + '_dec_attri.ply',
                                    res=res, normal=False, show=False)
        print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')

        print('bpp:\t', sum(bpps).round(3))
        print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
        print('D2 PSNR:\t', pc_error_metrics["mseF,PSNR (p2plane)"][0])
        print('YUV PSNR:\t', [pc_error_metrics["c[0],PSNRF"][0], pc_error_metrics["c[1],PSNRF"][0],
                              pc_error_metrics["c[2],PSNRF"][0]])
        result = {
            'Frame': 0,
            'num_points(input)': len(x),
            'num_points(output)': len(x_dec),
            'bits': sum(bits).round(3),
            'bpp(total)': sum(bpps).round(3),
            'bpp(coords)': bpps[0],
            'bpp(feats)': bpps[1],
            'D1-PSNR': pc_error_metrics["mseF,PSNR (p2point)"][0],
            'D2-PSNR': pc_error_metrics["mseF,PSNR (p2plane)"][0],
            'Y-PSNR': pc_error_metrics["c[0],PSNRF"][0],
            'U-PSNR': pc_error_metrics["c[1],PSNRF"][0],
            'V-PSNR': pc_error_metrics["c[2],PSNRF"][0],
            # 'resolution': res,
            'time(enc)': time_enc,
            'time(dec)': time_dec
        }
        results = pd.DataFrame.from_dict(result, orient='index').T
        torch.cuda.empty_cache()  # empty cache.


    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", default='/home/user/Desktop/unified/SEDDPCC/stage3Finetune/test_data/8iVFB')
    # parser.add_argument("--filedir", default='/home/user/Desktop/unified/SEDDPCC/stage3Finetune/test_data/Owlii')
    parser.add_argument("--Tstage", type=int, default=3, help='training stage')
    parser.add_argument("--outdir", default='/home/user/Desktop/unified/SEDDPCC/stage3Finetune/output')
    parser.add_argument("--resultdir", default='./results/'+timestr+'/')
    parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    parser.add_argument("--res", type=int, default=1024, help='resolution')
    parser.add_argument("--rho", type=float, default=1.0,help='the ratio of the number of output points to the number of input points')

    ckptdir_list = [
                    '/home/user/Desktop/unified/SEDDPCC/stage3Finetune/ckpts/ablation/R1.pth'


                    ]


    folder_path = './output/1120'

    args = parser.parse_args()

    if not os.path.exists(args.outdir): os.makedirs(args.outdir)
    if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)


    filedir_list = []

    for root, dirs, files in os.walk(args.filedir):
        for file in files:
            if file.endswith('.ply'):
                file_path = os.path.join(root, file)
                filedir_list.append(file_path)

    filedir_list = sorted(filedir_list)



    RD = './model_vox11_00000001'

    results = pd.DataFrame()

    for idx, ckptdir in enumerate(ckptdir_list):
        for i, filedir in enumerate(filedir_list):

            print("Frame_idx = ", i + 1)

            # step = (i + 1) % 32
            step = 1

            if step == 1:

                latent_buffer = list()
                all_results = test(filedir=filedir,
                                            ckptdir_list=[ckptdir],
                                            outdir=args.outdir,
                                            resultdir=args.resultdir,
                                            Tstage=args.Tstage,
                                            step='0',
                                            scaling_factor=args.scaling_factor,
                                            rho=args.rho,
                                            res=args.res,
                                            )





            # all_results['Frame']=i+1
            if i == 0:
                all_results['Frame'] = i + 1
                results = all_results.copy(deep=True)
            else:
                all_results['Frame'] = i + 1
                results = pd.concat([results, all_results], ignore_index=True)

        csv_name = os.path.join(args.resultdir, 'model_vox10_' + 'R' + str(idx + 1) + '_RC_result_.csv')
        results.to_csv(csv_name, index=False)

    print('-------------------Finish-------------------')

