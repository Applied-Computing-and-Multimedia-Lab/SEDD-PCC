import os, time
import numpy as np
import torch
import MinkowskiEngine as ME
from functional import bound
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
from data_utils import array2vector, istopk, sort_spare_tensor, load_sparse_tensor, scale_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo_test, sort_spare_tensor2, array2vector2

from gpcc import gpcc_encode, gpcc_decode
from pc_error import pc_error
from pcc_model import UModel
from cube_cut_eval import preprocess_eval, save_points, prune_voxel
from Mortonsort import Mortonsort, Mortonsort_C
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import numpy as np


class CoordinateCoder():
    """encode/decode coordinates using gpcc
    """

    def __init__(self, filename):
        self.filename = filename
        self.ply_filename = filename + '.ply'

    def encode(self, coords, postfix=''):
        coords = coords.numpy().astype('int')
        write_ply_ascii_geo(filedir=self.ply_filename, coords=coords)
        gpcc_encode(self.ply_filename, self.filename + postfix + '_C.bin')
        # os.system('rm ' + self.ply_filename)

        return

    def decode(self, postfix=''):
        gpcc_decode(self.filename + postfix + '_C.bin', self.ply_filename)
        coords = read_ply_ascii_geo_test(self.ply_filename)
        # os.system('rm ' + self.ply_filename)

        return coords


class FeatureCoder():
    """encode/decode feature using learned entropy model
    """

    def __init__(self, filename, entropy_model):
        self.filename = filename
        self.entropy_model = entropy_model.cpu()

    def encode(self, feats, postfix=''):
        strings, min_v, max_v = self.entropy_model.compress(feats.cpu())
        shape = feats.shape
        with open(self.filename + postfix + '_F.bin', 'wb') as fout:
            fout.write(strings)
        with open(self.filename + postfix + '_H.bin', 'wb') as fout:
            fout.write(np.array(shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(min_v), dtype=np.int8).tobytes())
            fout.write(np.array(min_v, dtype=np.float32).tobytes())
            fout.write(np.array(max_v, dtype=np.float32).tobytes())

        return

    def decode(self, postfix=''):
        with open(self.filename + postfix + '_F.bin', 'rb') as fin:
            strings = fin.read()
        with open(self.filename + postfix + '_H.bin', 'rb') as fin:
            shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
            len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            min_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)[0]
            max_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)[0]

        feats = self.entropy_model.decompress(strings, min_v, max_v, shape, channels=shape[-1])

        return feats



class ICoder():
    def __init__(self, model, filename):
        self.model = model.to(device)
        self.filename = filename
        self.coordinate_coder = CoordinateCoder(filename)
        self.feature_coder = FeatureCoder(self.filename, model.entropy_bottleneck)
        self.prune_voxel = prune_voxel()

    @torch.no_grad()
    def encode(self, Tstage, x, postfix=''):
        if Tstage == 1:
            input = x
            y_list = self.model.encoder(x)
            y1 = y_list[0]
            ground_truth_list = y_list[1:] + [x]
            self.feature_coder.encode(y1.F, postfix=postfix)
            # import matplotlib.pyplot as plt
            # # Assuming y is an instance of ME.SparseTensor
            # features = y1.F.detach().cpu().numpy().flatten()  # Extract features and convert to numpy array
            # # Plot the histogram of the features
            # # Define custom bin edges (for example, from -10 to 10 with a step size of 0.5)
            # bin_edges = np.arange(-10, 10, 1)  # Adjust the range and step size as needed
            # # Plot the histogram with custom bin edges
            # plt.figure(figsize=(8, 5))
            # plt.hist(features, bins=bin_edges)  # Use the custom bin edges
            # plt.title("Feature Distribution of SparseTensor")
            # plt.xlabel("Feature Value")
            # plt.ylabel("Frequency")
            # plt.grid(False)
            # plt.show()

            return y1

        if Tstage == 2:
            input = x
            y_list = self.model.encoder(x)
            y = y_list[0]
            y = sort_spare_tensor(y_list[0])
            num_points = [len(ground_truth) for ground_truth in y_list[1:] + [x]]
            with open(self.filename + postfix + '_num_points.bin', 'wb') as f:
                f.write(np.array(num_points, dtype=np.int32).tobytes())
            self.feature_coder.encode(y.F, postfix=postfix)
            import matplotlib.pyplot as plt
            # Assuming y is an instance of ME.SparseTensor
            features = y.F.detach().cpu().numpy().flatten()  # Extract features and convert to numpy array
            # Plot the histogram of the features
            # Define custom bin edges (for example, from -10 to 10 with a step size of 0.5)
            bin_edges = np.arange(-10, 10, 1)  # Adjust the range and step size as needed
            # Plot the histogram with custom bin edges
            plt.figure(figsize=(8, 5))
            plt.hist(features, bins=bin_edges)  # Use the custom bin edges
            plt.title("Feature Distribution of SparseTensor")
            plt.xlabel("Feature Value")
            plt.ylabel("Frequency")
            plt.grid(False)
            plt.show()

            self.coordinate_coder.encode(
                (torch.div(y.C, y.tensor_stride[0], rounding_mode='trunc')).detach().cpu()[:, 1:], postfix=postfix)
            return y


        if Tstage == 3:
            # Encoder
            input = x
            y_list = self.model.encoder(x)
            y = sort_spare_tensor(y_list[0])
            num_points = [len(ground_truth) for ground_truth in y_list[1:] + [x]]
            with open(self.filename + postfix + '_num_points.bin', 'wb') as f:
                f.write(np.array(num_points, dtype=np.int32).tobytes())
            self.feature_coder.encode(y.F, postfix=postfix)
            self.coordinate_coder.encode(
                (torch.div(y.C, y.tensor_stride[0], rounding_mode='trunc')).detach().cpu()[:, 1:], postfix=postfix)
            return y


    @torch.no_grad()
    def decode(self, Tstage, postfix='',rho=1,y_key='', y_manager=''):
        if Tstage == 1:
            y_F = self.feature_coder.decode(postfix=postfix)

            y = ME.SparseTensor(
                features=y_F,
                coordinate_map_key=y_key,
                coordinate_manager=y_manager,
                device=device)

            out_list, out ,ori= self.model.decoder_a(y)
            F = bound(out.F, 0, 255)
            out = ME.SparseTensor(
                features=F,
                coordinate_map_key=out.coordinate_map_key,
                coordinate_manager=out.coordinate_manager,
                device=out.device)
            return out

        if Tstage == 2:
            # decode coords
            y_C = self.coordinate_coder.decode(postfix=postfix)
            y_C = torch.cat((torch.zeros((len(y_C), 1)).int(), torch.tensor(y_C).int()), dim=-1)
            indices_sort = np.argsort(array2vector(y_C, y_C.max() + 1))
            y_C = y_C[indices_sort]
            # decode feat
            y_F = self.feature_coder.decode(postfix=postfix)
            y = ME.SparseTensor(features=y_F, coordinates=y_C * 8,
                                tensor_stride=8, device=device)





            # decode label
            with open(self.filename + postfix + '_num_points.bin', 'rb') as fin:
                num_points = np.frombuffer(fin.read(4 * 3), dtype=np.int32).tolist()
                num_points[-1] = int(rho * num_points[-1])  # update
                num_points = [[num] for num in num_points]
            outlist_G, out, ori = self.model.decoder_g(y, nums_list=num_points, ground_truth_list=[None] * 3,training=False)


            return out

        if Tstage == 3:
            # decode coords
            y_C = self.coordinate_coder.decode(postfix=postfix)
            y_C = torch.cat((torch.zeros((len(y_C), 1)).int(), torch.tensor(y_C).int()), dim=-1)
            indices_sort = np.argsort(array2vector(y_C, y_C.max() + 1))
            y_C = y_C[indices_sort]
            # decode feat
            y_F = self.feature_coder.decode(postfix=postfix)
            y = ME.SparseTensor(features=y_F, coordinates=y_C * 8,
                                tensor_stride=8, device=device)

            # decode label
            with open(self.filename + postfix + '_num_points.bin', 'rb') as fin:
                num_points = np.frombuffer(fin.read(4 * 3), dtype=np.int32).tolist()
                num_points[-1] = int(rho * num_points[-1])  # update
                num_points = [[num] for num in num_points]

            outlist_G, y_hat, ori = self.model.decoder_g(y, nums_list=num_points, ground_truth_list=[None] * 3,
                                                         training=False)
            with torch.no_grad():
                y_hat_g = ME.SparseTensor(coordinates=y_hat.C, features=y_hat.F, device=y_hat.device)
                _, coord2 = self.model.get_coordinate(y_hat_g)
            y_a = ME.SparseTensor(
                features=y.F,
                coordinate_map_key=coord2.coordinate_map_key,
                coordinate_manager=coord2.coordinate_manager,
                device=y.device)

            # Decoder_A
            outlist_A, out, _ = self.model.decoder_a(y_a)

            out = self.prune_voxel(out, num_points[2], rho=1.00)

            F = bound(out.F, 0, 255)

            out0 = ME.SparseTensor(
                features=F,
                coordinate_map_key=out.coordinate_map_key,
                coordinate_manager=out.coordinate_manager,
                device=out.device)

            return out0