import torch
import numpy as np
import MinkowskiEngine as ME
import time
import argparse
import os
import random
import sys
from data_utils import isin, istopk,istopk_ori
import MinkowskiEngine as ME


def points2voxels(set_points, cube_size):
    """Transform points to voxels (binary occupancy map).
    Args: points list; cube size;

    Return: A tensor with shape [batch_size, cube_size, cube_size, cube_size, 1]
    """

    voxels = []
    vol000 =[]
    vol001 = []
    for _, points in enumerate(set_points):
        points = points.astype("int")
        vol = np.zeros((cube_size,cube_size,cube_size))
        vol[points[:,0],points[:,1],points[:,2]] = 1.0
        vol = np.expand_dims(vol,-1)
        vol000.append(points)
        vol001.append(1)
        voxels.append(vol)
    voxels = np.array(voxels)
    vol000 = np.array(vol000)
    vol001 = np.array(vol001)
    # vol000.reshape(len(vol001),4)
    vol001.reshape(len(vol001),1)
    return voxels


def load_ply_data(filename):
    '''
    load data from ply file.
    '''

    f = open(filename)

    #1.read all points
    points = []
    for line in f:
        #only x,y,z
        wordslist = line.split(' ')
        try:
          x, y, z = float(wordslist[0]),float(wordslist[1]),float(wordslist[2])
        except ValueError:
          continue
        points.append([x,y,z])
    points = np.array(points)
    points = points.astype(np.int32)#np.uint8
    # print(filename,'\n','length:',points.shape)
    f.close()

    return points


def write_ply_data_have_header(filename, points):
    '''
    write data to ply file.
    '''
    if os.path.exists(filename):
        os.system('rm '+filename)
    f = open(filename,'a+')
    #print('data.shape:',data.shape)
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(points.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.write('end_header\n')
    for _, point in enumerate(points):
        f.writelines([str(point[0]), ' ', str(point[1]), ' ',str(point[2]), '\n'])
    f.close()

    return


def load_points(filename, cube_size=64, min_num=20):
    """Load point cloud & split to cubes.

    Args: point cloud file; voxel size; minimun number of points in a cube.

    Return: cube positions & points in each cube.
    """

    # load point clouds
    # print(filename)

    point_cloud = load_ply_data(filename)
    # partition point cloud to cubes.
    cubes = {}  # {block start position, points in block}
    for _, point in enumerate(point_cloud):
        cube_index = tuple((point // cube_size).astype("int"))
        local_point = point % cube_size
        if not cube_index in cubes.keys():
            cubes[cube_index] = local_point
        else:
            cubes[cube_index] = np.vstack((cubes[cube_index], local_point))
    # filter by minimum number.
    k_del = []
    for _, k in enumerate(cubes.keys()):
        if cubes[k].shape[0] < min_num:
            k_del.append(k)
    for _, k in enumerate(k_del):
        del cubes[k]
    # get points and cube positions.
    cube_positions = np.array(list(cubes.keys()))
    set_points = []
    # orderd
    step = cube_positions.max() + 1
    cube_positions_n = cube_positions[:, 0:1] + cube_positions[:, 1:2] * step + cube_positions[:, 2:3] * step * step
    cube_positions_n = np.sort(cube_positions_n, axis=0)
    x = cube_positions_n % step
    y = (cube_positions_n // step) % step
    z = cube_positions_n // step // step
    cube_positions_orderd = np.concatenate((x, y, z), -1)
    for _, k in enumerate(cube_positions_orderd):
        set_points.append(cubes[tuple(k)].astype("int16"))

    return set_points, cube_positions


def preprocess_eval(input_file, scale, cube_size, min_num):
    """Scaling, Partition & Voxelization.
    Input: .ply file and arguments for pre-process.
    Output: partitioned cubes, cube positions, and number of points in each cube.
    """
    outfile_eval_divide_cube=open('./divide_cube_test_ori.txt','w')

    prefix=input_file.split('/')[-1].split('_')[0]+str(random.randint(1,100))
    print('===== Preprocess =====')
    # scaling (optional)
    start = time.time()
    print(input_file)
    if scale == 1:
        scaling_file = input_file
        # print(scaling_file)
        # print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
    else:
        pc = load_ply_data(input_file)
        pc_down = np.round(pc.astype('float32') * scale)
        pc_down = np.unique(pc_down, axis=0)# remove duplicated points
        scaling_file = prefix+'downscaling.ply'
        write_ply_data_have_header(scaling_file, pc_down)
    print("Scaling: {}s".format(round(time.time()-start, 4)))

    # partition.
    start = time.time()
    partitioned_points, cube_positions = load_points(scaling_file, cube_size, min_num)
    print("Partition: {}s".format(round(time.time()-start, 4)))
    # print(partitioned_points)
    # print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
    outfile_eval_divide_cube.write((str(partitioned_points)))

    if scale != 1:
        os.system("rm "+scaling_file)

    # voxelization.
    start = time.time()
    cubes = points2voxels(partitioned_points, cube_size)
    # # print(cubes)
    # points_numbers = np.sum(cubes, axis=(1,2,3,4)).astype(np.uint16)
    # print("Voxelization: {}s".format(round(time.time()-start, 4)))
    # print('cubes shape: {}'.format(cubes.shape))
    # print('points numbers (sum/mean/max/min): {} {} {} {}'.format(
    # points_numbers.sum(), round(points_numbers.mean()), points_numbers.max(), points_numbers.min()))
    # points_numbers = 0
    points_numbers = np.sum(cubes, axis=(1, 2, 3, 4)).astype(np.uint16)
    print("Voxelization: {}s".format(round(time.time() - start, 4)))
    feats =[]
    for i in partitioned_points:
        fl = np.ones((len(i), 1))
        feats.append(fl)
    coords_batch, feats_batch = ME.utils.sparse_collate(partitioned_points, feats)

    return cube_positions, points_numbers, partitioned_points, coords_batch, feats_batch


def save_points(set_points, cube_positions, i, cube_size=64):
    """Combine & save points."""

    step = cube_positions.max() + 1
    cube_positions_n = cube_positions[:, 0:1] + cube_positions[:, 1:2] * step + cube_positions[:, 2:3] * step * step
    cube_positions_n = np.sort(cube_positions_n, axis=0)
    x = cube_positions_n % step
    y = (cube_positions_n // step) % step
    z = cube_positions_n // step // step
    cube_positions_orderd = np.concatenate((x, y, z), -1)

    point_cloud = []
    for v in set_points:
        points = v + cube_positions_orderd[i, :] * cube_size
        point_cloud.append(points)
        point_cloud1 = np.array(point_cloud)
    return point_cloud1


class prune_voxel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pruning = ME.MinkowskiPruning()

    def forward(self, data, nums, rho=1.000):
        data_cls = ME.SparseTensor(features=data.F[:, :1],
                                   coordinate_map_key=data.coordinate_map_key,
                                   coordinate_manager=data.coordinate_manager,
                                   device=data.device)
        mask_topk = istopk_ori(data_cls, nums, rho=rho)

        data_pruned = self.pruning(data, mask_topk.to(data_cls.device))

        return data_pruned


