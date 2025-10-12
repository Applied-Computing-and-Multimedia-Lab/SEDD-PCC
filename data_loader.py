import numpy as np
import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler
import MinkowskiEngine as ME
from data_utils import  read_h5_attri, read_ply_ascii_attri

class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)


def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1
    list_data = new_list_data
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    coords, feats = list(zip(*list_data))
    coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats)

    return coords_batch, feats_batch


class PCDataset(torch.utils.data.Dataset):

    def __init__(self, files):
        self.files = []
        self.cache = {}
        self.last_cache_percent = 0
        self.files = files

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):
        filedir = self.files[idx]

        if idx in self.cache:
            coords, new_feats = self.cache[idx]
        else:
            if filedir.endswith('.h5'): coords, features = read_h5_attri(filedir)
            if filedir.endswith('.ply'): coords, features = read_ply_ascii_attri(filedir)
            feats = np.expand_dims(np.ones(coords.shape[0]), 1).astype('int')
            attri = features
            new_feats = np.concatenate((feats, attri), axis=1)
            # cache
            self.cache[idx] = (coords, new_feats)
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                self.last_cache_percent = cache_percent
        new_feats = new_feats.astype("float32")


        return (coords, new_feats)


def make_data_loader(dataset, batch_size=1, shuffle=False, num_workers=0, repeat=False, 
                    collate_fn=collate_pointcloud_fn, pin_memory=True):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': False,
        'drop_last': False
    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader
def collate_pointcloud_fn_eval(list_data):
    feats = []
    for i in list_data:
        fl = np.ones((len(i), 1))
        feats.append(fl)
    coords_batch, feats_batch = ME.utils.sparse_collate(list_data, feats)

    return coords_batch, feats_batch


def make_data_loader_eval(dataset, batch_size=1, shuffle=False, num_workers=1, repeat=False,
                    collate_fn=collate_pointcloud_fn_eval):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': False,
        'drop_last': False

    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader


