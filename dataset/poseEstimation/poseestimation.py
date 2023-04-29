"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""
import os
import glob
import h5py
import numpy as np
import pickle
import logging
import ssl
import urllib
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive, check_integrity
from torchvision.transforms.functional import to_tensor

from ..build import DATASETS


def load_data(data_dir, partition):
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'seg_dataset_%s_0.h5' % partition)):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('<f8')
            label = f['label'][:].astype('uint8')
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


@DATASETS.register_module()
class PoseEstimation(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.
    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    dir_name = 'PoseEstimation'
    md5 = 'c9ab8e6dfb16f67afdab25e155c79e59'

    def __init__(self,
                 num_points=2048,
                 data_dir="./data",
                 split='train',
                 transform=None,
                 num_classes=6
                 ):
        data_dir = os.path.join(
            os.getcwd(), data_dir) if data_dir.startswith('.') else data_dir
        self.partition = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.data, self.label = load_data(data_dir, self.partition)
        self.num_points = num_points
        logging.info(f'==> sucessfully loaded {self.partition} data')
        self.transform = transform
        self.num_classes = num_classes

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item].squeeze(-1).astype(np.long)

        data = {'pos': pointcloud,
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = data['pos']
        return data

    def __len__(self):
        return self.data.shape[0]

    @property
    def num_classes(self):
        return self.num_classes

    """ for visulalization
    from openpoints.dataset import vis_multi_points
    import copy
    old_points = copy.deepcopy(data['pos'])
    if self.transform is not None:
        data = self.transform(data)
    new_points = copy.deepcopy(data['pos'])
    vis_multi_points([old_points, new_points.numpy()])
    End of visulization """