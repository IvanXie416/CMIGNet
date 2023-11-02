#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import numpy as np
import random
import torch
import pickle
import torchvision.transforms as tfs
import open3d
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from tqdm import tqdm
from PIL import Image

import common.se3 as se3
import common.so3 as so3

CM_DATA_DIR = "data/CCL_data"


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    # download()
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_DIR = "/home/dataset"
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_mv_data(root, partition):
    with open(os.path.join(root, 'modelnet40_%s_2048pts_20views.dat' % partition), 'rb') as f:
        all_pc, all_mv, all_label = pickle.load(f)
    print('load mv data')
    print('The size of %s data is %d' % (partition, len(all_pc)))
    return all_pc, all_mv, all_label


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05, seed=None):
    # if seed is not None:
    #     seed_everything(seed)
    N, C = pointcloud.shape
    pointcloud = pointcloud + np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    
    random_p2 = random_p1
    # random_p2 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


def random_Rt(angle, seed = None):
    # if seed is not None:
    #     seed_everything(seed)
    anglex = np.random.uniform() * angle
    angley = np.random.uniform() * angle
    anglez = np.random.uniform() * angle

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                    [0, cosx, -sinx],
                    [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                    [0, 1, 0],
                    [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                    [sinz, cosz, 0],
                    [0, 0, 1]])
    R = Rx.dot(Ry).dot(Rz)
    t = np.random.uniform(-0.50, 0.50, size=3)
    euler = np.array([anglez, angley, anglex])

    return R, t, euler


def random_Rt_real(angle, seed = None):
    # if seed is not None:
    #     seed_everything(seed)
    anglex = np.random.uniform() * angle
    angley = np.random.uniform() * angle
    anglez = np.random.uniform() * angle

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                    [0, cosx, -sinx],
                    [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                    [0, 1, 0],
                    [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                    [sinz, cosz, 0],
                    [0, 0, 1]])
    R = Rx.dot(Ry).dot(Rz)
    t = np.random.uniform(-0.05, 0.05, size=3)
    euler = np.array([anglez, angley, anglex])

    return R, t, euler


class ModelNet40(Dataset):
    def __init__(self, num_points=1024, partition='train', gaussian_noise=False, alpha=0.75, unseen=False, factor=4):
        super(ModelNet40, self).__init__()

        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.rot_factor = factor
        self.data, self.mv_data, self.label = load_mv_data(CM_DATA_DIR, partition)
        self.data = np.array(self.data)
        self.mv_data = np.array(self.mv_data)
        self.label = np.array(self.label)
        if self.unseen:         
            self.label = self.label.squeeze()
            if self.partition == 'test':
                self.data = self.data[self.label >= 20]
                self.label = self.label[self.label >= 20]
            elif self.partition == 'train':
                self.data = self.data[self.label < 20]
                self.label = self.label[self.label < 20]
            else:
                raise Exception('Invalid partition')
        else:
            self.label = self.label.squeeze()
        self.data = self.data[:, :self.num_points, 0:3]  # [?, 1024, 3]
        self.num_subsampled_points = int(self.data.shape[1]*alpha)

        self.transform_mv = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):

        _views = []
        views = self.mv_data[item]
        label = self.label[item]

        for i in range(20):
            _views.append(self.transform_mv(Image.fromarray(views[i], "RGB")))
        views = torch.stack(_views, 0) # torch.Size([20, 3, 224, 224])

        pointcloud1 = self.data[item].T
        R_ab, translation_ab, euler_ab = random_Rt(np.pi / self.rot_factor)
        pointcloud2 = np.matmul(R_ab, pointcloud1) + translation_ab[:, np.newaxis]  
        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                             num_subsampled_points=self.num_subsampled_points)


        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), euler_ab.astype('float32'), views

    def __len__(self):
        return self.data.shape[0]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True