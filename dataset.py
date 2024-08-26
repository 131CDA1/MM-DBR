import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import os

class MMDBR_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.modal = modal
        self.transform = transform
        self.data_list = glob.glob(root_dir + '\\*\\*.mat')
        self.folder = glob.glob(root_dir + '\\*\\')
        self.category = {self.folder[i].split('\\')[-2]: i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('\\')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]

        # 新增，3根天线对应的30个子载波一一相加，
        # x_1 = np.zeros((2000,30))
        # for i in range(30):
        #     indices = [j for j in range(i, 90, 30)]
        #     x_1[:,i] = np.sum(x[:, indices], axis=1)

        # normalize
        # u2到u7的数据集：均值mean=2.6466(7.9397)(2.4604), 标准差std=2.4737(2.8472)(2.1104)
        x = (x-2.4604)/2.1104
        # x: (2000,90)转置-->(90,2000) sampling--->(90,500)-->(3,30,500)
        # x = np.transpose(x)
        x = x[::8, :]
        x = x.reshape(1, 250, 90)

        x = torch.FloatTensor(x)

        # # normalize
        # x = (x - 42.3199) / 4.9802
        #
        # # sampling: 2000 -> 500
        # x = x[:, ::4]
        # x = x.reshape(3, 114, 500)
        #
        # if self.transform:
        #     x = self.transform(x)
        #
        # x = torch.FloatTensor(x)

        return x, y
