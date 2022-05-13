# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class ReadDataset(Dataset):
    """docstring for ReadDataset"""
    def __init__(self, root, classname='normal', transform=None):
        super(ReadDataset, self).__init__()
        self.root = root
        self.dataset_list = list()

        if classname == 'normal':
            normal_file = os.path.join(root, 'normal_data.npy')
            normal_dataset = np.load(normal_file)
            for i in range(normal_dataset.shape[0]):
                normal_data = normal_dataset[i]
                self.dataset_list.append(normal_data)

        if classname == 'fault':
            fault_file = os.path.join(root, 'data101.npy')
            fault_dataset = np.load(fault_file)
            for i in range(fault_dataset.shape[0]):
                fault_data = fault_dataset[i]
                self.dataset_list.append(fault_data)

    def __getitem__(self, index):
        """description_method """
        data = self.dataset_list[index]
        data = np.array([data]) / 4000.0

        return data

    def __len__(self):
        return len(self.dataset_list)





if __name__ == '__main__':

    # file_path = "./dataset/00_2015.xlsx"

    # root = "./dataset"

    # dataset = ReadDataset(root=root)


    import numpy as np
    import matplotlib.pyplot as plt
    dataset_root = "./dataset_npy"
    dataset_file = os.path.join(dataset_root, 'faultdata.npy')
    dataset = np.load(dataset_file)
    print(dataset)

    save_images_dir = "./dataset_images/fault"
    os.makedirs(save_images_dir, exist_ok=True)
    for i in range(dataset.shape[0]):
        data = dataset[i].tolist()
        print(data)
        plt.plot(data)
        # plt.show()
        file_path = os.path.join(save_images_dir, '%03d.jpg'%i)
        plt.savefig(file_path)
        plt.close()


