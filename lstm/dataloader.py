import glob
import os.path as osp
import numpy as np
import time

import torch
import torch.utils.data as data


def make_datapath_list(phase="train"):

    rootpath = "./data/"
    target_path = osp.join(rootpath+phase+'/**/*.csv')

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


class LoadDataset(data.Dataset):

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  # ファイルパスのリスト
        self.phase = phase  # train or valの指定

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        csv_path = self.file_list[index]
        #csv_dst = np.loadtxt( csv_path, delimiter=',', dtype='float32', skiprows=1 )
        csv_dst = np.genfromtxt( csv_path, delimiter=',', dtype='float32', skip_header=1, filling_values=(-1), max_rows=50 )
        csv_dst = torch.from_numpy( csv_dst )

        # extract label
        if self.phase == "train":
            label = csv_path[13:14]
            #print( f'labelname = {label}')
        elif self.phase == "val":
            label = csv_path[11:12]
            #print( f'labelname = {label}')
        elif self.phase == "test":
            label = csv_path[12:13]

        """
        # label2number
        if label == "ani":
            label = 0
        elif label == "exa":
            label = 1
        elif label == "hum":
            label = 2
        """
        # label2number
        if label == "A":
            label = 0
        elif label == "B":
            label = 1
        elif label == "C":
            label = 2
        elif label == "D":
            label = 3
        elif label == "E":
            label = 4
        elif label == "F":
            label = 5
        elif label == "G":
            label = 6

        return csv_dst, label


def main():

    train_list = make_datapath_list(phase="train")
    val_list = make_datapath_list(phase="val")

    """
    print(f'\n')
    print(f'train_list = {train_list}')
    print(f'\n')
    print(f'val_list = {train_list}')
    print(f'\n')
    """

    # 実行
    train_dataset = LoadDataset( file_list=train_list, phase='train' )
    val_dataset = LoadDataset( file_list=val_list, phase='val' )

    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=1, shuffle=True )
    val_loader = torch.utils.data.DataLoader( val_dataset, batch_size=1, shuffle=True )

    index = 0
    #print(train_dataset.__getitem__(index)[0].size())
    #print(train_dataset.__getitem__(index)[1])


if __name__ == '__main__':
    main()