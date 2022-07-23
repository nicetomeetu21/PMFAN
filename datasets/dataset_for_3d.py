# -*- coding:utf-8 -*-
import os
from torch.utils.data.dataset import Dataset
from natsort import natsorted
from utils.utils_for_datasets import check_interpolation, read_cube, augmentation_torch_multi

class Opt3D_multi_cat_Dataset(Dataset):
    def __init__(self, data_roots, cube_names, with_path = False, with_aug = False, cube_size=(400,640,400), total_cube=False, cube_num=10):
        self.data_name_lists = []
        self.data_path_lists = []
        for data_root in data_roots:
            assert os.path.exists(data_root)
            if total_cube:
                cube_names = natsorted(os.listdir(data_root))
            elif len(cube_names)==0:
                cube_names = natsorted(os.listdir(data_root))[:cube_num]
            # print(data_root, cube_names)
            cube_pathes = [os.path.join(data_root, name) for name in cube_names]
            self.data_name_lists.append(cube_names)
            self.data_path_lists.append(cube_pathes)
        self.cube_size = cube_size
        self.with_path = with_path
        self.with_aug = with_aug
        print(data_roots, self.__len__())
    def __getitem__(self, index):

        imgs_list = []
        for cube_paths in self.data_path_lists:
            imgs = read_cube(cube_paths[index])
            imgs = check_interpolation(imgs, self.cube_size)
            imgs_list.append(imgs)

        if self.with_aug:
            imgs_list = augmentation_torch_multi(imgs_list)

        if self.with_path:
            return imgs_list, self.data_name_lists[0][index]
        else:
            return imgs_list
    def __len__(self):
        return len(self.data_name_lists[0])