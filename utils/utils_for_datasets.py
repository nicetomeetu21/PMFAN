# -*- coding:utf-8 -*-
import cv2 as cv
import os
import torch
from natsort import natsorted
from torchvision import transforms


def read_cube(path):
    # print(path)
    imgs = []
    names = natsorted(os.listdir(path))
    for name in names:
        img = cv.imread(os.path.join(path, name), cv.IMREAD_COLOR)
        img = img[:,:,:1]
        # print(path, name, img.shape)
        img = transforms.ToTensor()(img)
        imgs.append(img)
    imgs = torch.stack(imgs, dim=1)
    return imgs

def check_interpolation(imgs, target_cube_size):
    if target_cube_size[-1] != imgs.shape[-1] or target_cube_size[-2] != imgs.shape[-2] or target_cube_size[-3] != \
            imgs.shape[-3]:
        imgs = imgs.unsqueeze(0)
        imgs = torch.nn.functional.interpolate(imgs, size=target_cube_size, mode='trilinear', align_corners=True)
        imgs = imgs.squeeze(0)
    return imgs

def augmentation_torch_multi(images):
    #horizontal flip
    if torch.randint(0, 2, (1,)).item()==0:
        for i in range(len(images)):
            images[i] = torch.flip(images[i], dims=(1,))
    #Vertical flip
    if torch.randint(0, 2, (1,)).item()==0:
        for i in range(len(images)):
            images[i] = torch.flip(images[i], dims=(3,))
    #rot90
    if torch.randint(0, 2, (1,)).item()==0:
        k=torch.randint(1, 4, (1,)).item()
        for i in range(len(images)):
            images[i] = torch.flip(images[i], dims=(3,))
            images[i]=torch.rot90(images[i],k,dims=(1,3))
    for i in range(len(images)):
        images[i] = images[i].detach()
    return images