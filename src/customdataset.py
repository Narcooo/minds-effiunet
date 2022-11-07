# -*- coding: utf-8 -*-
"""
@author Majx
@date 2022年10月20日 17:28:43

@describe TODO
"""


import os
import cv2
import mindspore.dataset as ds
import glob
import mindspore.dataset.vision as vision_C
import mindspore.dataset.transforms as C_transforms
import random
import mindspore
from PIL import Image
import numpy as np
from mindspore.dataset.vision import Inter


class CustomDataset:
    def __init__(self,
                 data_root,
                 split,
                 img_path,
                 msk_path,
                 img_suffix,
                 msk_suffix,
                 key=lambda name:int(name.split('/')[-1].split('.')[0].split('_')[1])):

        self.data_root = data_root
        self.data_path = os.path.join(data_root, split)
        self.img = os.path.join(self.data_path, img_path)
        self.msk = os.path.join(self.data_path,msk_path)
        # a = os.path.join(self.img, '*.' + img_suffix)
        self.img_list = glob.glob(os.path.join(self.img, '*.' + img_suffix))
        self.msk_list = glob.glob(os.path.join(self.msk, '*.' + msk_suffix))
        # self.img_list = os.listdir(self.img)
        self.img_list.sort(key=key)
        self.msk_list.sort(key=key)
        # b = self.img_list[0].split('/')[-1].split('.')[0].split('_')[1]
        index = 0
        for i in range(len(self.img_list)):
            if self.img_list[i].split('/')[-1].split('.')[0] != self.msk_list[i].split('/')[-1].split('.')[0]:
                index+=1
        assert (index == 0) , "sorted img and mask not right"


    def __getitem__(self, index):

        image = cv2.imread(self.img_list[index])
        label = cv2.imread(self.msk_list[index], cv2.IMREAD_GRAYSCALE)
        # print(label.shape)
        label = label.reshape((label.shape[0], label.shape[1], 1))
        # print(label.shape)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # suimg = self.img_list[index].split('/')[-1].split('.')[0]
        # sumsk = self.msk_list[index].split('/')[-1].split('.')[0]
        # label[label==1]=255
        # os.makedirs('/data/debugimg',exist_ok=True)
        #
        # os.makedirs('/data/debugmask', exist_ok=True)
        # cv2.imwrite(f'/data/debugimg/{suimg}.jpg', image)
        # cv2.imwrite(f'/data/debugmask/{sumsk}.jpg', label)
        # print(1)
        return image, label

    @property
    def column_names(self):
        column_names = ['image', 'label']
        return column_names

    def __len__(self):

        return len(self.img_list)


def create_dataset(data_root,
                   split,
                   img_path,
                   msk_path,
                   img_suffix,
                   msk_suffix,
                   config,

                   num_parallel_workers=4,
                   key=lambda name:int(name.split('/')[-1].split('.')[0])):
    # cv2.setNumThreads(0)
    mc_dataset = CustomDataset(data_root=data_root,img_path=img_path,
                             msk_path=msk_path,
                             img_suffix=img_suffix,
                             msk_suffix=msk_suffix,
                             split=split,
                             key=key)
    config.dataset_size = len(mc_dataset)
    dataset = mc_dataset
    return dataset



if __name__ == '__main__':

    train_dataset = create_dataset('/data',
                                   split='train',
                                   img_path='images_clear_clip_seg',
                                   msk_path='masks_clear_clip',
                                   img_suffix = 'tif',
                                   msk_suffix = 'png')
    for item, (image, label) in enumerate(train_dataset):
        if item < 5:
            print(label.shape)

        # img = image.asnumpy()[3].transpose(1,2,0)  #.squeeze(0)
        # cv2.imshow("image", img)