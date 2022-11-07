# -*- coding: utf-8 -*-
"""
@author Majx
@date 2022年10月25日 13:43:35

@describe TODO
"""
import os

from PIL import Image
import cv2

import albumentations as A
import random
import numpy as np
trans_image = A.Compose([
    A.OneOf(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.),
            A.RandomGamma(gamma_limit=(70, 130), p=1),
            A.ChannelShuffle(p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=20, p=1),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1),
        ],
        p=0.8,
    ),
    A.OneOf(
        [
            A.GaussNoise(p=1),
            A.MultiplicativeNoise(p=1),
            A.GaussianBlur(p=1),
        ],
        p=0.2,
    ),
])
trans_both = A.Compose([
    A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, border_mode=0, value=0, p=0.7),
    A.PadIfNeeded(1024, 1024, border_mode=0, value=0, p=1.),
    A.Flip(p=0.5),
    A.RandomCrop(1024, 1024, p=1.),

])
trans_test = []

def _rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a
def _data_aug(img, mask,):

    transformed =trans_both(image=img,mask=mask)
    image = transformed["image"]
    mask = transformed["mask"]
    transformed_img = trans_image(image=image)
    image = transformed_img["image"]
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = np.array(image)
    image_data = image.astype(np.float32)
    mask_data = mask.astype(np.float32)
    # os.makedirs('/data/debugimg',exist_ok=True)
    # mask_data[mask_data==1]=255
    # os.makedirs('/data/debugmask', exist_ok=True)
    # cv2.imwrite(f'/data/debugimg/img_{n}.jpg',image_data)
    # cv2.imwrite(f'/data/debugmask/mask_{n}.jpg', mask_data)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_data, mask_data

class BaseTransform:
    """transform."""
    def __init__(self, config, device_num):
        self.config = config
        self.seed = 0
        self.device_num = device_num
        self.num_classes = config.num_classes
        self.dataset_size = config.dataset_size // config.batch_size

    def __call__(self, imgs, masks, batchInfo):
        ret_imgs = []
        ret_masks = []

        # print(f'inputsize:{input_size}')
        for img, mask in zip(imgs, masks):
            # print(img,anno)
            img, mask = _data_aug(img, mask)
            # print(f'imageshape,{img.shape}，box_data:{anno}')
            ret_imgs.append(img.transpose(2, 0, 1).copy())
            ret_masks.append(mask.transpose(2, 0, 1).copy())
        return np.array(ret_imgs), np.array(ret_masks)