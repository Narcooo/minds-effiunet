# -*- coding: utf-8 -*-
"""
@author Majx
@date 2022年10月25日 14:29:38

@describe TODO
"""
import albumentations as A

data_root = '/data'
img_path = 'images_clear_clip_seg'
msk_path = 'masks_clear_clip'
img_suffix = 'tif'
msk_suffix = 'png'
imgsize = 1024
batch_size = 4
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.1,
                               contrast_limit=0.1,
                               p=0.5),
    A.HueSaturationValue(hue_shift_limit=1,
                         sat_shift_limit=5,
                         val_shift_limit=5,
                         p=0.5,),
    A.ImageCompression(quality_lower=85,
                       quality_upper=95,
                       p=0.5,),
    A.MedianBlur(blur_limit=3, p=0.5),
])