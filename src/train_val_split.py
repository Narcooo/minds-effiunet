# -*- coding: utf-8 -*-
"""
@author Majx
@date 2022年11月04日 15:29:46

@describe TODO
"""
import glob
import shutil,os
from tqdm import tqdm, trange
mmdir = '/data/'
# oridir = '/data/train_ori/images_clear_clip_seg/'
os.makedirs(mmdir + 'train/images',exist_ok=True)
os.makedirs(mmdir + 'train/masks',exist_ok=True)
os.makedirs(mmdir + 'val/images',exist_ok=True)
os.makedirs(mmdir + 'val/masks',exist_ok=True)
imgdir = glob.glob('/data/train_ori/images_clear_clip_seg/*.tif')
imgdir = sorted(imgdir, key = lambda name:int(name.split('/')[-1].split('.')[0]))
anndir = glob.glob('/data/train_ori/masks_clear_clip/*.png')
anndir = sorted(anndir, key = lambda name:int(name.split('/')[-1].split('.')[0]))

for x in trange(len(imgdir)):
    if x % 10 < 9:
        shutil.copy(imgdir[x],mmdir + 'train/images')
        shutil.copy(anndir[x],mmdir + 'train/masks')
    else:
        shutil.copy(imgdir[x],mmdir + 'val/images')
        shutil.copy(anndir[x],mmdir + 'val/masks')