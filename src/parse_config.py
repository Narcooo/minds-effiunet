# -*- coding: utf-8 -*-
"""
@author Majx
@date 2022年10月25日 15:09:52

@describe TODO
"""
import importlib
import os
import sys
import config.helloconfig

def ma_get_config(path):
    # temp_module_name = os.path.splitext(path)[0]
    # cur_path = sys.argv[0]
    # absp = os.path.abspath('.')
    rel_base = os.path.relpath(path)
    temp_base_name = os.path.splitext(rel_base)[0]
    base_name = temp_base_name.replace('/','.')
    # abs = os.path.split(os.path.realpath(__file__))[0]
    base = importlib.import_module(base_name)
    base = base.base
    rel_dataset = base['dataset']
    temp_dataset_name = os.path.splitext(rel_dataset)[0]
    dataset_name = temp_dataset_name.replace('/', '.')
    dataset = importlib.import_module(dataset_name)
    data_root = dataset.data_root
    img_path = 'images_clear_clip_seg'
    msk_path = 'masks_clear_clip'
    img_suffix = 'tif'
    msk_suffix = 'png'
    imgsize = 1024
    batch_size = 4
    transform = dataset.transform
    a=1
