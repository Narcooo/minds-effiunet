# -*- coding: utf-8 -*-
"""
@author Majx
@date 2022年10月20日 19:24:22

@describe TODO
"""

import warnings

import mindspore as ms
from mindspore.dataset import vision, transforms
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
import albumentations as A
from .transform import BaseTransform

"""
Create dataloader
"""


__all__ = ["create_loader"]


def create_loader(
        dataset,
        batch_size,
        drop_remainder=False,
        is_training=False,
        mixup=0.0,
        cutmix=0.0,
        cutmix_prob=0.0,
        num_classes=1000,
        config=None,
        device_num=1,
        num_parallel_workers=4,
        python_multiprocessing=True,
):
    cv2.setNumThreads(0)
    ds.config.set_enable_shared_mem(True)
    # import mindspore.dataset as ds
    dataset = ds.GeneratorDataset(dataset, dataset.column_names, num_parallel_workers=num_parallel_workers, max_rowsize=200,
                        python_multiprocessing=True)

    # if transform is None:
    #     warnings.warn("Using None as the default value of transform will set it back to "
    #                   "traditional image transform, which is not recommended. "
    #                   "You should explicitly call `create_transforms` and pass it to `create_loader`.")
    # onehot_op = transforms.OneHot(num_classes=2)
    #
    # dataset = dataset.map(operations=onehot_op, input_columns=["label"])
    trans = BaseTransform(config, device_num)
    dataset_column_names = ["image", "label"]
    # t_transform = []
    # t_transform += [transforms.TypeCast(ms.float32)]
    # t_transform += [vision.ToNumpy()]
    # t_transform += transform()
    # t_transform += [vision.ToTensor()]
    # dataset = dataset.map(operations=t_transform,
    #                       input_columns='image',
    #                       output_columns='decoded_images',
    #                       num_parallel_workers=num_parallel_workers,
    #                       python_multiprocessing=python_multiprocessing)
    # if target_transform is None:
    #     target_transform = []
    #     target_transform += [transforms.TypeCast(ms.float32)]
    #     target_transform += [vision.HWC2CHW()]
    # target_input_columns = 'label' if 'label' in dataset.get_col_names() else 'fine_label'
    # # one_hot = C.OneHot(num_classes=num_classes)
    # # dataset = dataset.map(input_columns="label", num_parallel_workers=num_parallel_workers,
    # #                       operations=one_hot)
    # dataset = dataset.map(operations=target_transform,
    #                       input_columns=target_input_columns,
    #                       num_parallel_workers=num_parallel_workers,
    #                       python_multiprocessing=python_multiprocessing)

    dst = dataset.batch(batch_size, per_batch_map=trans, input_columns=dataset_column_names,
                  num_parallel_workers=num_parallel_workers, drop_remainder=True, python_multiprocessing=True,max_rowsize=200)

    # dataset = dataset.batch(batch_size,
    #                         num_parallel_workers=num_parallel_workers, drop_remainder=True)


    # assert (mixup * cutmix == 0), 'Currently, mixup and cutmix cannot be applied together'

    # if is_training:
    #     trans_batch = []
    #     if mixup > 0:
    #         trans_batch = vision.MixUpBatch(alpha=mixup)
    #     elif cutmix > 0:
    #         trans_batch = vision.CutMixBatch(vision.ImageBatchFormat.NCHW, cutmix, cutmix_prob)
    #     if trans_batch != []:
    #         dataset = dataset.map(input_columns=["image", target_input_columns],
    #                               num_parallel_workers=num_parallel_workers, operations=trans_batch)

    return dst
