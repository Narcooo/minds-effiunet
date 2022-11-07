# # -*- coding: utf-8 -*-
# """
# @author Majx
# @date 2022年10月30日 15:11:44
#
# @describe TODO
# """
import mindspore.nn as nn
import mindspore.nn as F
from mindspore.ops import operations as P
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore import dtype as mstype
import numpy as np
import mindspore as ms


class BCEDiceLossWithWeights(nn.LossBase):
    def __init__(self, weights, num_classes=1, ignore_label=255,loss_weights=None, cfg=None):
        super(BCEDiceLossWithWeights, self).__init__()
        self.weights = weights
        # self.resize = F.ResizeBilinear(cfg.train.image_size)
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.focal = nn.FocalLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = nn.DiceLoss()
        self.zeros = ops.Zeros()
        self.fill = ops.Fill()
        self.equal = ops.Equal()
        self.select = ops.Select()
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.argmax = P.Argmax(output_type=mstype.int32)
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.activation = ops.Sigmoid()
        if loss_weights is not None:
            self.loss_weights = loss_weights

    def construct(self, logits, labels):

        # logits_ = logits
        # logits = self.activation(logits)
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))  # (12, 1024, 2048, 19)
        logits_ = self.reshape(logits_, (-1, self.num_classes))
        # li = self.reshape(labels_int, (-1,))
        labels_float = self.cast(labels_int, mstype.float32)
        weights = self.zeros(labels_float.shape, mstype.float32)
        for i in range(0, self.num_classes):
            fill_weight = self.fill(mstype.float32, labels_float.shape, self.weights[i])
            equal_ = self.equal(labels_float, i)
            weights = self.select(equal_, fill_weight, weights)

            # return [fill_weight, equal_, weights]
        one_hot_labels = self.one_hot(labels_int, self.num_classes, self.on_value, self.off_value)
        # loss = self.ce(logits_, one_hot_labels)
        # loss = self.mul(weights, loss)
        # loss = self.div(self.sum(loss), weights.size)
        # print(logits_.shape)
        loss_dice = self.dice(logits_, one_hot_labels)

        loss_dice = self.mul(weights, loss_dice)
        loss_dice = self.div(self.sum(loss_dice), weights.size)
        # loss_focal = self.focal(logits_, labels_float)
        #
        # loss_focal = self.mul(weights, loss_focal)
        # loss = self.sum(loss)
        loss_bce = self.bce(logits_, one_hot_labels)
        loss_bce = self.mul(weights, loss_bce)
        loss_bce= self.div(self.sum(loss_bce), weights.size)
        # loss_focal = self.div(self.sum(loss_focal), weights.size)
        # ldn, idx, counts = loss_dice.unique_consecutive(return_idx=True, return_counts=True, axis=None)
        # lfn, idx, counts = loss_focal.unique_consecutive(return_idx=True, return_counts=True, axis=None)
        output = self.loss_weights[0] * loss_dice + self.loss_weights[1] * loss_bce
        return output

if __name__ == '__main__':
    # i = Tensor(np.zeros([1, 2, 1312, 2080]), ms.float32)
    i = Tensor(np.random.random((1, 2,1312 ,2080)), ms.float32)
    m = Tensor(np.random.randint(low=0,high=2,size=(1, 1, 1312 ,2080),dtype='int'), ms.float32)
    # m = Tensor(np.ones([1, 1, 1312, 2080]), ms.float32)
    model = BCEDiceLossWithWeights([1, 3], num_classes=2,loss_weights=[3, 10])
    a = model(i, m)
    print(a)