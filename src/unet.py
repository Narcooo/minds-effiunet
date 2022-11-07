import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from mindspore import nn
import mindspore.numpy as np
import mindspore.ops as ops
import mindspore as ms
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor

def double_conv(in_ch, out_ch):
    return nn.SequentialCell(nn.Conv2d(in_ch, out_ch, 3),
                              nn.BatchNorm2d(out_ch), nn.ReLU(),
                              nn.Conv2d(out_ch, out_ch, 3),
                              nn.BatchNorm2d(out_ch), nn.ReLU())
class UNet(nn.Cell):
    def __init__(self, in_ch = 3, n_classes = 1):
        super(UNet, self).__init__()
        self.concat1 = P.Concat(axis=1)
        self.concat2 = P.Concat(axis=1)
        self.concat3 = P.Concat(axis=1)
        self.concat4 = P.Concat(axis=1)
        self.double_conv1 = double_conv(in_ch, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv2 = double_conv(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv3 = double_conv(128, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv4 = double_conv(256, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv5 = double_conv(512, 1024)

        self.upsample1 = nn.ResizeBilinear()
        self.double_conv6 = double_conv(1024 + 512, 512)
        self.upsample2 = nn.ResizeBilinear()
        self.double_conv7 = double_conv(512 + 256, 256)
        self.upsample3 = nn.ResizeBilinear()
        self.double_conv8 = double_conv(256 + 128, 128)
        self.upsample4 = nn.ResizeBilinear()
        self.double_conv9 = double_conv(128 + 64, 64)

        self.final = nn.Conv2d(64, n_classes, 1)
        # self.sigmoid = ops.Sigmoid()

    def construct(self, x):

        feature1 = self.double_conv1(x)
        tmp = self.maxpool1(feature1)
        feature2 = self.double_conv2(tmp)
        tmp = self.maxpool2(feature2)
        feature3 = self.double_conv3(tmp)
        tmp = self.maxpool3(feature3)
        feature4 = self.double_conv4(tmp)
        tmp = self.maxpool4(feature4)
        feature5 = self.double_conv5(tmp)

        up_feature1 = self.upsample1(feature5, scale_factor=2)
        tmp = self.concat1((feature4, up_feature1))
        tmp = self.double_conv6(tmp)
        up_feature2 = self.upsample2(tmp, scale_factor=2)
        tmp = self.concat2((feature3, up_feature2))
        tmp = self.double_conv7(tmp)
        up_feature3 = self.upsample3(tmp, scale_factor=2)
        tmp = self.concat3((feature2, up_feature3))
        tmp = self.double_conv8(tmp)
        up_feature4 = self.upsample4(tmp, scale_factor=2)
        tmp = self.concat4((feature1, up_feature4))
        tmp = self.double_conv9(tmp)
        output = self.final(tmp)

        return output


class SegLossBlock(nn.Cell):
    """
    Loss block cell of YOLOV4 network.
    """

    def __init__(self, scale):
        super(SegLossBlock, self).__init__()

        self.anchors = Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.ignore_threshold = Tensor(self.config.ignore_threshold, ms.float32)
        self.concat = P.Concat(axis=-1)
        self.iou = Iou()
        self.reduce_max = P.ReduceMax(keep_dims=False)
        self.xy_loss = XYLoss()
        self.wh_loss = WHLoss()
        self.confidenceLoss = ConfidenceLoss()
        self.classLoss = ClassLoss()

        self.reduce_sum = P.ReduceSum()
        self.giou = Giou()

    def construct(self, grid, prediction, pred_xy, pred_wh, y_true, gt_box, input_shape):
        # prediction : origin output from yolo
        # pred_xy: (sigmoid(xy)+grid)/grid_size
        # pred_wh: (exp(wh)*anchors)/input_shape
        # y_true : after normalize
        # gt_box: [batch, maxboxes, xyhw] after normalize

        object_mask = y_true[:, :, :, :, 4:5]
        class_probs = y_true[:, :, :, :, 5:]
        true_boxes = y_true[:, :, :, :, :4]

        grid_shape = P.Shape()(prediction)[1:3]
        grid_shape = P.Cast()(F.tuple_to_array(grid_shape[::-1]), ms.float32)

        pred_boxes = self.concat((pred_xy, pred_wh))
        true_xy = y_true[:, :, :, :, :2] * grid_shape - grid
        true_wh = y_true[:, :, :, :, 2:4]
        true_wh = P.Select()(P.Equal()(true_wh, 0.0),
                             P.Fill()(P.DType()(true_wh),
                                      P.Shape()(true_wh), 1.0),
                             true_wh)
        true_wh = P.Log()(true_wh / self.anchors * input_shape)
        # 2-w*h for large picture, use small scale, since small obj need more precise
        box_loss_scale = 2 - y_true[:, :, :, :, 2:3] * y_true[:, :, :, :, 3:4]

        gt_shape = P.Shape()(gt_box)
        gt_box = P.Reshape()(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))

        # add one more dimension for broadcast
        iou = self.iou(P.ExpandDims()(pred_boxes, -2), gt_box)
        # gt_box is x,y,h,w after normalize
        # [batch, grid[0], grid[1], num_anchor, num_gt]
        best_iou = self.reduce_max(iou, -1)
        # [batch, grid[0], grid[1], num_anchor]

        # ignore_mask IOU too small
        ignore_mask = best_iou < self.ignore_threshold
        ignore_mask = P.Cast()(ignore_mask, ms.float32)
        ignore_mask = P.ExpandDims()(ignore_mask, -1)
        # ignore_mask backpro will cause a lot maximunGrad and minimumGrad time consume.
        # so we turn off its gradient
        ignore_mask = F.stop_gradient(ignore_mask)

        confidence_loss = self.confidenceLoss(object_mask, prediction[:, :, :, :, 4:5], ignore_mask)
        class_loss = self.classLoss(object_mask, prediction[:, :, :, :, 5:], class_probs)

        object_mask_me = P.Reshape()(object_mask, (-1, 1))  # [8, 72, 72, 3, 1]
        box_loss_scale_me = P.Reshape()(box_loss_scale, (-1, 1))
        pred_boxes_me = xywh2x1y1x2y2(pred_boxes)
        pred_boxes_me = P.Reshape()(pred_boxes_me, (-1, 4))
        true_boxes_me = xywh2x1y1x2y2(true_boxes)
        true_boxes_me = P.Reshape()(true_boxes_me, (-1, 4))
        ciou = self.giou(pred_boxes_me, true_boxes_me)
        ciou_loss = object_mask_me * box_loss_scale_me * (1 - ciou)
        ciou_loss_me = self.reduce_sum(ciou_loss, ())
        loss = ciou_loss_me + confidence_loss + class_loss
        batch_size = P.Shape()(prediction)[0]

        # print(f'loss/batchsize:{loss/batch_size}')

        return loss / batch_size

class SegWithLossCell(nn.Cell):
    """YOLOV4 loss."""
    def __init__(self, network):
        super(SegWithLossCell, self).__init__()
        self.yolo_network = network
        self.config = ConfigYOLOV4CspDarkNet53()
        self.loss_big = YoloLossBlock('l', self.config)
        self.loss_me = YoloLossBlock('m', self.config)
        self.loss_small = YoloLossBlock('s', self.config)

    def construct(self, x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape):
        yolo_out = self.yolo_network(x, input_shape)
        loss_l = self.loss_big(*yolo_out[0], y_true_0, gt_0, input_shape)
        loss_m = self.loss_me(*yolo_out[1], y_true_1, gt_1, input_shape)
        loss_s = self.loss_small(*yolo_out[2], y_true_2, gt_2, input_shape)
        # print(f'loss_l:{loss_l},loss_m:{loss_m},loss_s:{loss_s}')
        return (loss_l + loss_m + loss_s) / 4

if __name__ == '__main__':
    t = Tensor(np.ones([1, 3, 1312, 2080]), ms.float32)
    model = UNet(n_classes=2)
    a = model(t)
    print(a)