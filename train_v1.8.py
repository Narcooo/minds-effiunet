import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import time
from src.metric import metrics_
from src.unet import UNet
import mindspore.nn as nn
from src.customdataset import create_dataset
from src.configs import get_config
from mindspore import ops
import mindspore
from mindspore import ms_function
from src.dataloader import create_loader
from mindspore.common.tensor import Tensor
import argparse
from src.effiunet import EfficientUnet
import mindspore as ms
from src.losswithweight import BCEDiceLossWithWeights
from src.lr_scheduler import get_lr
# from mindsvision.engine.callback import LossMonitor
# from mindspore.common.tensor import Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from src.initializer import default_recurisive_init, load_effiunet_params
cfg = get_config()
train_dataset = create_dataset(data_root=cfg.data_root,
                               split='train',
                               img_path=cfg.img_path,
                               msk_path=cfg.msk_path,
                               img_suffix=cfg.img_suffix,
                               msk_suffix=cfg.msk_suffix,
                                num_parallel_workers=cfg.num_worker,

                               config=cfg)
val_dataset = create_dataset(data_root=cfg.data_root,
                               split='val',
                               img_path=cfg.img_path,
                               msk_path=cfg.msk_path,
                               img_suffix=cfg.img_suffix,
                               msk_suffix=cfg.msk_suffix,
                             num_parallel_workers=cfg.num_worker,
                             config=cfg)

# cfg.dataset_size = train_dataset.source_len
# train_dataset.dataset_size = cfg.dataset_size // cfg.batch_size
train_dataset = create_loader(dataset=train_dataset,
                             config=cfg,
                                batch_size=cfg.batch_size,
                             num_parallel_workers=cfg.num_worker,
                                num_classes=cfg.num_classes,)
# a=0
# for batch in train_dataset.create_dict_iterator(num_epochs=1):
#     a+=1
#     print(a)
#     print("time: ", time.time())
val_dataset = create_loader(dataset=val_dataset,
                             config=cfg,
                                batch_size=cfg.batch_size,
                             num_parallel_workers=cfg.num_worker,
                                num_classes=cfg.num_classes,)
# train_dataset = ms.DatasetHelper(train_dataset, dataset_sink_mode=False)
# val_dataset = ms.DatasetHelper(val_dataset, dataset_sink_mode=False)
activation = ops.Sigmoid()
squeeze = ops.Squeeze(1)


# for data, label in enumerate(train_dataset):
#     print(label[1])
def train(model, dataset, loss_fn, optimizer, met):

    # Define forward function
    def forward_fn(data, label):
        logits = model(data)
        # print(logits.shape)
        loss = loss_fn(logits, label)
        return loss, logits
    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    # Define function of one-step training
    @ms_function
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss, logits

    size = dataset.get_dataset_size()
    model.set_train(True)
    train_loss = 0
    train_pred = []
    train_label = []
    num = 0
    call = 50
    metric = metrics_(met, smooth=1e-5)
    # dataloader = dataset.create_dict_iterator(output_numpy=True, num_epochs=1)
    for batch, datadict in enumerate(train_dataset):
        data = datadict[0]
        label = datadict[1]
        # data = Tensor.from_numpy(data)
        #
        # label = Tensor.from_numpy(label)
        # one_hot = P.OneHot(axis=-1)
        label_s = squeeze(label)
        loss, logits = train_step(data, label)
        logits_act = activation(logits)
        logits_r = ops.Argmax(axis=1, output_type=mindspore.int32)(logits_act)
        train_loss += loss.asnumpy()
        # cast = P.Cast()
        # label_int = cast(label, mstype.int32)
        # label_onehot = one_hot(label_int, cfg.num_classes, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32))
        metric.update(logits_r.asnumpy(), label_s.asnumpy())

        num+=1
        if num % call == 0:
            print(f'iter:{num}/{size}',
                  f'train loss:{loss.asnumpy():>4f}',
                  f'lr:{lr[iters_per_epoch * epoch + batch * cfg.batch_size]}'
                  )
    train_loss /= size

    res = metric.eval()
    metric.clear()

    print(f'Train loss:{train_loss:>4f}','丨acc: %.3f丨丨iou: %.3f丨丨dice: %.3f丨丨sens: %.3f丨丨spec: %.3f丨' % (res[0], res[1], res[2], res[3], res[4]))


def val(model, dataset, loss_fn, met):
    size = dataset.get_dataset_size()
    model.set_train(False)
    val_loss = 0
    val_pred = []
    val_label = []
    metric = metrics_(met, smooth=1e-5)
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        label_s = squeeze(label)
        pred = model(data)
        pred_act = activation(pred)
        pred_r = ops.Argmax(axis=1, output_type=mindspore.int32)(pred_act)
        # train_loss += loss.asnumpy()
        val_loss += loss_fn(pred, label).asnumpy()
        # val_pred.extend(pred.asnumpy())
        # val_label.extend(label.asnumpy())
        metric.update(pred_r.asnumpy(), label_s.asnumpy())

    val_loss /= size


    # metric.update(val_pred, val_label)
    res = metric.eval()
    metric.clear()

    print(f'Val loss:{val_loss:>4f}','丨acc: %.3f丨丨iou: %.3f丨丨dice: %.3f丨丨sens: %.3f丨丨spec: %.3f丨' % (res[0], res[1], res[2], res[3], res[4]))

    checkpoint = res[1]
    return checkpoint

if __name__ == '__main__':
    # net = UNet(cfg.in_channel, cfg.num_classes)
    net = EfficientUnet(encoder=cfg.encoder, num_classes=cfg.num_classes)
    default_recurisive_init(net)
    # param_dict = mindspore.load_checkpoint(r'/data/efficientnetb3_ascend_v170_imagenet2012_research_cv_top1acc80.37_top5acc95.17.ckpt')
    # mindspore.load_param_into_net(net, param_dict)
    load_effiunet_params(cfg, net)
    # if hasattr(train_dataset, 'iter'):
    #     train_dataset = train_dataset.iter.dataset
    # if hasattr(val_dataset, 'iter'):
    #     val_dataset = val_dataset.iter.dataset
    criterion = BCEDiceLossWithWeights(weights=cfg.weights,loss_weights=cfg.loss_weights, num_classes=cfg.num_classes)
    # criterion = nn.BCEWithLogitsLoss()
    learning_rate = 0.1  # 学习率的初始值
    decay_rate = 0.9  # 衰减率
    decay_steps = cfg.epochs


    iters_per_epoch = train_dataset.get_dataset_size()

    total_train_steps = iters_per_epoch * cfg.epochs
    print('iters_per_epoch: ', iters_per_epoch)
    print('total_train_steps: ', total_train_steps)
    cfg.steps_per_epoch = iters_per_epoch
    lr = get_lr(cfg)

    optimizer = nn.Adam(params=net.trainable_params(),learning_rate=lr)

    metrics_name = ["acc", "iou", "dice", "sens", "spec"]

    best_iou = 0
    os.makedirs('checkpoint', exist_ok=True)
    ckpt_path = 'checkpoint/best_model.ckpt'
    for epoch in range(cfg.epochs):
        print(f"Epoch [{epoch+1} / {cfg.epochs}]")
        # model = ms.Model(net, loss_fn=criterion, optimizer=optimizer, metrics=None,)
        # model.train(epoch=cfg.epochs, train_dataset=train_dataset, dataset_sink_mode=False)

        train(net, train_dataset, criterion, optimizer, metrics_name)
        checkpoint_best = val(net, val_dataset, criterion, metrics_name)

        # train(net, train_dataset, criterion, optimizer, metrics_name)
        # checkpoint_best = val(net, val_dataset, criterion, metrics_name)
        if checkpoint_best > best_iou:
            print('IoU improved from %0.4f to %0.4f' % (best_iou, checkpoint_best))
            best_iou = checkpoint_best
            mindspore.save_checkpoint(net, ckpt_path)
            print("saving best checkpoint at: {} ".format(ckpt_path))
        else:
            print('IoU did not improve from %0.4f' % (best_iou),"\n-------------------------------")
    print("Done!")
