import ml_collections

def get_config():

    config = ml_collections.ConfigDict()

    config.epochs = 50
    config.data_root = '/data'
    config.img_path = 'images'
    config.msk_path = 'masks'
    config.img_suffix = 'tif'
    config.msk_suffix = 'png'

    config.imgsize = 1024

    config.batch_size = 6

    config.pretrained_path = None

    config.in_channel = 3

    config.num_classes = 2

    config.lr = 0.001

    config.weights=[1,10]

    config.loss_weights=[1,3]
    config.lr_scheduler = 'cosine_annealing'

    # config.lr =
    # config.steps_per_epoch =
    config.warmup_epochs = 1
    config.max_epoch = config.epochs
    config.T_max = config.epochs
    config.eta_min = 0.
    config.num_worker = 1
    config.encoder = 'efficientnet-b3'
    config.resume_effiunet = None
    config.pretrained_backbone = None
    # config.pretrained_backbone = '/data/efficientnetb3_ascend_v170_imagenet2012_research_cv_top1acc80.37_top5acc95.17.ckpt'
    return config