# model settings
model = dict(
    type='ImageClassifierCILWithMemory',
    backbone=dict(
        type='ResNet12',
        with_avgpool=False,
        flatten=False,
        norm_cfg=dict(type='SyncBN')
    ),
    neck=dict(type='MLPNeck', in_channels=640, out_channels=512),
    head=dict(
        type='ETFHeadCE',
        num_classes=100,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
        cal_acc=True,
    ),
    loss_dist=dict(
        type='DotLoss',
        loss_weight=2.,
    ),
    dist_mode='pre_logits',
)
