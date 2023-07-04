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
        type='ETFHead',
        num_classes=100,
        in_channels=512,
        loss=dict(type='DRLoss', loss_weight=10.),
        topk=(1, 5),
        cal_acc=True,
    ),
    loss_dist=dict(
        type='DRLoss',
        loss_weight=25.,
    ),
    train_cfg=dict(augments=[
        dict(type='BatchMixupTwoLabel', alpha=0.8, num_classes=-1, prob=0.4),
        dict(type='BatchCutMixTwoLabel', alpha=1.0, num_classes=-1, prob=0.4),
        dict(type='IdentityTwoLabel', num_classes=-1, prob=0.2),
    ]),
)
