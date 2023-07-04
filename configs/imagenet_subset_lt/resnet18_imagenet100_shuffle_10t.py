_base_ = [
    '../_base_/models/resnet_etf_dist_aug.py',
    '../_base_/datasets/imagenet100_lt_shuffle.py',
    '../_base_/schedules/imagenet100_200e.py',
    '../_base_/default_runtime.py'
]

total_cls_num = 100
base_cls_num = 50
cls_num_step = 5

checkpoint_config = dict(interval=50, max_keep_ckpts=2)
evaluation = dict(interval=50, start=0)

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=18,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    neck=dict(type='MLPBNNeck', in_channels=512, out_channels=512, norm_cfg=dict(type='SyncBN')),
    head=dict(
        type='ETFHeadInc',
        loss=dict(type='DRLoss', loss_weight=10.),
        with_len=False,
    ),
    dist_mode='pre_logits_old',
    loss_dist=dict(type='DRLoss', loss_weight=50.),
)

incremental_settings = dict(lr=5., use_lambda_lucir=True, vec_init=True)
incremental_update = dict(
    optimizer=dict(
        type='SGD',
        lr=0.25,
        momentum=0.9,
        weight_decay=0.0005,
        paramwise_cfg=dict(
            custom_keys={
                'backbone': dict(lr_mult=0.01, decay_mult=0.01),
            },
            norm_decay_mult=1.
        )
    ),
    optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=3.0)),
    log_config=dict(
        interval=4,
    ),
)
optimizer = dict(
    type='SGD', lr=0.25, momentum=0.9, weight_decay=0.0005
)
