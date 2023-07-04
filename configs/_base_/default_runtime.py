# checkpoint saving
checkpoint_config = dict(interval=5, max_keep_ckpts=2)
evaluation = dict(interval=5, start=0, broadcast_bn_buffer=False)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHookCIL'),
    ]
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]

load_from = None
resume_from = None
save_state = None

total_cls_num = 100
base_cls_num = 50
cls_num_step = 50

incremental_update = dict()
incremental_settings = dict(
    lr=None,
    use_lambda_lucir=False,
    vec_init=False,
    fix_backbone=False,
)
