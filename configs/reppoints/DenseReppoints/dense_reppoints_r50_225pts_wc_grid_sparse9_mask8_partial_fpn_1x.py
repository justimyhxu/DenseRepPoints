# model settings
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='DenseRepPointsMaskDetector',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        with_cp=False,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='DenseRepPointsMaskHead',
        num_classes=81,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        stacked_mask_convs=7,
        num_group=9,
        use_sparse_pts_for_cls=True,
        center_init=False,
        num_points=225,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        loss_mask_score_init=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_mask_weight_init=0.5,
        loss_mask_weight_refine=1.0,
        use_chamfer=True,
        # feat_or_cor=True,
        # ps_offset_field=True,
        # share_score=True,
        # en_4conv=False,
        # en_b4conv=False,
        # en_mask_feat=True,
        transform_method='partial_minmax'))
# training and testing settings
train_cfg = dict(
    init=dict(
        assigner=dict(type='PointAssigner', scale=4, pos_num=1),
        allowed_border=-1,
        pos_weight=-1,
        mask_size=28,
        debug=False,
        resize_sample=True,
        sample_dist_p=1.5,
        sample_func='sample_dist'),
    refine=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        mask_size=28,
        debug=False,
        resize_sample=True,
        sample_dist_p=1.5,
        sample_func='sample_dist'))
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    mask_thr_binary=0.5,
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = './data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
evaluation = dict(eval_epochs=(2, 8, 9, 11, 12))
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/dense_reppoints_r50_225pts_wc_grid_sparse9_mask8_partial_fpn_1x'
load_from = None
resume_from = None
auto_resume = True
workflow = [('train', 1)]
