_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# # data
# data = dict(samples_per_gpu=8)

# 1. dataset settings
# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('Infected_cells','Uninfected_cells', 'Irrelevant_cells' )

img_scale = (int(1360/4*3), int(1024/4*3))
img_norm_cfg = dict(
    mean=[25.526, 0.386, 52.850], std=[53.347, 9.402, 53.172], to_rgb=True)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor')),

]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

base = "basement_path"
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file= base+'train_json_file',
        img_prefix= base,
        classes=classes,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file= base+'val_json_file',
        img_prefix= base,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file= base+'test_json_file',
        img_prefix= base,
        classes=classes,
        pipeline=test_pipeline,
    )
)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)


# optimizer
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')
    ),
    neck=dict(in_channels=[256, 512, 1024, 2048],
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=len(classes)
    ),
    test_cfg=dict(
        max_per_img=300)
)

fp16 = dict(loss_scale=512.)

load_from = 'pretrained_models/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth'
resume_from = None
workflow = [('train', 1),('val', 1)]
device='cuda'

runner = dict(type='EpochBasedRunner', max_epochs=15)
evaluation = dict(interval=1,metric='bbox', save_best='bbox_mAP')
work_dir='./work_dirs/New_OCT/Retina_R101_fp16_Infect'
