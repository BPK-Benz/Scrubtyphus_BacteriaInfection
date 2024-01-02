_base_ = './mask_rcnn_r50_fpn_2x_coco.py'

# 1. dataset settings
# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('Infected_cells', 'Uninfected_cells', 'Irrelevant_cells')

img_scale = (int(1360/4*3), int(1024/4*3))
img_norm_cfg = dict(
    mean=[25.526, 0.386, 52.850], std=[53.347, 9.402, 53.172], to_rgb=True)

# Image augmentation by Albumentations
albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=(3,5), p=0.5),
            dict(type='MedianBlur', blur_limit=(3,5), p=0.5),
            dict(type='MotionBlur', blur_limit=(3,5), p=0.5),
        ],
        p=0.5),
]


train_pipeline = [

    dict(type='LoadImageFromFile', ),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal','vertical'] ),
    dict(type='Pad', size_divisor=32),
    
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
        'img': 'image',
        'gt_masks': 'masks',
        'gt_bboxes': 'bboxes'
    },
    update_pad_shape=False,
    skip_img_without_anno=True),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels','gt_masks'],
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
            dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal','vertical'] ),
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

# 2. model settings
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101'
                      )
    ),

    roi_head=dict(
        bbox_head=dict(num_classes=len(classes)),
        mask_head=dict(num_classes=len(classes)) 
    ),

    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=1000,
            mask_thr_binary=0.5))                     
)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
fp16 = dict(loss_scale=512.)

load_from="pretrained_models/mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth"
resume_from = None
workflow = [('train', 1),('val', 1)]
device='cuda'

runner = dict(type='EpochBasedRunner', max_epochs=15)
evaluation = dict(interval=1,metric='bbox', save_best='bbox_mAP')
work_dir='./work_dirs/New_OCT/MaskRCNN_R101_fp16_Infect'