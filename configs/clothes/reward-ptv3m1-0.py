_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 8  # total across all GPUs (reduce to fit 1x24GB)
num_worker = 8
mix_prob = 0.0
empty_cache = False
enable_amp = True
evaluate = True  # enable pair-reward validation

# slim hook list for regression
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="PairRewardEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

# model settings
model = dict(
    type="PairRewardPTv3",
    backbone_out_channels=64,
    mlp_hidden=256,
    pair_mode="concat_diff",
    loss_type="kl",
    tau=2.0,
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,  # color + normal; dataset will fill zeros if absent
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(256, 256, 256, 256),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
)

# scheduler settings (can be tuned)
epoch = 300
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
dataset_type = "PairRewardDataset"
data_root = "data/clothes"

data = dict(
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        val_ratio=0.1,
        test_ratio=0.0,
        split_seed=0,
        reward_abs_max=10.0,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-3 / 180, 3 / 180], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-3 / 180, 3 / 180], axis="x", p=0.5),  # uniform small tilt
            dict(type="RandomRotate", angle=[-3 / 180, 3 / 180], axis="y", p=0.5),  # uniform small tilt
            dict(type="RandomShift", shift=((-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01))),  # uniform tiny translation
            dict(type="RandomJitter", sigma=0.001, clip=0.005),
            dict(type="NormalizeColor"),
            dict(type="AddGridCoord", grid_size=0.02),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "offset", "pairs", "pair_reward", "pair_offset"),
                feat_keys=("color", "normal"),
                offset_keys_dict=dict(offset="coord", pair_offset="pairs"),
            ),
        ],
        test_mode=False,
    ),
    # You can enable val/test by providing corresponding json manifests.
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        val_ratio=0.1,
        test_ratio=0.0,
        split_seed=0,
        reward_abs_max=10.0,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            dict(type="AddGridCoord", grid_size=0.02),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "offset", "pairs", "pair_reward", "pair_offset"),
                feat_keys=("color", "normal"),
                offset_keys_dict=dict(offset="coord", pair_offset="pairs"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        val_ratio=0.1,
        test_ratio=0.0,
        split_seed=0,
        reward_abs_max=10.0,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="AddGridCoord", grid_size=0.02),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "offset", "pairs", "pair_reward", "pair_offset"),
                    feat_keys=("color", "normal"),
                    offset_keys_dict=dict(offset="coord", pair_offset="pairs"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
    ),
)
