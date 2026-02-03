_base_ = ["../_base_/default_runtime.py"]

# ---- tunable knobs (change in child configs or via cli) ----
loss_type = "l1"              # "l1" or "pair_rank"
rank_mse_weight = 0.05         # used when loss_type == "pair_rank"
rank_reg_loss = "mae"          # "mae" or "mse"
pair_subsample_ratio = 0.2
learning_rate = 0.002
backbone_out_channels = 32
mlp_hidden = 128

# misc
batch_size = 32
num_worker = 8
mix_prob = 0.0
empty_cache = False
enable_amp = True
evaluate = True

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="PairRewardEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

# model (backbone is supplied/overridden by child configs)
model = dict(
    type="PairRewardPTv3",
    backbone_out_channels=backbone_out_channels,
    mlp_hidden=mlp_hidden,
    pair_mode="concat_diff",
    loss_type=loss_type,
    tau=2.0,
    rank_mse_weight=rank_mse_weight,
    rank_reg_loss=rank_reg_loss,
    backbone=dict(),
)

# schedule
epoch = 150
eval_epoch = 50
optimizer = dict(type="AdamW", lr=learning_rate, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[learning_rate * 2.5, learning_rate * 0.25],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=1.0,
    final_div_factor=1.0,
)
param_dicts = [dict(keyword="block", lr=learning_rate * 0.1)]

# dataset
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
        loop=3,
        reward_abs_max=10.0,
        pair_subsample_ratio=pair_subsample_ratio,
        pair_subsample_max=4096,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-3 / 180, 3 / 180], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-3 / 180, 3 / 180], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-3 / 180, 3 / 180], axis="y", p=0.5),
            dict(type="RandomShift", shift=((-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01))),
            dict(type="RandomJitter", sigma=0.002, clip=0.005),
            dict(type="AddGridCoord", grid_size=0.02),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "offset", "pairs", "pair_reward", "pair_offset"),
                feat_keys=("normal",),
                offset_keys_dict=dict(offset="coord", pair_offset="pairs"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        val_ratio=0.1,
        test_ratio=0.0,
        split_seed=0,
        reward_abs_max=10.0,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            dict(type="AddGridCoord", grid_size=0.02),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "offset", "pairs", "pair_reward", "pair_offset"),
                feat_keys=("normal",),
                offset_keys_dict=dict(offset="coord", pair_offset="pairs"),
            ),
        ],
        test_mode=False,
    ),
)
