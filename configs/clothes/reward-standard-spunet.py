_base_ = ["reward-standard-base.py"]

# SparseUNet (SpConv v1m1 recommended in repo)
model = dict(
    type="PairRewardSparseUNet",
    backbone_out_channels=32,
    mlp_hidden=128,
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=3,
        num_classes=1,  # not used but required by constructor
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
)
