_base_ = ["reward-standard-base.py"]

# PT-v3m1: minimal overrides on top of reward-standard-base
model = dict(
    type="PairRewardPTv3",
    backbone_out_channels=64,  # decoder stage 0 output dimension
<<<<<<< HEAD
    backbone=dict(type="PT-v3m1"),
=======
    backbone=dict(
        type="PT-v3m1",
        enable_flash=False,
        in_channels=3,
        ),
>>>>>>> f15dfe2ac5362ecc99e2d357272b536ae300fb66
)
