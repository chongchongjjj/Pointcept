_base_ = ["reward-standard-base.py"]

model = dict(
    type="PairRewardSparseUNet",
    backbone_out_channels=96,  # SpUNet v1m1 默认最终特征维
    backbone=dict(type="SpUNet-v1m1", in_channels=3, num_classes=0),  # num_classes=0 -> Identity head
)
