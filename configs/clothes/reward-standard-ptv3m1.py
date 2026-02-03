_base_ = ["reward-standard-base.py"]

# PT-v3m1: only override what differs from base/defaults
model = dict(
    backbone_out_channels=64,  # final decoder stage output
    backbone=dict(type="PT-v3m1"),
)
