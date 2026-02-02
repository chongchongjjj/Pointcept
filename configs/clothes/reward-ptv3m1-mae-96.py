_base_ = ["reward-ptv3m1-base.py"]

model = dict(
    loss_type="l1",
    backbone_out_channels=96,
    mlp_hidden=256,
    backbone=dict(
        enc_channels=(24, 48, 96, 192, 384),
        dec_channels=(96, 96, 192, 384),
    ),
)
