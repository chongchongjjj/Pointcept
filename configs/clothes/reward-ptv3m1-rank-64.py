_base_ = ["reward-ptv3m1-base.py"]

model = dict(
    loss_type="pair_rank",
    rank_mse_weight=0.05,
    rank_reg_loss="mae",
    backbone_out_channels=64,
    mlp_hidden=192,
    backbone=dict(
        enc_channels=(16, 32, 64, 128, 256),
        dec_channels=(64, 64, 128, 256),
    ),
)
