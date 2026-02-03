# Clothes Reward Experiments (standard backbones)

Tunable knobs (override via child configs or CLI): `learning_rate`, `loss_type` (`l1` or `pair_rank`), `rank_mse_weight`, `rank_reg_loss`, `pair_subsample_ratio`.

Base template
- `reward-standard-base.py`: common dataloader/schedule + knobs above.

Standard recommended backbones
- PT-v3m1: `reward-standard-ptv3m1.py` (paper default widths/depths).
- PT-v3m2: `reward-standard-ptv3m2.py` (paper default widths/depths).
- SparseUNet (SpConv v1m1): `reward-standard-spunet.py`.

Usage examples
- Default MAE, pr=0.2: `python tools/train.py configs/clothes/reward-standard-ptv3m1.py`
- Change loss to rank+mae: `python tools/train.py configs/clothes/reward-standard-ptv3m1.py --cfg-options loss_type=pair_rank rank_mse_weight=0.05 rank_reg_loss=mae`
- Sweep pr sampling: `python tools/train.py configs/clothes/reward-standard-spunet.py --cfg-options pair_subsample_ratio=0.1`
- Lower LR: `python tools/train.py configs/clothes/reward-standard-ptv3m2.py --cfg-options learning_rate=0.001`
