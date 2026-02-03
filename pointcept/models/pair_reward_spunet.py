import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.builder import MODELS
from pointcept.utils.point import Point
from pointcept.models.utils import offset2batch


def _hash_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack (b, x, y, z) int indices into a single int64 key.
    Assumes values fit in signed 16 bits; adequate for typical grid coords.
    """
    indices = indices.long()
    b = indices[:, 0] & 0xFFFF
    x = indices[:, 1] & 0xFFFF
    y = indices[:, 2] & 0xFFFF
    z = indices[:, 3] & 0xFFFF
    return (b << 48) | (x << 32) | (y << 16) | z


@MODELS.register_module()
class PairRewardSparseUNet(nn.Module):
    """
    Use SparseUNet backbone to encode assets; per-point features are reconstructed
    by matching sparse indices back to the original order, so existing pair indices
    remain valid. Supports same losses as PairRewardPTv3.
    """

    def __init__(
        self,
        backbone,
        backbone_out_channels=64,
        mlp_hidden=256,
        pair_mode="concat_diff",
        loss_type="l1",
        tau=2.0,
        rank_mse_weight=0.1,
        rank_reg_loss="mse",
    ):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        self.backbone_out_channels = backbone_out_channels
        self.pair_mode = pair_mode
        self.loss_type = loss_type
        self.tau = tau
        self.rank_mse_weight = rank_mse_weight
        if rank_reg_loss not in ["mse", "mae"]:
            raise ValueError(f"Unknown rank_reg_loss: {rank_reg_loss}")
        self.rank_reg_loss = rank_reg_loss

        if pair_mode == "concat_diff":
            head_in_dim = backbone_out_channels * 3
        elif pair_mode == "concat":
            head_in_dim = backbone_out_channels * 2
        elif pair_mode == "diff":
            head_in_dim = backbone_out_channels
        else:
            raise ValueError(f"Unknown pair_mode: {pair_mode}")

        self.mlp = nn.Sequential(
            nn.Linear(head_in_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 1),
        )

    def _forward_backbone(self, data_dict):
        # SparseUNet consumes grid_coord/feat/offset; reconstruct per-point feat order
        grid_coord = data_dict["grid_coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"]
        batch = offset2batch(offset)

        # forward
        sparse_out = self.backbone(data_dict)
        # backbone returns voxel features; need corresponding indices
        if not hasattr(self.backbone, "_last_indices"):
            raise RuntimeError("SparseUNet backbone must expose _last_indices; update wrapper if needed.")
        voxel_indices = self.backbone._last_indices  # (M,4) int
        voxel_features = sparse_out  # (M,C)

        # map each input point to its voxel feature
        point_indices = torch.cat([batch.unsqueeze(-1).int(), grid_coord.int()], dim=1)
        voxel_keys = _hash_indices(voxel_indices)
        point_keys = _hash_indices(point_indices)

        # sort voxel keys for binary search
        voxel_keys_sorted, order = torch.sort(voxel_keys)
        loc = torch.searchsorted(voxel_keys_sorted, point_keys)
        # guard: keys should exist; but clamp to valid range to avoid crash if missing
        loc = loc.clamp(max=voxel_keys_sorted.numel() - 1)
        matched = order[loc]
        point_feat = voxel_features[matched]

        # build Point container compatible with PairReward head
        return Point(feat=point_feat, offset=offset)

    def forward(self, data_dict):
        point = self._forward_backbone(data_dict)
        feat = point.feat
        point_offset = point.offset

        pairs = data_dict["pairs"]
        pair_reward = data_dict.get("pair_reward")
        pair_offset = data_dict.get("pair_offset")

        shifted_pairs = pairs.clone()
        if pair_offset is None:
            raise ValueError("pair_offset is required to align pairs with points.")
        point_offset_full = torch.cat([point_offset.new_zeros(1), point_offset], dim=0)
        pair_offset_full = torch.cat([pair_offset.new_zeros(1), pair_offset], dim=0)
        num_samples = point_offset.shape[0]
        for i in range(num_samples):
            p_start = pair_offset_full[i].item()
            p_end = pair_offset_full[i + 1].item()
            if p_end <= p_start:
                continue
            shift = point_offset_full[i].item()
            shifted_pairs[p_start:p_end] += shift

        f0 = feat[shifted_pairs[:, 0]]
        f1 = feat[shifted_pairs[:, 1]]
        if self.pair_mode == "concat_diff":
            pair_feat = torch.cat([f0, f1, f0 - f1], dim=1)
        elif self.pair_mode == "concat":
            pair_feat = torch.cat([f0, f1], dim=1)
        elif self.pair_mode == "diff":
            pair_feat = f0 - f1
        else:
            raise ValueError(f"Unknown pair_mode: {self.pair_mode}")

        pred = self.mlp(pair_feat).squeeze(-1)
        if pair_reward is None:
            return dict(pred=pred)

        # reuse loss handling from PairRewardPTv3
        if self.loss_type == "l1":
            if not isinstance(pair_reward, torch.Tensor):
                pair_reward = torch.tensor(pair_reward, device=pred.device, dtype=pred.dtype)
            if pair_reward.ndim > 1:
                pair_reward = pair_reward.squeeze(-1)
            loss = F.l1_loss(pred, pair_reward)
            return dict(loss=loss, pred_mean=pred.mean())

        if self.loss_type == "pair_rank":
            if not isinstance(pair_reward, torch.Tensor):
                pair_reward = torch.tensor(pair_reward, device=pred.device, dtype=pred.dtype)
            if pair_reward.ndim > 1:
                pair_reward = pair_reward.squeeze(-1)
            if pair_offset is None:
                raise ValueError("pair_offset is required to align pairs with points.")

            pair_offset_full = torch.cat([pair_offset.new_zeros(1), pair_offset], dim=0)

            rank_loss_sum = pred.new_zeros(1)
            mse_loss_sum = pred.new_zeros(1)
            count_rank_assets = 0
            count_assets = 0

            for i in range(pair_offset_full.numel() - 1):
                p_start = pair_offset_full[i].item()
                p_end = pair_offset_full[i + 1].item()
                if p_end <= p_start:
                    continue

                pred_i = pred[p_start:p_end]
                target_i = pair_reward[p_start:p_end]
                count_assets += 1

                if pred_i.numel() > 1:
                    diff_pred = pred_i.unsqueeze(1) - pred_i.unsqueeze(0)
                    diff_target = target_i.unsqueeze(1) - target_i.unsqueeze(0)
                    sign = torch.sign(diff_target)
                    upper_mask = torch.triu(torch.ones_like(sign, dtype=torch.bool), diagonal=1)
                    sign = sign[upper_mask]
                    diff_pred = diff_pred[upper_mask]
                    non_zero = sign != 0
                    sign = sign[non_zero]
                    diff_pred = diff_pred[non_zero]
                    if sign.numel() > 0:
                        rank_loss_sum = rank_loss_sum + F.softplus(-sign * diff_pred).mean()
                        count_rank_assets += 1

                if self.rank_reg_loss == "mae":
                    reg_loss = F.l1_loss(pred_i, target_i)
                else:
                    reg_loss = F.mse_loss(pred_i, target_i)
                mse_loss_sum = mse_loss_sum + reg_loss

            rank_loss = pred.sum() * 0 if count_rank_assets == 0 else rank_loss_sum / count_rank_assets
            mse_loss = pred.sum() * 0 if count_assets == 0 else mse_loss_sum / count_assets
            loss = rank_loss + self.rank_mse_weight * mse_loss
            return dict(loss=loss, rank_loss=rank_loss, mse_loss=mse_loss, pred_mean=pred.mean())

        raise ValueError(f"Unknown loss_type: {self.loss_type}")
