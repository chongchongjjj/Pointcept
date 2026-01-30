import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from pointcept.models.builder import MODELS


@MODELS.register_module()
class PairRewardPTv3(nn.Module):
    """
    Use PTv3 backbone to encode two assets, then an MLP predicts a scalar reward.

    Expected input keys:
        - coord / grid_coord / feat (via Collect)
        - offset: cumulative lengths for each asset (2 per sample)
        - pair_id: per-point {0,1} label (kept for debugging/verification)
        - reward: scalar tensor shape [B] or [B, 1] (optional at inference)
    """

    def __init__(
        self,
        backbone,
        backbone_out_channels=64,
        mlp_hidden=256,
        pair_mode="concat_diff",  # concat [f0, f1, f0-f1]
    ):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        self.backbone_out_channels = backbone_out_channels
        self.pair_mode = pair_mode

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

    def forward(self, data_dict):
        """
        data_dict keys:
            coord/grid_coord/offset/... (for backbone)
            pairs: (M,2) long indices per-sample (before batching)
            pair_offset: cumulative counts of pairs per sample
            pair_reward: (M,) float, same order as pairs
        """
        point = self.backbone(data_dict)
        feat = point.feat  # (N, C)
        point_offset = point.offset  # cumulative lengths of points per sample

        pairs = data_dict["pairs"]
        pair_reward = data_dict.get("pair_reward")
        pair_offset = data_dict.get("pair_offset")

        # shift pairs per sample according to point_offset
        shifted_pairs = pairs.clone()
        if pair_offset is None:
            raise ValueError("pair_offset is required to align pairs with points.")
        point_offset_full = torch.cat(
            [point_offset.new_zeros(1), point_offset], dim=0
        )  # (B+1)
        pair_offset_full = torch.cat(
            [pair_offset.new_zeros(1), pair_offset], dim=0
        )  # (B+1)
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
        if pair_reward is not None:
            if isinstance(pair_reward, torch.Tensor):
                pair_reward = pair_reward.to(pred)
            if pair_reward.ndim > 1:
                pair_reward = pair_reward.squeeze(-1)
            loss = F.l1_loss(pred, pair_reward)
            return dict(loss=loss, pred_mean=pred.mean())
        else:
            return dict(pred=pred)
