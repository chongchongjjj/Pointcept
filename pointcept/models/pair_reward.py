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
        loss_type="kl",  # "kl" or "l1" or "pair_rank"
        tau=2.0,
        rank_mse_weight=0.1,
    ):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        self.backbone_out_channels = backbone_out_channels
        self.pair_mode = pair_mode
        self.loss_type = loss_type
        self.tau = tau
        self.rank_mse_weight = rank_mse_weight

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
        if pair_reward is None:
            return dict(pred=pred)

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

            pair_offset_full = torch.cat(
                [pair_offset.new_zeros(1), pair_offset], dim=0
            )  # (B+1)

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

                # matrix-based pairwise ranking
                if pred_i.numel() > 1:
                    diff_pred = pred_i.unsqueeze(1) - pred_i.unsqueeze(0)  # (k, k)
                    diff_target = target_i.unsqueeze(1) - target_i.unsqueeze(0)
                    sign = torch.sign(diff_target)
                    upper_mask = torch.triu(
                        torch.ones_like(sign, dtype=torch.bool), diagonal=1
                    )
                    sign = sign[upper_mask]
                    diff_pred = diff_pred[upper_mask]

                    non_zero = sign != 0
                    sign = sign[non_zero]
                    diff_pred = diff_pred[non_zero]

                    if sign.numel() > 0:
                        # softplus(-s * delta) is logistic pairwise loss
                        rank_loss_sum = rank_loss_sum + F.softplus(-sign * diff_pred).mean()
                        count_rank_assets += 1

                mse_loss_sum = mse_loss_sum + F.mse_loss(pred_i, target_i)

            if count_rank_assets == 0:
                rank_loss = pred.sum() * 0
            else:
                rank_loss = rank_loss_sum / count_rank_assets

            if count_assets == 0:
                mse_loss = pred.sum() * 0
            else:
                mse_loss = mse_loss_sum / count_assets

            loss = rank_loss + self.rank_mse_weight * mse_loss
            return dict(loss=loss, rank_loss=rank_loss, mse_loss=mse_loss, pred_mean=pred.mean())

        # KL-based distribution matching per asset (loss_type == "kl")
        if not isinstance(pair_reward, torch.Tensor):
            pair_reward = torch.tensor(pair_reward, device=pred.device, dtype=pred.dtype)
        if pair_reward.ndim > 1:
            pair_reward = pair_reward.squeeze(-1)
        eps = 1e-8

        if pair_offset is None:
            raise ValueError("pair_offset is required to align pairs with points.")

        pair_offset_full = torch.cat(
            [pair_offset.new_zeros(1), pair_offset], dim=0
        )  # (B+1)

        loss_kl = pred.new_zeros(1)
        count_assets = 0
        for i in range(pair_offset_full.numel() - 1):
            p_start = pair_offset_full[i].item()
            p_end = pair_offset_full[i + 1].item()
            if p_end <= p_start + 1:
                # too few pairs to form a distribution; skip
                continue
            logits_i = pred[p_start:p_end]
            target_i = pair_reward[p_start:p_end]

            log_q = F.log_softmax(logits_i / self.tau, dim=0)
            p = F.softmax(target_i / self.tau, dim=0)
            p = torch.clamp(p, min=eps)

            loss_kl = loss_kl + F.kl_div(log_q, p, reduction="batchmean")
            count_assets += 1

        if count_assets == 0:
            # no valid assets; return zero loss (keeps graph)
            loss_kl = pred.sum() * 0
        else:
            loss_kl = loss_kl / count_assets

        return dict(loss=loss_kl, pred_mean=pred.mean())
