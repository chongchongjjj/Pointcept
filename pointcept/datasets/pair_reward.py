"""
Pair-wise reward dataset for point clouds（资产目录扫描版）。

数据格式：
data_root/assets/asset_xxxx/
    coord.npy
    segment.npy (可全 -1)
    normal.npy (可选)
    pairs.npy   (K, 2) int64，资产对的索引（默认从目录名里的数字解析，如 asset_0001 -> id=1）
    reward.npy  (K,) float32，与 pairs 对齐

会遍历所有资产目录，把 pairs.npy 里的索引映射到资产路径，生成配对样本并附上 reward。
同一对资产只保留第一次出现的 reward 以去重。支持在 Dataset 内通过 val_ratio/test_ratio
划分 train/val/test。
"""

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class PairRewardDataset(Dataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
        "pose",
    ]

    def __init__(
        self,
        split: str = "train",
        data_root: str = "data/clothes",
        val_ratio: float = 0.0,
        test_ratio: float = 0.0,
        split_seed: int = 0,
        assets_dir: str = "assets",
        transform=None,
        test_mode: bool = False,
        cache: bool = False,
        ignore_index: int = -1,
        loop: int = 1,
    ):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.assets_dir = assets_dir
        self.transform = Compose(transform)
        self.cache = cache
        self.ignore_index = ignore_index
        self.test_mode = test_mode
        self.loop = loop if not test_mode else 1

        self.data_list = self._get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} {} set.".format(
                len(self.data_list), self.loop, os.path.basename(self.data_root), split
            )
        )

    def _get_data_list(self) -> List[Dict]:
        raw_list = self._build_pairs_from_assets()
        raw_list = self._auto_split(raw_list)
        data_list = []
        for idx, item in enumerate(raw_list):
            a_path = item["a"]
            b_path = item["b"]
            reward = item.get("reward", 0.0)
            name = item.get("name", f"{self.split}_{idx}")
            if not os.path.isabs(a_path):
                a_path = os.path.join(self.data_root, a_path)
            if not os.path.isabs(b_path):
                b_path = os.path.join(self.data_root, b_path)
            data_list.append(dict(a=a_path, b=b_path, reward=reward, name=name))
        return data_list

    def _auto_split(self, raw_list: List[Dict]) -> List[Dict]:
        if self.val_ratio == 0 and self.test_ratio == 0:
            return raw_list
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(len(raw_list))
        raw_list = [raw_list[i] for i in perm]
        n_total = len(raw_list)
        n_val = int(n_total * self.val_ratio)
        n_test = int(n_total * self.test_ratio)
        n_train = n_total - n_val - n_test
        split_map = {
            "train": raw_list[:n_train],
            "val": raw_list[n_train : n_train + n_val],
            "test": raw_list[n_train + n_val : n_train + n_val + n_test],
        }
        return split_map.get(self.split, [])

    def _parse_asset_id(self, name: str, fallback: int) -> int:
        m = re.findall(r"\d+", name)
        if len(m) > 0:
            try:
                return int(m[-1])
            except Exception:
                pass
        return fallback

    def _build_pairs_from_assets(self) -> List[Dict]:
        assets_root = (
            self.assets_dir
            if os.path.isabs(self.assets_dir)
            else os.path.abspath(os.path.join(self.data_root, self.assets_dir))
        )
        if not os.path.isdir(assets_root):
            raise FileNotFoundError(
                f"Assets directory {assets_root} not found for automatic pairing."
            )
        asset_dirs = sorted(
            [
                os.path.join(assets_root, d)
                for d in os.listdir(assets_root)
                if os.path.isdir(os.path.join(assets_root, d))
            ]
        )
        id_to_path: Dict[int, str] = {}
        for idx, path in enumerate(asset_dirs):
            name = os.path.basename(path)
            asset_id = self._parse_asset_id(name, fallback=idx)
            id_to_path[asset_id] = path

        pair_entries: Dict[Tuple[int, int], float] = {}
        for path in asset_dirs:
            pairs_path = os.path.join(path, "pairs.npy")
            reward_path = os.path.join(path, "reward.npy")
            if not (os.path.isfile(pairs_path) and os.path.isfile(reward_path)):
                continue
            pairs = np.load(pairs_path)
            rewards = np.load(reward_path)
            if len(pairs.shape) != 2 or pairs.shape[1] != 2:
                raise ValueError(f"{pairs_path} should have shape (K, 2)")
            if pairs.shape[0] != rewards.shape[0]:
                raise ValueError(f"pairs and reward length mismatch in {path}")
            for row, r in zip(pairs, rewards):
                a_id, b_id = int(row[0]), int(row[1])
                if a_id not in id_to_path or b_id not in id_to_path:
                    continue
                key = tuple(sorted((a_id, b_id)))
                # keep first occurrence to avoid duplicates across assets
                if key not in pair_entries:
                    pair_entries[key] = float(r)

        data_list = []
        for idx, ((a_id, b_id), reward) in enumerate(pair_entries.items()):
            data_list.append(
                dict(
                    a=id_to_path[a_id],
                    b=id_to_path[b_id],
                    reward=reward,
                    name=f"{a_id}-{b_id}",
                )
            )
        return data_list

    def _load_asset(self, asset_path: str) -> Dict:
        data_dict = {}
        assets = os.listdir(asset_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            key = asset[:-4]
            if key not in self.VALID_ASSETS:
                continue
            data_dict[key] = np.load(os.path.join(asset_path, asset))

        if "coord" not in data_dict:
            raise FileNotFoundError(f"coord.npy is required in {asset_path}")

        # fallbacks for optional assets to keep shapes consistent
        n_points = data_dict["coord"].shape[0]
        if "color" not in data_dict:
            data_dict["color"] = np.zeros((n_points, 3), dtype=np.float32)
        if "normal" not in data_dict:
            data_dict["normal"] = np.zeros((n_points, 3), dtype=np.float32)
        if "segment" not in data_dict:
            data_dict["segment"] = np.ones(n_points, dtype=np.int32) * -1
        if "instance" not in data_dict:
            data_dict["instance"] = np.ones(n_points, dtype=np.int32) * -1

        # dtypes
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)
        data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        return data_dict

    def get_data(self, idx: int) -> Dict:
        sample = self.data_list[idx % len(self.data_list)]
        data_a = self._load_asset(sample["a"])
        data_b = self._load_asset(sample["b"])

        coord = np.concatenate([data_a["coord"], data_b["coord"]], axis=0)
        color = np.concatenate([data_a["color"], data_b["color"]], axis=0)
        normal = np.concatenate([data_a["normal"], data_b["normal"]], axis=0)
        segment = np.concatenate([data_a["segment"], data_b["segment"]], axis=0)
        instance = np.concatenate([data_a["instance"], data_b["instance"]], axis=0)

        len_a = data_a["coord"].shape[0]
        len_b = data_b["coord"].shape[0]
        offset = np.array([len_a, len_a + len_b], dtype=np.int64)

        # pairs and rewards belong to asset A only (per-asset vertex indices)
        pairs = data_a.get("pairs")
        pair_reward = data_a.get("pair_reward", None)
        if pairs is None:
            # try load pairs/reward from asset A folder
            pairs_path = os.path.join(sample["a"], "pairs.npy")
            reward_path = os.path.join(sample["a"], "reward.npy")
            if not (os.path.isfile(pairs_path) and os.path.isfile(reward_path)):
                raise FileNotFoundError(f"pairs.npy or reward.npy missing in {sample['a']}")
            pairs = np.load(pairs_path)
            pair_reward = np.load(reward_path)
        pairs = pairs.astype(np.int64)
        pair_reward = pair_reward.astype(np.float32)
        pair_offset = np.array([pairs.shape[0]], dtype=np.int64)

        data_dict = dict(
            coord=coord,
            color=color,
            normal=normal,
            segment=segment,
            instance=instance,
            offset=offset,
            pairs=pairs,
            pair_reward=pair_reward,
            pair_offset=pair_offset,
            name=sample["name"],
            split=self.split,
        )
        data_dict["index_valid_keys"] = [
            "coord",
            "color",
            "normal",
            "strength",
            "segment",
            "instance",
        ]
        return data_dict

    def prepare_train_data(self, idx: int) -> Dict:
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx: int) -> Dict:
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def __getitem__(self, idx: int) -> Dict:
        if self.split == "test":
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self) -> int:
        return len(self.data_list) * self.loop
