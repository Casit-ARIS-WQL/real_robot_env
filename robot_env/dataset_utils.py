"""
dataset_utils.py  ——  VLA 多格式数据集加载工具

支持格式
--------
* HDF5       —— 本地 .h5 文件，episodes/i/{images, tcp_poses, …}
* LeRobot    —— Hugging Face LeRobot 格式 (parquet + meta/episodes.json)
* RLDS       —— Google Robot Learning Dataset (tensorflow_datasets / TFRecord)

快速入门
--------
    from dataset_utils import load_dataset, compute_dataset_stats

    ds = load_dataset("/data/robot.h5")           # 自动检测 HDF5
    ds = load_dataset("/data/lerobot_dataset/")   # 自动检测 LeRobot
    ep = ds.get_episode(0)                        # 返回 EpisodeData
    print(len(ds))                                # 共几条轨迹
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  通用 Episode 数据容器
# ---------------------------------------------------------------------------

@dataclass
class EpisodeData:
    """单条轨迹数据，所有字段均为 numpy 数组。

    字段说明
    --------
    images        : (T, H, W, 3)   uint8   —— RGB 帧序列
    tcp_poses     : (T, D_tcp)     float32  —— 末端位姿；单臂 6 维 [x,y,z,rx,ry,rz]，
                                              双臂 14 维（左右各 7 维，含夹爪）
    joint_states  : (T, D_jst)     float32  —— 关节状态；单臂 12 维，双臂 14 维
    force_torques : (T, D_ft)      float32  —— 六轴力/力矩（通常 6 维，可全零）
    actions       : (T, D_act)     float32  —— 专家动作；单臂 7 维，双臂 14 维
    language      : str                     —— 任务语言描述
    extra         : dict                    —— 额外字段（多相机、关节动作等）
    """
    images:        np.ndarray
    tcp_poses:     np.ndarray
    joint_states:  np.ndarray
    force_torques: np.ndarray
    actions:       np.ndarray
    language:      str = ""
    extra:         Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.actions)

    def __len__(self) -> int:
        return self.length


# ---------------------------------------------------------------------------
#  抽象基类
# ---------------------------------------------------------------------------

class DatasetLoader(ABC):
    """VLA 数据集加载器抽象接口。"""

    @abstractmethod
    def __len__(self) -> int:
        """返回数据集中 episode 总数。"""

    @abstractmethod
    def get_episode(self, idx: int) -> EpisodeData:
        """按索引加载单条 episode。"""

    def __getitem__(self, idx: int) -> EpisodeData:
        return self.get_episode(idx)

    def __iter__(self) -> Iterator[EpisodeData]:
        for i in range(len(self)):
            yield self.get_episode(i)

    def random_episode(self, rng: Optional[np.random.Generator] = None) -> EpisodeData:
        """随机采样一条 episode。"""
        rng = rng or np.random.default_rng()
        idx = int(rng.integers(0, len(self)))
        return self.get_episode(idx)

    def transition_iter(
        self,
        batch_size: int = 256,
        shuffle: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterator[Dict[str, np.ndarray]]:
        """
        迭代离散 (s, a, r, s', done) 转换对，用于离线 RL 算法（CQL / IQL / TD3-BC）。

        每次 yield 一个字典，key 说明
        ---------------------------
        obs_image       : (B, H, W, 3)  uint8
        obs_proprio     : (B, D)        float32   D = D_jst（仅关节状态，由数据决定）
        actions         : (B, D_act)    float32
        next_obs_image  : (B, H, W, 3)  uint8
        next_obs_proprio: (B, D)        float32
        dones           : (B,)          float32   episode 末步=1
        """
        rng = rng or np.random.default_rng()
        all_transitions = self._build_transition_pool()
        indices = np.arange(len(all_transitions["actions"]))
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start: start + batch_size]
            yield {k: v[batch_idx] for k, v in all_transitions.items()}

    def _build_transition_pool(self) -> Dict[str, np.ndarray]:
        """将全部 episode 展开为转换对池（全量加载，适合中小型数据集）。"""
        pools: Dict[str, List[np.ndarray]] = {
            k: [] for k in (
                "obs_image", "obs_proprio",
                "actions",
                "next_obs_image", "next_obs_proprio",
                "dones",
            )
        }
        for ep in self:
            T = ep.length
            proprio = _build_proprio(ep)   # (T, 14) 双臂关节状态
            for t in range(T):
                t_next = min(t + 1, T - 1)
                done = float(t == T - 1)
                pools["obs_image"].append(ep.images[t])
                pools["obs_proprio"].append(proprio[t])
                pools["actions"].append(ep.actions[t])
                pools["next_obs_image"].append(ep.images[t_next])
                pools["next_obs_proprio"].append(proprio[t_next])
                pools["dones"].append(np.array(done, dtype=np.float32))
        return {k: np.stack(v) for k, v in pools.items()}


# ---------------------------------------------------------------------------
#  HDF5 加载器
# ---------------------------------------------------------------------------

class HDF5DatasetLoader(DatasetLoader):
    """
    加载本地 HDF5 格式数据集。

    期望文件结构
    -----------
    episodes/
      0/
        images        : (T, H, W, 3)  uint8
        tcp_poses     : (T, 6)        float32
        joint_states  : (T, J)        float32
        force_torques : (T, 6)        float32
        actions       : (T, 7)        float32
        language      : bytes  (可选)
      1/ ...
    """

    def __init__(self, path: str):
        try:
            import h5py
        except ImportError:
            raise ImportError("HDF5 加载器需要: pip install h5py")

        self._h5 = __import__("h5py").File(path, "r")
        self._num_episodes = len(self._h5["episodes"])
        logger.info("HDF5 数据集加载: %s  (%d 条轨迹)", path, self._num_episodes)

    def __len__(self) -> int:
        return self._num_episodes

    def get_episode(self, idx: int) -> EpisodeData:
        ep = self._h5[f"episodes/{idx}"]
        lang = ""
        if "language" in ep:
            raw = ep["language"][()]
            lang = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

        # 兼容 joint_states 可能缺失的数据集
        if "joint_states" in ep:
            joint_states = ep["joint_states"][:]
        else:
            T = len(ep["tcp_poses"])
            joint_states = np.zeros((T, 12), dtype=np.float32)

        # 兼容 force_torques 可能缺失的数据集
        if "force_torques" in ep:
            force_torques = ep["force_torques"][:]
        else:
            T = len(ep["tcp_poses"])
            force_torques = np.zeros((T, 6), dtype=np.float32)

        return EpisodeData(
            images=ep["images"][:],
            tcp_poses=ep["tcp_poses"][:],
            joint_states=joint_states,
            force_torques=force_torques,
            actions=ep["actions"][:],
            language=lang,
        )

    def close(self):
        self._h5.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
#  LeRobot 加载器
# ---------------------------------------------------------------------------

class LeRobotDatasetLoader(DatasetLoader):
    """
    加载 Hugging Face LeRobot 格式数据集。

    目录结构
    --------
    <root>/
      meta/
        episodes.json       —— 每条 episode 的帧范围 [{id, from, to, task}]
        info.json           —— 数据集元信息（FPS、特征维度等）
      data/
        chunk-000/
          episode_000000.parquet
          ...

    字段映射（默认）
    --------
    observation.image          -> images
    observation.state          -> tcp_poses (前6列) + joint_states (余下列)
    action                     -> actions
    observation.effort         -> force_torques (可选)
    """

    # LeRobot 默认图像列名
    _IMAGE_KEYS = ("observation.image", "observation.images.top",
                   "observation.images.wrist")
    _STATE_KEY  = "observation.state"
    _ACTION_KEY = "action"
    _EFFORT_KEY = "observation.effort"

    def __init__(self, root: str, image_key: Optional[str] = None):
        try:
            import pandas as pd
            import json
        except ImportError:
            raise ImportError("LeRobot 加载器需要: pip install pandas pyarrow")

        self._root = Path(root)
        import json as _json
        meta_path = self._root / "meta" / "episodes.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"未找到 LeRobot meta 文件: {meta_path}")

        with open(meta_path) as f:
            self._episode_meta: List[Dict] = _json.load(f)

        # 选定图像列
        self._image_key = image_key  # None = 自动检测

        # 延迟加载 parquet，避免一次性占用过多内存
        self._parquet_cache: Dict[str, object] = {}
        logger.info("LeRobot 数据集加载: %s  (%d 条轨迹)", root, len(self._episode_meta))

    def __len__(self) -> int:
        return len(self._episode_meta)

    def get_episode(self, idx: int) -> EpisodeData:
        import pandas as pd

        meta = self._episode_meta[idx]
        ep_id = meta["episode_index"] if "episode_index" in meta else idx
        task  = meta.get("task", meta.get("task_index", ""))

        df = self._load_parquet(ep_id)

        # ---- 动作 ----
        action_cols = [c for c in df.columns if c == self._ACTION_KEY or
                       (c.startswith("action") and df[c].dtype in (np.float32, np.float64, "float32", "float64"))]
        if action_cols:
            actions = np.stack(df[action_cols[0]].values).astype(np.float32)
            if actions.ndim == 1:
                actions = np.array([np.array(a) for a in df[action_cols[0]]], dtype=np.float32)
        else:
            T = len(df)
            actions = np.zeros((T, 7), dtype=np.float32)

        # ---- 状态 ----
        state_col = None
        for c in [self._STATE_KEY] + [k for k in df.columns if "state" in k.lower()]:
            if c in df.columns:
                state_col = c
                break

        if state_col:
            states = np.array([np.array(s) for s in df[state_col]], dtype=np.float32)
        else:
            states = np.zeros((len(df), 12), dtype=np.float32)

        tcp_poses    = states[:, :6]   if states.shape[1] >= 6  else np.pad(states, [(0,0),(0,6-states.shape[1])])
        joint_states = states[:, 6:18] if states.shape[1] >= 18 else np.zeros((len(df), 12), dtype=np.float32)

        # ---- 力矩 ----
        effort_col = self._EFFORT_KEY if self._EFFORT_KEY in df.columns else None
        if effort_col:
            force_torques = np.array([np.array(e) for e in df[effort_col]], dtype=np.float32)
            if force_torques.ndim == 1:
                force_torques = force_torques.reshape(-1, 1).repeat(6, axis=1)
        else:
            force_torques = np.zeros((len(df), 6), dtype=np.float32)

        # ---- 图像 ----
        img_key = self._image_key or self._detect_image_key(df)
        if img_key and img_key in df.columns:
            raw_imgs = df[img_key].values
            images = np.stack([self._decode_image(img) for img in raw_imgs])
        else:
            T = len(df)
            images = np.zeros((T, 224, 224, 3), dtype=np.uint8)

        # ---- 语言 ----
        if isinstance(task, str):
            lang = task
        else:
            lang = str(task)

        return EpisodeData(
            images=images,
            tcp_poses=tcp_poses,
            joint_states=joint_states,
            force_torques=force_torques,
            actions=actions,
            language=lang,
        )

    # ---- 内部工具 ----

    def _detect_image_key(self, df) -> Optional[str]:
        for key in self._IMAGE_KEYS:
            if key in df.columns:
                return key
        # 模糊匹配
        for col in df.columns:
            if "image" in col.lower():
                return col
        return None

    @staticmethod
    def _decode_image(raw) -> np.ndarray:
        """解码图像：支持 bytes(JPEG/PNG)、dict{"path":…}、ndarray。"""
        if isinstance(raw, np.ndarray):
            return raw.astype(np.uint8)
        if isinstance(raw, (bytes, bytearray)):
            try:
                import cv2
                buf = np.frombuffer(raw, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except ImportError:
                from PIL import Image
                import io
                return np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
        if isinstance(raw, dict) and "path" in raw:
            try:
                import cv2
                img = cv2.imread(raw["path"])
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
            except ImportError:
                from PIL import Image
                return np.array(Image.open(raw["path"]).convert("RGB"), dtype=np.uint8)
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def _load_parquet(self, ep_id: int):
        import pandas as pd
        # 查找对应 parquet 文件
        pattern = f"episode_{ep_id:06d}.parquet"
        for p in self._root.glob(f"data/**/{pattern}"):
            if str(p) not in self._parquet_cache:
                self._parquet_cache[str(p)] = pd.read_parquet(p)
            return self._parquet_cache[str(p)]
        raise FileNotFoundError(f"未找到 parquet 文件: {pattern}  (root={self._root})")


# ---------------------------------------------------------------------------
#  RLDS 加载器  (Google tensorflow_datasets / TFRecord)
# ---------------------------------------------------------------------------

class RLDSDatasetLoader(DatasetLoader):
    """
    加载 RLDS / Open X-Embodiment 格式数据集。

    依赖
    ----
        pip install tensorflow tensorflow-datasets

    参数
    ----
    dataset_name : tf.data.Dataset name 或本地路径
    split        : "train" / "test" 等
    image_key    : observation 中的图像键名，默认 "image"
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        data_dir: Optional[str] = None,
        image_key: str = "image",
        action_key: str = "action",
    ):
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError("RLDS 加载器需要: pip install tensorflow tensorflow-datasets")

        self._image_key  = image_key
        self._action_key = action_key

        builder = tfds.builder(dataset_name, data_dir=data_dir)
        builder.download_and_prepare()
        ds = builder.as_dataset(split=split)

        # 预加载所有 episode（RLDS episode 为 tf.data Dataset 中的 record）
        self._episodes: List[Dict] = []
        for ep in ds:
            self._episodes.append(self._parse_episode(ep))

        logger.info("RLDS 数据集加载: %s/%s  (%d 条轨迹)",
                    dataset_name, split, len(self._episodes))

    def __len__(self) -> int:
        return len(self._episodes)

    def get_episode(self, idx: int) -> EpisodeData:
        return self._episodes[idx]

    def _parse_episode(self, ep) -> EpisodeData:
        import tensorflow as tf
        steps = ep["steps"]
        images, actions, tcp_poses, joint_states, force_torques = [], [], [], [], []
        lang = ""

        for step in steps:
            obs = step["observation"]
            # 图像
            img_raw = obs.get(self._image_key, None)
            if img_raw is not None:
                images.append(img_raw.numpy().astype(np.uint8))
            # 动作
            act = step.get(self._action_key, step["action"])
            actions.append(act.numpy().astype(np.float32))
            # 本体感觉（尽力解析）
            state = obs.get("state", obs.get("proprio", None))
            if state is not None:
                s = state.numpy().astype(np.float32)
                tcp_poses.append(s[:6] if len(s) >= 6 else np.pad(s, (0, 6 - len(s))))
                joint_states.append(s[6:18] if len(s) >= 18 else np.zeros(12, dtype=np.float32))
            else:
                tcp_poses.append(np.zeros(6, dtype=np.float32))
                joint_states.append(np.zeros(12, dtype=np.float32))
            force_torques.append(np.zeros(6, dtype=np.float32))
            # 语言
            if not lang:
                for k in ("language_instruction", "natural_language_instruction", "instruction"):
                    v = step.get(k, None)
                    if v is not None:
                        raw = v.numpy()
                        lang = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                        break

        T = len(actions)
        return EpisodeData(
            images=np.stack(images) if images else np.zeros((T, 224, 224, 3), dtype=np.uint8),
            tcp_poses=np.stack(tcp_poses),
            joint_states=np.stack(joint_states),
            force_torques=np.stack(force_torques),
            actions=np.stack(actions),
            language=lang,
        )


# ---------------------------------------------------------------------------
#  自动检测工厂
# ---------------------------------------------------------------------------

def load_dataset(
    path: str,
    fmt: str = "auto",
    **kwargs,
) -> DatasetLoader:
    """
    自动检测并加载 VLA 数据集。

    Parameters
    ----------
    path : str
        本地文件/目录路径，或 tensorflow_datasets 数据集名称。
    fmt  : str
        "auto" | "hdf5" | "lerobot" | "rlds"
    **kwargs :
        透传给对应 Loader 的额外参数。

    Returns
    -------
    DatasetLoader 实例
    """
    p = Path(path)

    if fmt == "auto":
        if p.suffix in (".h5", ".hdf5"):
            fmt = "hdf5"
        elif p.is_dir() and (p / "meta" / "episodes.json").exists():
            fmt = "lerobot"
        elif p.is_dir() and any(p.glob("*.tfrecord*")):
            fmt = "rlds"
        elif not p.exists():
            # 假设是 tensorflow_datasets 数据集名称
            fmt = "rlds"
        else:
            raise ValueError(
                f"无法自动检测数据集格式: {path}  "
                "请显式指定 fmt='hdf5'|'lerobot'|'rlds'"
            )

    if fmt == "hdf5":
        return HDF5DatasetLoader(path, **kwargs)
    elif fmt == "lerobot":
        return LeRobotDatasetLoader(path, **kwargs)
    elif fmt == "rlds":
        return RLDSDatasetLoader(path, **kwargs)
    else:
        raise ValueError(f"未知数据集格式: {fmt}")


# ---------------------------------------------------------------------------
#  数据集统计工具
# ---------------------------------------------------------------------------

def compute_dataset_stats(
    loader: DatasetLoader,
    max_episodes: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    计算数据集动作/本体感觉的均值和标准差，用于归一化。

    Returns
    -------
    stats : dict with keys
        action_mean, action_std, action_min, action_max,
        proprio_mean, proprio_std
    """
    all_actions: List[np.ndarray] = []
    all_proprios: List[np.ndarray] = []
    n = min(len(loader), max_episodes) if max_episodes else len(loader)

    for i in range(n):
        ep = loader.get_episode(i)
        all_actions.append(ep.actions)
        proprio = _build_proprio(ep)
        all_proprios.append(proprio)

    actions_cat = np.concatenate(all_actions, axis=0)   # (N_total, D_act)
    proprios_cat = np.concatenate(all_proprios, axis=0) # (N_total, D_proprio)

    return {
        "action_mean": actions_cat.mean(axis=0),
        "action_std":  actions_cat.std(axis=0) + 1e-8,
        "action_min":  actions_cat.min(axis=0),
        "action_max":  actions_cat.max(axis=0),
        "proprio_mean": proprios_cat.mean(axis=0),
        "proprio_std":  proprios_cat.std(axis=0) + 1e-8,
    }


# ---------------------------------------------------------------------------
#  内部工具
# ---------------------------------------------------------------------------

def _build_proprio(ep: EpisodeData) -> np.ndarray:
    """将 joint_states 作为 (T, D) proprio 向量。

    D = D_jst，由实际数据决定：
    * 单臂：D = 12
    * 双臂：D = 14
    """
    return ep.joint_states.astype(np.float32)
