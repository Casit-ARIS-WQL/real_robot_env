"""
json_dataloader.py  ——  episode_with_rewards.json 数据加载器（含 Q-Transformer 适配）

支持格式
--------
加载由 convert_episode.py 生成的行式（row-based）episode JSON 文件，结构如下：

    {
      "episode_id": "ep_000000",
      "task": "<task_label>",
      "milestones": [...],
      "frames": [
        {
          "frame_index": 0,
          "timestamp": 0.0,
          "reward": 0.0,
          "is_terminal": false,
          "observation": {
            "state":       [...],          # 双臂末端位姿，14 维（左右各 7 维：x,y,z,rx,ry,rz,gripper）
            "state_joint": [...],          # 双臂关节状态，14 维（左右各 7 维）
            "images": {
              "right": {"path": "right/frame_000000.jpg", "timestamp": 0.0},
              "top":   {"path": "top/frame_000000.jpg",   "timestamp": 0.0},
              "left":  {"path": "left/frame_000000.jpg",  "timestamp": 0.0},
              "top_2": {"path": "top_2/frame_000000.jpg", "timestamp": 0.0}
            }
          },
          "action": {
            "cartesian": [...],            # 双臂笛卡尔动作，14 维
            "joint":     [...]             # 双臂关节动作，14 维
          }
        },
        ...
      ]
    }

每个 frame 的图像均以独立图像文件（.jpg / .png 等）存储，每帧对应一个文件。
``path`` 字段为相对于 JSON 所在目录（或 base_dir）的路径。

快速入门
--------
    from json_dataloader import load_json_dataset, JsonEpisodeDataLoader

    # 单文件
    ds = load_json_dataset("data/1/episode_with_rewards.json")

    # 二级子目录结构（推荐）：data/1/*.json, data/2/*.json, ...
    # 每个子目录对应一条完整轨迹，子目录内的图像按视角存放于 right/、top/、left/ 等子文件夹。
    ds = load_json_dataset("data/")

    # 扁平结构（兼容旧版）：目录根层级直接存放 .json 文件
    ds = load_json_dataset("/data/episodes/")

    ep = ds.get_episode(0)           # 返回 EpisodeDataWithRewards
    print(ep.rewards)                # (T,) float32 奖励序列
    print(ep.terminals)              # (T,) bool   是否终止帧

    # 支持 DatasetLoader 的所有接口
    for batch in ds.transition_iter(batch_size=64):
        obs, act, rew, next_obs, done = (
            batch["obs_proprio"],
            batch["actions"],
            batch["rewards"],
            batch["next_obs_proprio"],
            batch["dones"],
        )

Q-Transformer 适配入门（单视角）
--------------------------------
    from torch.utils.data import DataLoader
    from json_dataloader import load_qtransformer_dataset

    # 单步（num_timesteps=1）
    qt_ds = load_qtransformer_dataset(
        "episode_with_rewards.json",
        num_frames    = 6,    # 每个状态堆叠的图像帧数，对应 Q-Transformer (C, F, H, W) 中的 F
        image_size    = 224,
        action_bins   = 256,  # 每个动作维度的离散化区间数
        num_actions   = 14,   # 双臂笛卡尔动作全部 14 维；若只使用单臂可传 7
        num_timesteps = 1,    # 1=单步 Q-learning，>1=N-step Q-learning
    )

    # 直接传入 QLearner
    from q_transformer import QRoboticTransformer, QLearner
    model = QRoboticTransformer(vit=..., num_actions=14, action_bins=256)
    learner = QLearner(model, dataset=qt_ds, batch_size=8, num_train_steps=10000,
                       learning_rate=3e-4)
    learner()

    # 或直接用 DataLoader 迭代
    dl = DataLoader(qt_ds, batch_size=4, shuffle=True)
    for instruction, state, action, next_state, reward, done in dl:
        # instruction : List[str]               长度=batch_size
        # state       : FloatTensor(B,3,F,H,W)  （num_timesteps=1，单视角）
        #               FloatTensor(B,T,3,F,H,W) （num_timesteps=T，单视角）
        # action      : LongTensor(B,A)          A=num_actions
        # next_state  : FloatTensor(B,3,F,H,W)
        # reward      : FloatTensor(B,)          （num_timesteps=1）
        #               FloatTensor(B,T)          （num_timesteps=T）
        # done        : BoolTensor(B,)           （num_timesteps=1）
        #               BoolTensor(B,T)           （num_timesteps=T）
        ...

Q-Transformer 适配入门（多视角）
--------------------------------
    qt_ds = load_qtransformer_dataset(
        "episode_with_rewards.json",
        image_keys    = ["right", "top", "left"],  # 指定多个相机视角
        num_frames    = 6,
        image_size    = 224,
        action_bins   = 256,
        num_actions   = 14,
        num_timesteps = 1,
    )

    # 多视角时各视角图像沿通道维度拼接，C = V*3
    # state       : FloatTensor(B, V*3, F, H, W)  （num_timesteps=1）
    #               FloatTensor(B, T, V*3, F, H, W) （num_timesteps=T）
    # next_state  : FloatTensor(B, V*3, F, H, W)

    # 构建多视角模型时，需传入 num_cameras=V
    model = QRoboticTransformer(vit=..., num_actions=14, action_bins=256, num_cameras=3)
    learner = QLearner(model, dataset=qt_ds, batch_size=8, num_train_steps=10000,
                       learning_rate=3e-4)
    learner()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# 复用公共抽象和工具
from dataset_utils import DatasetLoader, EpisodeData, _build_proprio

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  带奖励的 Episode 数据容器
# ---------------------------------------------------------------------------

@dataclass
class EpisodeDataWithRewards(EpisodeData):
    """在 EpisodeData 基础上新增奖励和终止标志序列。

    额外字段
    --------
    rewards   : (T,)  float32  —— 每步奖励
    terminals : (T,)  bool     —— 每步是否为 episode 的终止帧
    timestamps: (T,)  float64  —— 每步时间戳（秒）
    """
    rewards:    np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    terminals:  np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))


# ---------------------------------------------------------------------------
#  JSON Episode 数据集加载器
# ---------------------------------------------------------------------------

class JsonEpisodeDataLoader(DatasetLoader):
    """加载行式 episode JSON 文件（episode_with_rewards.json 格式）。

    参数
    ----
    path : str
        * 若指向单个 .json 文件，则视为一条 episode。
        * 若指向目录，则按以下优先级扫描 JSON 文件：

          1. **二级子目录结构**（推荐）：``data/1/*.json``、``data/2/*.json`` 等。
             每个子目录代表一条完整轨迹，子目录名可为任意字符串（如 "1"、"2"、"3"）。
             子目录内的图像按视角存放于 ``right/``、``top/``、``left/`` 等子文件夹中。
          2. **扁平结构**（兼容旧版）：直接扫描目录下所有 ``*.json`` 文件。
             若目录根层级存在 ``.json`` 文件，则优先使用此模式。

    image_key : str, 可选
        使用哪个相机视角的图像，默认 "right"。
        可选值: "right" | "top" | "left" | "top_2"。
        当 ``image_keys`` 已指定时，此参数被忽略。
    image_keys : List[str], 可选
        使用多个相机视角时，传入视角名称列表，例如 ``["right", "top", "left"]``。
        列表中第一个视角作为主视角（存入 ``EpisodeData.images``），其余视角
        存入 ``EpisodeData.extra["images_{key}"]``。
        若未指定，则退回至 ``image_key`` 的单视角模式。
    base_dir : str, 可选
        图像文件的基准目录。若 JSON 中的图像路径为相对路径，则相对于此目录解析。
        默认与 JSON 文件同级目录。
    load_images : bool
        是否从磁盘加载图像文件，默认 True。
        设为 False 时 images 字段填充零数组，可加速只需状态/动作的场景。
    """

    def __init__(
        self,
        path: str,
        image_key: str = "right",
        image_keys: Optional[List[str]] = None,
        base_dir: Optional[str] = None,
        load_images: bool = True,
    ):
        # 优先使用 image_keys（多视角），否则退回单视角 image_key
        if image_keys is not None:
            self._image_keys = list(image_keys)
        else:
            self._image_keys = [image_key]
        self._image_key  = self._image_keys[0]   # 主视角（向后兼容）
        self._load_images = load_images

        p = Path(path)
        if p.is_file():
            self._json_files: List[Path] = [p]
            self._base_dir = Path(base_dir) if base_dir else p.parent
        elif p.is_dir():
            # 优先使用扁平结构（根目录下直接存在 .json 文件），兼容旧版数据集。
            # 若根目录下无 .json 文件，则按二级子目录结构扫描（data/1/*.json，…）。
            flat_files = sorted(p.glob("*.json"))
            if flat_files:
                self._json_files = flat_files
            else:
                # 二级子目录结构：每个子目录对应一条轨迹，收集所有子目录内的 .json 文件
                self._json_files = sorted(p.glob("*/*.json"))
            if not self._json_files:
                raise FileNotFoundError(
                    f"目录中未找到任何 .json 文件（已尝试根目录及一级子目录）: {path}"
                )
            self._base_dir = Path(base_dir) if base_dir else p
        else:
            raise FileNotFoundError(f"路径不存在: {path}")

        logger.info(
            "JsonEpisodeDataLoader 初始化: %s  (%d 条轨迹, image_keys=%s)",
            path, len(self._json_files), self._image_keys,
        )

    # ------------------------------------------------------------------
    #  DatasetLoader 接口
    # ------------------------------------------------------------------

    @property
    def image_keys(self) -> List[str]:
        """已加载的相机视角名称列表（只读）。"""
        return self._image_keys

    def __len__(self) -> int:
        return len(self._json_files)

    def get_episode(self, idx: int) -> EpisodeDataWithRewards:
        """按索引加载并解析一条 episode。"""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"episode 索引越界: {idx}，共 {len(self)} 条")

        json_path = self._json_files[idx]
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._parse_episode(data, json_path)

    def transition_iter(
        self,
        batch_size: int = 256,
        shuffle: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterator[Dict[str, np.ndarray]]:
        """迭代 (s, a, r, s', done) 转换对，包含奖励信息。

        每次 yield 字典，key 说明
        ---------------------------
        obs_image        : (B, H, W, 3)  uint8
        obs_proprio      : (B, D)        float32   —— 关节状态（双臂 D=14，单臂 D=12）
        actions          : (B, 14)       float32   —— 双臂笛卡尔动作全维度
        rewards          : (B,)          float32
        next_obs_image   : (B, H, W, 3)  uint8
        next_obs_proprio : (B, D)        float32
        dones            : (B,)          float32   episode 末步=1
        """
        rng = rng or np.random.default_rng()
        pool = self._build_transition_pool_with_rewards()
        indices = np.arange(len(pool["actions"]))
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start: start + batch_size]
            yield {k: v[batch_idx] for k, v in pool.items()}

    # ------------------------------------------------------------------
    #  解析单条 episode
    # ------------------------------------------------------------------

    def _parse_episode(
        self, data: dict, json_path: Path
    ) -> EpisodeDataWithRewards:
        """将 JSON dict 转换为 EpisodeDataWithRewards。"""
        _validate_episode_dict(data, json_path)

        frames: List[dict] = data["frames"]
        T = len(frames)
        task: str = data.get("task", "")

        # 预分配数组
        state_list:       List[np.ndarray] = []
        state_joint_list: List[np.ndarray] = []
        action_cart_list: List[np.ndarray] = []
        action_joint_list: List[np.ndarray] = []
        rewards    = np.zeros(T, dtype=np.float32)
        terminals  = np.zeros(T, dtype=bool)
        timestamps = np.zeros(T, dtype=np.float64)

        # 用于图像加载的路径列表，按视角分组
        # key: 视角名称 → List[img_path]
        all_img_paths: Dict[str, List[str]] = {
            k: [] for k in self._image_keys
        }

        # 解析每一帧
        for i, frame in enumerate(frames):
            obs    = frame["observation"]
            action = frame["action"]

            state_list.append(np.array(obs["state"],       dtype=np.float32))
            state_joint_list.append(np.array(obs["state_joint"], dtype=np.float32))
            action_cart_list.append(np.array(action["cartesian"], dtype=np.float32))
            action_joint_list.append(np.array(action["joint"],    dtype=np.float32))

            rewards[i]    = float(frame.get("reward", 0.0))
            terminals[i]  = bool(frame.get("is_terminal", False))
            timestamps[i] = float(frame.get("timestamp", 0.0))

            # 各视角的图像路径
            img_dict = obs.get("images", {})
            for key in self._image_keys:
                img_info = img_dict.get(key, {})
                if img_info:
                    img_path = self._resolve_path(img_info.get("path", ""), json_path)
                else:
                    img_path = ""
                all_img_paths[key].append(img_path)

        tcp_poses    = np.stack(state_list)             # (T, 14)  双臂末端位姿
        joint_states = np.stack(state_joint_list)       # (T, 14)  双臂关节状态
        actions_cart = np.stack(action_cart_list)       # (T, 14)  双臂笛卡尔动作
        actions_joint = np.stack(action_joint_list)     # (T, 14)  双臂关节动作

        # 将笛卡尔动作作为主动作（与 HDF5/LeRobot 加载器保持一致）
        actions = actions_cart

        # 加载各视角图像
        images_per_view: Dict[str, np.ndarray] = {}
        for key in self._image_keys:
            paths = all_img_paths[key]
            if self._load_images and paths:
                images_per_view[key] = _load_images_from_paths(paths, T)
            else:
                images_per_view[key] = np.zeros(
                    (T, _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, 3), dtype=np.uint8
                )

        # 主视角图像（向后兼容）
        primary_key = self._image_keys[0]
        images = images_per_view[primary_key]

        # 额外视角存入 extra（供多视角 Dataset 使用）
        extra: Dict[str, np.ndarray] = {
            "actions_joint": actions_joint,
            "actions_cart":  actions_cart,
        }
        for key in self._image_keys[1:]:
            extra[f"images_{key}"] = images_per_view[key]

        return EpisodeDataWithRewards(
            images=images,
            tcp_poses=tcp_poses,
            joint_states=joint_states,
            force_torques=np.zeros((T, 6), dtype=np.float32),
            actions=actions,
            language=task,
            extra=extra,
            rewards=rewards,
            terminals=terminals,
            timestamps=timestamps,
        )

    # ------------------------------------------------------------------
    #  转换对池（包含奖励）
    # ------------------------------------------------------------------

    def _build_transition_pool_with_rewards(self) -> Dict[str, np.ndarray]:
        """将全部 episode 展开为带奖励的转换对池。"""
        pools: Dict[str, List[np.ndarray]] = {
            k: [] for k in (
                "obs_image", "obs_proprio",
                "actions",
                "rewards",
                "next_obs_image", "next_obs_proprio",
                "dones",
            )
        }
        for ep in self:
            assert isinstance(ep, EpisodeDataWithRewards)
            T      = ep.length
            proprio = _build_proprio(ep)           # (T, 14) 双臂关节状态
            for t in range(T):
                t_next = min(t + 1, T - 1)
                pools["obs_image"].append(ep.images[t])
                pools["obs_proprio"].append(proprio[t])
                pools["actions"].append(ep.actions[t])
                pools["rewards"].append(np.float32(ep.rewards[t]))
                pools["next_obs_image"].append(ep.images[t_next])
                pools["next_obs_proprio"].append(proprio[t_next])
                pools["dones"].append(np.float32(ep.terminals[t]))

        return {k: np.stack(v) for k, v in pools.items()}

    # ------------------------------------------------------------------
    #  路径解析
    # ------------------------------------------------------------------

    def _resolve_path(self, rel_path: str, json_path: Path) -> str:
        """将 JSON 中记录的（可能为相对）路径解析为绝对路径。"""
        if not rel_path:
            return ""
        p = Path(rel_path)
        if p.is_absolute():
            return str(p)
        # 优先尝试相对于 json 文件所在目录
        candidate = json_path.parent / p
        if candidate.exists():
            return str(candidate)
        # 再尝试相对于 base_dir
        candidate2 = self._base_dir / p
        if candidate2.exists():
            return str(candidate2)
        # 返回相对于 base_dir 的路径（即使不存在，留给调用方处理）
        return str(self._base_dir / p)


# ---------------------------------------------------------------------------
#  工厂函数
# ---------------------------------------------------------------------------

def load_json_dataset(
    path: str,
    image_key: str = "right",
    base_dir: Optional[str] = None,
    load_images: bool = True,
) -> JsonEpisodeDataLoader:
    """加载 episode_with_rewards.json 格式数据集的快捷入口。

    Parameters
    ----------
    path        : 单个 .json 文件路径，或包含多个 .json 文件的目录路径。
    image_key   : 使用的相机视角，默认 "right"。
    base_dir    : 图像文件基准目录，默认与 json 同级目录。
    load_images : 是否加载图像文件，默认 True。

    Returns
    -------
    JsonEpisodeDataLoader 实例
    """
    return JsonEpisodeDataLoader(
        path=path,
        image_key=image_key,
        base_dir=base_dir,
        load_images=load_images,
    )


# ---------------------------------------------------------------------------
#  内部工具
# ---------------------------------------------------------------------------

def _validate_episode_dict(data: dict, source: Path) -> None:
    """检查 episode JSON 是否包含必要字段。"""
    required = {"frames"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(
            f"episode JSON 缺少必要字段 {missing}，来源: {source}"
        )
    if not isinstance(data["frames"], list) or len(data["frames"]) == 0:
        raise ValueError(f"episode JSON frames 为空，来源: {source}")

    # 检查第一帧结构
    first = data["frames"][0]
    frame_required = {"observation", "action"}
    missing_frame = frame_required - set(first.keys())
    if missing_frame:
        raise ValueError(
            f"episode JSON frame 缺少字段 {missing_frame}，来源: {source}"
        )


def _load_images_from_paths(
    img_paths: List[str],
    T: int,
    default_hw: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """从图像文件路径列表中批量加载已分帧的图像。

    每个路径对应一帧独立图像文件（.jpg / .png / .bmp / .webp 等）。

    Parameters
    ----------
    img_paths  : 每帧对应的图像文件路径列表，长度应与 T 相同。
    T          : 帧数，应与 len(img_paths) 相同。
    default_hw : 加载失败时使用的默认分辨率 (H, W)。

    Returns
    -------
    images : (T, H, W, 3) uint8
    """
    H, W = default_hw
    images = np.zeros((T, H, W, 3), dtype=np.uint8)

    for i, fpath in enumerate(img_paths):
        if not fpath:
            continue
        img = _load_single_image(fpath, (H, W))
        if img is not None:
            fh, fw = img.shape[:2]
            if images.shape[1] != fh or images.shape[2] != fw:
                if i == 0:
                    images = np.zeros((T, fh, fw, 3), dtype=np.uint8)
                else:
                    oh, ow = images.shape[1], images.shape[2]
                    mh, mw = min(fh, oh), min(fw, ow)
                    tmp = np.zeros((oh, ow, 3), dtype=np.uint8)
                    tmp[:mh, :mw] = img[:mh, :mw]
                    img = tmp
            images[i] = img

    return images


# ---------------------------------------------------------------------------
#  OpenCV 懒加载（避免强依赖 cv2）
# ---------------------------------------------------------------------------

_DEFAULT_IMAGE_SIZE = 224  # 图像加载失败时使用的默认边长（像素）


def _load_single_image(path: str, default_hw: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """从磁盘加载单张图像文件，返回 (H, W, 3) uint8 RGB 数组，失败时返回 None。

    优先使用 cv2，否则退回 PIL。
    """
    try:
        import cv2
        img = cv2.imread(path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logger.warning("cv2.imread 无法读取图像: %s", path)
    except ImportError:
        pass

    try:
        from PIL import Image as _PILImage
        img = np.array(_PILImage.open(path).convert("RGB"), dtype=np.uint8)
        return img
    except Exception:
        pass

    logger.warning("无法读取图像文件: %s，将使用零图像", path)
    return None


# ---------------------------------------------------------------------------
#  Q-Transformer 适配层
# ---------------------------------------------------------------------------

def compute_action_stats(
    episodes: List[EpisodeDataWithRewards],
    num_actions: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算所有 episode 中动作各维度的 min / max，用于离散化。

    Parameters
    ----------
    episodes    : EpisodeDataWithRewards 列表（由 JsonEpisodeDataLoader 加载）
    num_actions : 取动作的前 N 维。None 表示使用全部维度。

    Returns
    -------
    action_min : (A,) float32
    action_max : (A,) float32
    """
    all_actions = np.concatenate([ep.actions for ep in episodes], axis=0)  # (N_total, A_full)
    if num_actions is not None:
        all_actions = all_actions[:, :num_actions]
    action_min = all_actions.min(axis=0).astype(np.float32)
    action_max = all_actions.max(axis=0).astype(np.float32)
    # 防止零区间导致除零，用一个小 epsilon 扩展上界
    _ACTION_RANGE_EPS = 1e-6
    zero_range = (action_max - action_min) == 0
    action_max[zero_range] = action_min[zero_range] + _ACTION_RANGE_EPS
    return action_min, action_max


def discretize_actions(
    actions: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
    action_bins: int,
) -> np.ndarray:
    """将连续动作向量线性映射到离散 bin 索引。

    Parameters
    ----------
    actions    : (..., A) float32  连续动作
    action_min : (A,)    float32  各维最小值
    action_max : (A,)    float32  各维最大值
    action_bins: int              每维离散区间数

    Returns
    -------
    bin_indices : (..., A) int64  取值范围 [0, action_bins-1]
    """
    normed = (actions - action_min) / (action_max - action_min)  # [..., A] in [0,1]
    bins = np.round(normed * (action_bins - 1)).astype(np.int64)
    return np.clip(bins, 0, action_bins - 1)


def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """将 (H, W, 3) uint8 图像缩放至 (size, size, 3)。

    优先使用 cv2，否则退回 PIL。若两者均不可用则直接 reshape（最近邻，仅限整数倍缩放）。
    """
    if img.shape[0] == size and img.shape[1] == size:
        return img
    try:
        import cv2
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        pass
    try:
        from PIL import Image
        return np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR), dtype=np.uint8)
    except ImportError:
        pass
    # 最后手段：简单 numpy 最近邻缩放
    h, w = img.shape[:2]
    row_idx = (np.arange(size) * h / size).astype(int)
    col_idx = (np.arange(size) * w / size).astype(int)
    return img[np.ix_(row_idx, col_idx)]


class QTransformerEpisodeDataset(Dataset):
    """PyTorch ``Dataset`` 适配层，将 JSON episode 数据转换为 Q-Transformer 所需格式。

    每条样本返回元组::

        (instruction, state, action, next_state, reward, done)

    单视角、单步（num_timesteps=1）维度说明
    ----------------------------------------
    instruction : str                              任务语言描述
    state       : FloatTensor(3, F, H, W)          视频状态（F 帧，归一化到 [0,1]）
    action      : LongTensor(A,)                   离散化动作 bin 索引
    next_state  : FloatTensor(3, F, H, W)
    reward      : FloatTensor()                    标量奖励
    done        : BoolTensor()                     是否终止

    多视角（V 个视角）、单步维度说明
    ----------------------------------
    state       : FloatTensor(V*3, F, H, W)        各视角图像沿通道维拼接
    next_state  : FloatTensor(V*3, F, H, W)

    N 步（num_timesteps=N>1）维度说明
    -----------------------------------
    state       : FloatTensor(T, 3, F, H, W)       单视角
                  FloatTensor(T, V*3, F, H, W)      多视角
    next_state  : FloatTensor(3, F, H, W)           单视角
                  FloatTensor(V*3, F, H, W)          多视角

    Parameters
    ----------
    episodes     : List[EpisodeDataWithRewards]
        由 ``JsonEpisodeDataLoader`` 加载的 episode 列表。
    action_min   : np.ndarray (A,)
        每个动作维度的最小值，用于线性离散化。
    action_max   : np.ndarray (A,)
        每个动作维度的最大值，用于线性离散化。
    action_bins  : int
        每个动作维度的离散区间数，默认 256。
    num_actions  : int, 可选
        使用笛卡尔动作的前 N 维。None 表示使用全部维度（双臂数据为 14 维）。
        若只训练单臂策略，可传 7 以仅使用前 7 维。
    num_frames   : int
        每个状态帧窗口大小 F，默认 1。
    image_size   : int
        图像边长（正方形），默认 224。
    num_timesteps: int
        N-step Q-learning 的步数，1 = 单步。
    image_keys   : List[str], 可选
        多视角训练时指定的视角名称列表（与 ``JsonEpisodeDataLoader`` 中保持一致）。
        None 或单元素列表表示单视角模式（向后兼容）。
        多视角时各视角图像沿通道维度拼接：C = V * 3。
    """

    def __init__(
        self,
        episodes: List[EpisodeDataWithRewards],
        action_min: np.ndarray,
        action_max: np.ndarray,
        *,
        action_bins: int = 256,
        num_actions: Optional[int] = None,
        num_frames: int = 1,
        image_size: int = 224,
        num_timesteps: int = 1,
        image_keys: Optional[List[str]] = None,
    ):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("QTransformerEpisodeDataset 需要 PyTorch: pip install torch")

        if num_timesteps < 1:
            raise ValueError(f"num_timesteps 必须 >= 1，当前值: {num_timesteps}")

        self.episodes     = episodes
        self.action_min   = action_min.astype(np.float32)
        self.action_max   = action_max.astype(np.float32)
        self.action_bins  = action_bins
        self.num_actions  = num_actions
        self.num_frames   = num_frames
        self.image_size   = image_size
        self.num_timesteps = num_timesteps

        # 多视角支持：None 或单元素列表 → 单视角模式
        self.image_keys = list(image_keys) if image_keys else None
        self.num_cameras = len(self.image_keys) if self.image_keys else 1

        # 构建全局索引表：(episode_idx, timestep_start)
        # 有效起点：0 ≤ t ≤ T - num_timesteps（确保 t+num_timesteps 有 next_state）
        self._index: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(episodes):
            T = ep.length
            max_start = T - num_timesteps  # 包含
            if max_start >= 0:
                for t in range(max_start + 1):
                    self._index.append((ep_idx, t))

    # ------------------------------------------------------------------
    #  Dataset 协议
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        import torch

        ep_idx, t_start = self._index[idx]
        ep = self.episodes[ep_idx]
        T  = ep.length
        N  = self.num_timesteps

        # ---- 动作离散化 ----
        acts = ep.actions                                             # (T, A_full)
        if self.num_actions is not None:
            acts = acts[:, :self.num_actions]                        # (T, A)
        disc_actions = discretize_actions(
            acts, self.action_min, self.action_max, self.action_bins
        )                                                             # (T, A)

        # ---- 组装 state / next_state ----
        # 单视角: (N, 3, F, H, W)；多视角: (N, V*3, F, H, W)
        state_seq = self._build_state_seq(ep, t_start, N)
        t_next    = min(t_start + N, T - 1)
        next_state = self._build_frame_window(ep, t_next)

        # ---- 奖励 / 终止 ----
        t_end = min(t_start + N, T)
        rewards   = ep.rewards[t_start:t_end].astype(np.float32)     # (N,)
        terminals = ep.terminals[t_start:t_end]                      # (N,) bool

        # 不足 N 步时补零（episode 末尾）
        if len(rewards) < N:
            pad = N - len(rewards)
            rewards   = np.concatenate([rewards,   np.zeros(pad, dtype=np.float32)])
            terminals = np.concatenate([terminals, np.ones(pad, dtype=bool)])
            action_seq = disc_actions[t_start:t_end]                 # (<=N, A)
            action_seq = np.concatenate([
                action_seq,
                np.zeros((pad, action_seq.shape[1]), dtype=np.int64)
            ], axis=0)                                                # (N, A)
        else:
            action_seq = disc_actions[t_start:t_start + N]           # (N, A)

        # ---- 转换为 Tensor ----
        state_tensor      = torch.from_numpy(state_seq)
        next_state_tensor = torch.from_numpy(next_state)
        action_tensor     = torch.from_numpy(action_seq).long()      # (N, A)
        reward_tensor     = torch.from_numpy(rewards)                # (N,)
        done_tensor       = torch.from_numpy(terminals)              # (N,) bool

        # 单步时 squeeze 掉时间维度，与 MockReplayDataset 对齐
        if N == 1:
            state_tensor  = state_tensor.squeeze(0)                  # (C, F, H, W)
            action_tensor = action_tensor.squeeze(0)                 # (A,)
            reward_tensor = reward_tensor.squeeze(0)                 # scalar
            done_tensor   = done_tensor.squeeze(0)                   # scalar bool

        return ep.language, state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor

    # ------------------------------------------------------------------
    #  内部工具
    # ------------------------------------------------------------------

    def _build_single_view_window(
        self,
        images: np.ndarray,
        ep_length: int,
        t: int,
    ) -> np.ndarray:
        """从图像序列中提取以 t 为终点的 num_frames 帧历史窗口，返回 (3, F, H, W) float32 in [0,1]。

        窗口对应公式中的 s_{t-w:t}，即包含 t 在内向过去延伸 F 帧的状态历史。
        f=0 为最旧帧（max(0, t-F+1)），f=F-1 为当前帧 t。

        Parameters
        ----------
        images    : (T, H, W, 3) uint8  某一视角的图像序列
        ep_length : episode 帧数 T
        t         : 当前帧索引（窗口终点）
        """
        F  = self.num_frames
        sz = self.image_size
        out = np.zeros((3, F, sz, sz), dtype=np.float32)

        for f in range(F):
            # f=0 → 最旧帧 t-(F-1)，f=F-1 → 当前帧 t；不足时用第一帧填充
            ti = max(t - (F - 1 - f), 0)
            img = images[ti]                                         # (H, W, 3) uint8
            if img.shape[0] != sz or img.shape[1] != sz:
                img = _resize_image(img, sz)
            out[:, f, :, :] = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        return out  # (3, F, H, W)

    def _build_frame_window(self, ep: EpisodeDataWithRewards, t: int) -> np.ndarray:
        """构建单个时间步的图像窗口。

        单视角返回 (3, F, H, W)；多视角返回 (V*3, F, H, W)（通道维度拼接）。
        """
        T = ep.length

        if self.image_keys is None or self.num_cameras == 1:
            # 单视角
            return self._build_single_view_window(ep.images, T, t)

        # 多视角：依次读取各视角，沿通道维度拼接
        view_frames = []
        for i, key in enumerate(self.image_keys):
            if i == 0:
                img_seq = ep.images
            else:
                img_seq = ep.extra.get(f"images_{key}")
                if img_seq is None:
                    # 找不到该视角时退回主视角（避免崩溃）
                    logger.warning(
                        "episode 中未找到视角 '%s'，将使用主视角替代", key
                    )
                    img_seq = ep.images
            view_frames.append(self._build_single_view_window(img_seq, T, t))
        # view_frames: List[(3, F, H, W)] → concat → (V*3, F, H, W)
        return np.concatenate(view_frames, axis=0)

    def _build_state_seq(
        self, ep: EpisodeDataWithRewards, t_start: int, N: int
    ) -> np.ndarray:
        """构建 N 步状态序列。

        单视角返回 (N, 3, F, H, W)；多视角返回 (N, V*3, F, H, W)。
        """
        frames = [self._build_frame_window(ep, t_start + n) for n in range(N)]
        return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
#  Q-Transformer 工厂函数
# ---------------------------------------------------------------------------

def load_qtransformer_dataset(
    path: str,
    *,
    image_key: str = "right",
    image_keys: Optional[List[str]] = None,
    base_dir: Optional[str] = None,
    num_frames: int = 1,
    image_size: int = 224,
    action_bins: int = 256,
    num_actions: Optional[int] = None,
    num_timesteps: int = 1,
    action_min: Optional[np.ndarray] = None,
    action_max: Optional[np.ndarray] = None,
) -> QTransformerEpisodeDataset:
    """加载 JSON episode 数据并封装为 Q-Transformer 兼容的 PyTorch Dataset。

    Parameters
    ----------
    path          : 单个 .json 文件或目录路径。
    image_key     : 单视角模式下使用的相机视角，默认 "right"。
                    当 ``image_keys`` 已指定时，此参数被忽略。
    image_keys    : 多视角模式下使用的视角名称列表，例如 ``["right", "top", "left"]``。
                    指定后将加载所有列出视角并沿通道维度拼接（C = V*3）。
                    相应地，``QRoboticTransformer`` 须设置 ``num_cameras=V``。
    base_dir      : 视频文件基准目录。
    num_frames    : 每个状态堆叠的图像帧数 F，默认 1（即单帧）。
    image_size    : 图像边长（正方形），默认 224。
    action_bins   : 每个动作维度的离散区间数，默认 256。
    num_actions   : 使用笛卡尔动作前 N 维。None 表示全部（双臂为 14 维）。
    num_timesteps : N-step Q-learning 的步数，1 = 单步 Q-learning。
    action_min    : 手动指定动作下界 (A,)；None 时从数据自动计算。
    action_max    : 手动指定动作上界 (A,)；None 时从数据自动计算。

    Returns
    -------
    QTransformerEpisodeDataset  （torch.utils.data.Dataset 子类）
    """
    loader = JsonEpisodeDataLoader(
        path=path,
        image_key=image_key,
        image_keys=image_keys,
        base_dir=base_dir,
        load_images=True,
    )

    episodes: List[EpisodeDataWithRewards] = [loader.get_episode(i) for i in range(len(loader))]

    if action_min is None or action_max is None:
        a_min, a_max = compute_action_stats(episodes, num_actions=num_actions)
        if action_min is None:
            action_min = a_min
        if action_max is None:
            action_max = a_max

    # 将 image_keys 传给 Dataset，以便 _build_frame_window 知道需要哪些视角
    effective_image_keys = loader.image_keys if len(loader.image_keys) > 1 else None

    return QTransformerEpisodeDataset(
        episodes=episodes,
        action_min=action_min,
        action_max=action_max,
        action_bins=action_bins,
        num_actions=num_actions,
        num_frames=num_frames,
        image_size=image_size,
        num_timesteps=num_timesteps,
        image_keys=effective_image_keys,
    )