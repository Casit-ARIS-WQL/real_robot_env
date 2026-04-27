"""
train_real_robot_td3.py  ——  真机 Residual TD3 在线 / 离线强化学习训练脚本

算法概述
--------
Residual TD3：在 VLA 基础策略（OpenVLA 等）输出之上学习一个残差策略。
    combined_action = base_action + residual_action

支持三种训练模式（通过 cfg.algo.offline_fraction 控制）
    纯离线  (offline_fraction = 1.0) ：仅使用 VLA 数据集
    纯在线  (offline_fraction = 0.0) ：仅使用真机交互数据
    混合    (0 < offline_fraction < 1)：两者混合（推荐）

快速启动（示例）
----------------
    from config_real_robot_td3 import ResidualTD3RealRobotConfig
    from train_real_robot_td3 import main

    cfg = ResidualTD3RealRobotConfig()
    cfg.task_language = "pick up the red block"
    cfg.vla_model_path = "/path/to/openvla"
    cfg.offline_data.dataset_path = "/data/demos.h5"
    cfg.algo.offline_fraction = 0.5
    main(cfg)

依赖
----
    resfit          （QAgent、MultiStepTransform 等 RL 工具）
    torchrl         （TensorDictPrioritizedReplayBuffer）
    tensordict      （TensorDict）
    wandb           （日志）
    env.py          （RealRobotEnv）
    vla_interface.py（VLAModelWrapper / OpenVLAWrapper）
    dataset_utils.py（DatasetLoader / load_dataset）
    real_robot_env_wrapper.py（RealRobotBasePolicyEnvWrapper）
    config_real_robot_td3.py  （ResidualTD3RealRobotConfig）
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pprint
import random
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from tqdm import tqdm

# ── RL 工具（来自 resfit）────────────────────────────────────────────────────
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform

# 可选：replay buffer 持久化 & HuggingFace 上传
try:
    from resfit.rl_finetuning.utils.hugging_face import (
        _hf_download_buffer,
        _hf_upload_buffer,
        optimized_replay_buffer_dumps,
        optimized_replay_buffer_loads,
    )

    _HF_UTILS_AVAILABLE = True
except ImportError:
    _HF_UTILS_AVAILABLE = False

    def _hf_download_buffer(*a, **kw):
        return None

    def _hf_upload_buffer(*a, **kw):
        pass

    def optimized_replay_buffer_dumps(rb, path):
        torch.save(rb.state_dict(), path / "rb.pt")

    def optimized_replay_buffer_loads(rb, path):
        rb.load_state_dict(torch.load(path / "rb.pt"))


import wandb

# ── 本地模块 ────────────────────────────────────────────────────────────────
from config_real_robot_td3 import ResidualTD3RealRobotConfig
from dataset_utils import DatasetLoader, EpisodeData, load_dataset
from env import EnvMode, RealRobotEnv
from real_robot_env_wrapper import RealRobotBasePolicyEnvWrapper
from vla_interface import OpenVLAWrapper, VLAModelWrapper

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── HuggingFace buffer 缓存仓库（从环境变量读取，可不配置）────────────────
OFFLINE_HF_REPO = os.environ.get("HF_OFFLINE_BUFFER_REPO", None)
ONLINE_HF_REPO = os.environ.get("HF_ONLINE_BUFFER_REPO", None)

_CACHE_ROOT = Path(os.environ.get("CACHE_DIR", ".")).expanduser().resolve()
OFFLINE_CACHE_DIR = _CACHE_ROOT / "offline_buffer_cache"
ONLINE_CACHE_DIR = _CACHE_ROOT / "online_buffer_cache"


# ===========================================================================
#  计时工具
# ===========================================================================


class TrainingTimer:
    """各训练阶段耗时统计（百分比 & 均值 ms）。"""

    def __init__(self):
        self.times = defaultdict(list)
        self.reset_time = time.perf_counter()

    @contextmanager
    def time(self, stage_name: str):
        start = time.perf_counter()
        yield
        self.times[stage_name].append(time.perf_counter() - start)

    def get_timing_stats(self) -> dict:
        if not self.times:
            return {}
        total = sum(sum(v) for v in self.times.values())
        if total == 0:
            return {}
        stats = {}
        for name, times_list in self.times.items():
            s = sum(times_list)
            avg = s / len(times_list)
            stats[f"timing/{name}_percentage"] = (s / total) * 100
            stats[f"timing/{name}_avg_ms"] = avg * 1000
            stats[f"timing/{name}_total_s"] = s
        return stats

    def reset(self):
        self.times = defaultdict(list)
        self.reset_time = time.perf_counter()


# ===========================================================================
#  状态标准化 & 动作缩放（轻量实现，不依赖 resfit）
# ===========================================================================


class SimpleStateStandardizer:
    """
    对本体感觉状态做零均值、单位方差标准化。

    在 replay buffer 存储前和传入 QAgent 前统一调用，保证一致性。
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, device: str = "cpu"):
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(std, dtype=torch.float32, device=device)

    def standardize(self, state: torch.Tensor) -> torch.Tensor:
        return (state.to(self.mean.device) - self.mean) / (self.std + 1e-8)

    @classmethod
    def from_dataset(
        cls,
        dataset_loader: DatasetLoader,
        min_std: float = 0.01,
        device: str = "cpu",
    ) -> "SimpleStateStandardizer":
        """从离线数据集计算均值和标准差。"""
        all_proprio: List[np.ndarray] = []
        for ep in dataset_loader:
            all_proprio.append(ep.joint_states)
        states = np.concatenate(all_proprio, axis=0)
        mean = states.mean(axis=0).astype(np.float32)
        std = np.maximum(states.std(axis=0), min_std).astype(np.float32)
        return cls(mean, std, device)

    @classmethod
    def identity(cls, dim: int = 24, device: str = "cpu") -> "SimpleStateStandardizer":
        """恒等变换（不做标准化）。"""
        return cls(np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32), device)


class SimpleActionScaler:
    """
    将动作从原始范围 [low, high] 双向缩放至 [-1, 1]。

    VLA 输出通常已在 [-1, 1]，可使用 identity() 跳过缩放。
    """

    def __init__(self, low: np.ndarray, high: np.ndarray, device: str = "cpu"):
        low_t = torch.tensor(low, dtype=torch.float32, device=device)
        high_t = torch.tensor(high, dtype=torch.float32, device=device)
        self.bias = (high_t + low_t) / 2
        self.scale_factor = torch.clamp((high_t - low_t) / 2, min=1e-8)

    def scale(self, action: torch.Tensor) -> torch.Tensor:
        """原始动作 → [-1, 1]"""
        return (action.to(self.bias.device) - self.bias) / self.scale_factor

    def unscale(self, action: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → 原始动作"""
        return action.to(self.bias.device) * self.scale_factor + self.bias

    @classmethod
    def from_dataset(
        cls,
        dataset_loader: DatasetLoader,
        min_range: float = 0.01,
        percentile: float = 1.0,
        device: str = "cpu",
    ) -> "SimpleActionScaler":
        """从离线数据集估计动作范围（去除尾部异常值）。"""
        all_actions: List[np.ndarray] = []
        for ep in dataset_loader:
            all_actions.append(ep.actions)
        actions = np.concatenate(all_actions, axis=0)
        low = np.percentile(actions, percentile, axis=0).astype(np.float32)
        high = np.percentile(actions, 100 - percentile, axis=0).astype(np.float32)
        mid = (low + high) / 2
        half_range = np.maximum((high - low) / 2, min_range / 2)
        return cls(mid - half_range, mid + half_range, device)

    @classmethod
    def identity(cls, dim: int = 7, device: str = "cpu") -> "SimpleActionScaler":
        """恒等缩放（动作已在 [-1, 1]，无需变换）。"""
        return cls(
            np.full(dim, -1.0, dtype=np.float32),
            np.full(dim, 1.0, dtype=np.float32),
            device,
        )


# ===========================================================================
#  Replay Buffer 工具
# ===========================================================================


def _to_uint8_inplace(obs_dict: Dict, image_keys: List[str]) -> None:
    """将观测字典中的浮点图像原地转换为 uint8，节省存储空间。"""
    for k in image_keys:
        if k in obs_dict:
            v = obs_dict[k]
            if isinstance(v, torch.Tensor) and v.dtype != torch.uint8:
                obs_dict[k] = (v * 255.0).clamp(0, 255).to(torch.uint8)


def _add_transitions_to_buffer(
    *,
    obs: Dict,
    next_obs: Dict,
    actions: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    info: Dict,
    device: torch.device,
    image_keys: List[str],
    lowdim_keys: List[str],
    num_envs: int,
    online_rb: TensorDictPrioritizedReplayBuffer,
) -> None:
    """
    将当前步的转换对添加到 online replay buffer。

    正确处理 episode 终止时的 final_obs（terminal state），
    确保 bootstrapping 使用真实的终止状态。
    """
    obs_keys_set = set(image_keys) | set(lowdim_keys)

    for i in range(num_envs):
        # terminal state 使用 final_obs（真实终止时刻的观测）
        if done[i] and "final_obs" in info and info["final_obs"][i] is not None:
            final_obs_dict = info["final_obs"][i]
            next_obs_i = {
                k: torch.as_tensor(v, device=device)
                for k, v in final_obs_dict.items()
            }
        else:
            next_obs_i = {k: v[i] for k, v in next_obs.items()}

        curr_obs_i = {k: v[i] for k, v in obs.items()}

        # 只保留 RL 相关的 key，图像转 uint8
        curr_obs_i = {k: v for k, v in curr_obs_i.items() if k in obs_keys_set}
        next_obs_i = {k: v for k, v in next_obs_i.items() if k in obs_keys_set}
        _to_uint8_inplace(curr_obs_i, image_keys)
        _to_uint8_inplace(next_obs_i, image_keys)

        td = TensorDict(
            {
                "obs": TensorDict(curr_obs_i, batch_size=[]),
                "next": TensorDict(
                    {
                        "obs": TensorDict(next_obs_i, batch_size=[]),
                        "done": done[i],
                        "reward": reward[i],
                    },
                    batch_size=[],
                ),
                "action": actions[i],
                "_priority": torch.tensor(10.0, dtype=torch.float32),
            },
            batch_size=[],
        ).unsqueeze(0)

        online_rb.add(td)


# ===========================================================================
#  离线 Buffer 填充
# ===========================================================================


def _populate_offline_buffer(
    dataset_loader: DatasetLoader,
    rb: TensorDictPrioritizedReplayBuffer,
    image_keys: List[str],
    camera_key: str,
    action_scaler: SimpleActionScaler,
    state_standardizer: SimpleStateStandardizer,
    lang_embed: np.ndarray,
    num_episodes: Optional[int] = None,
    use_base_policy: bool = False,
    base_policy: Optional[VLAModelWrapper] = None,
) -> int:
    """
    将 VLA 离线数据集转换为 Residual TD3 格式，填充 offline replay buffer。

    每条 episode 的 (t, t+1) 帧对构成一条 transition，转换规则：
        obs.image (H,W,C) uint8       → <camera_key>: (C,H,W) uint8 Tensor
        obs.proprio / tcp_pose + joint → observation.state: (24,) float32 (标准化)
        base_action                   → observation.base_action: (7,) float32
        gt_action（缩放后）            → action
        reward = float(done_flag)     → next.reward（稀疏奖励：episode 末步=1）

    Parameters
    ----------
    dataset_loader  : DatasetLoader 实例
    rb              : 目标离线 replay buffer
    image_keys      : 图像键名列表（如 ["observation.images.top"]）
    camera_key      : 图像键名（与 image_keys[0] 相同）
    action_scaler   : 动作缩放器
    state_standardizer : 状态标准化器
    lang_embed      : 任务语言嵌入（用于 VLA 推断 base_action）
    num_episodes    : 加载的最大 episode 数（None=全部）
    use_base_policy : 是否用 VLA 模型推断 base_action（False=用 GT 动作作为 base）
    base_policy     : VLAModelWrapper 实例（use_base_policy=True 时必须提供）

    Returns
    -------
    transitions : 添加的转换对数量
    """
    if use_base_policy and base_policy is None:
        raise ValueError("use_base_policy=True 时必须提供 base_policy")

    n_episodes = len(dataset_loader)
    if num_episodes is not None:
        n_episodes = min(n_episodes, num_episodes)

    transitions = 0
    for ep_idx in tqdm(range(n_episodes), desc="Filling offline buffer"):
        ep: EpisodeData = dataset_loader.get_episode(ep_idx)
        T = ep.length

        if T < 2:
            continue

        # 预计算每帧的 proprio 向量（24 维）
        proprio_all = _build_proprio_from_episode(ep)  # (T, 24)

        # 构建 obs 字典列表（用于 VLA 推断）
        obs_list = [
            {
                "image": ep.images[t],
                "proprio": proprio_all[t],
                "language": lang_embed.astype(np.float32),
            }
            for t in range(T)
        ]

        # 预计算 base_action（可选：批量 VLA 推断 或 GT 动作）
        if use_base_policy:
            base_actions = np.stack(
                [base_policy.predict_action(obs_list[t]) for t in range(T)],
                axis=0,
            ).astype(np.float32)  # (T, 7)
        else:
            # GT-as-base：用专家动作作为 base，残差策略目标输出为零
            base_actions = ep.actions.astype(np.float32)  # (T, 7)

        # 逐帧构建 transitions
        for t in range(T - 1):
            t_next = t + 1
            done_flag = t_next == T - 1

            # ── 当前帧 obs ─────────────────────────────────
            img_t = torch.as_tensor(
                ep.images[t].transpose(2, 0, 1), dtype=torch.uint8
            )  # (C,H,W)
            state_t = state_standardizer.standardize(
                torch.tensor(proprio_all[t], dtype=torch.float32)
            )
            base_t = torch.tensor(base_actions[t], dtype=torch.float32)
            curr_obs = {
                camera_key: img_t,
                "observation.state": state_t,
                "observation.base_action": base_t,
            }
            _to_uint8_inplace(curr_obs, image_keys)

            # ── 下一帧 obs ─────────────────────────────────
            img_n = torch.as_tensor(
                ep.images[t_next].transpose(2, 0, 1), dtype=torch.uint8
            )
            state_n = state_standardizer.standardize(
                torch.tensor(proprio_all[t_next], dtype=torch.float32)
            )
            base_n = torch.tensor(base_actions[t_next], dtype=torch.float32)
            next_obs = {
                camera_key: img_n,
                "observation.state": state_n,
                "observation.base_action": base_n,
            }
            _to_uint8_inplace(next_obs, image_keys)

            # ── 动作（专家 GT，缩放后）─────────────────────
            gt_action_scaled = action_scaler.scale(
                torch.tensor(ep.actions[t], dtype=torch.float32)
            )

            transition = TensorDict(
                {
                    "obs": TensorDict(curr_obs, batch_size=[]),
                    "action": gt_action_scaled,
                    "next": TensorDict(
                        {
                            "obs": TensorDict(next_obs, batch_size=[]),
                            "done": torch.tensor(done_flag, dtype=torch.bool),
                            # 稀疏奖励：episode 末步 = 1.0，其余 = 0.0
                            "reward": torch.tensor(
                                float(done_flag), dtype=torch.float32
                            ),
                        },
                        batch_size=[],
                    ),
                    "_priority": torch.tensor(10.0, dtype=torch.float32),
                },
                batch_size=[],
            ).unsqueeze(0)

            rb.add(transition)
            transitions += 1

    print(f"离线 buffer 填充完成：共添加 {transitions} 条 transitions")
    return transitions


def _build_proprio_from_episode(ep: EpisodeData) -> np.ndarray:
    """
    从 EpisodeData 中构建 24 维本体感觉向量，与 env.py 中 _build_obs 逻辑一致。

    格式：[tcp_pose(6) | joint_pos_vel(12) | force_torque(6)]
    """
    T = ep.length

    def pad_or_clip(arr: np.ndarray, length: int) -> np.ndarray:
        arr = arr.flatten() if arr.ndim > 1 else arr
        if len(arr) >= length:
            return arr[:length].astype(np.float32)
        return np.pad(arr, (0, length - len(arr))).astype(np.float32)

    proprio_list = []
    for t in range(T):
        tcp = pad_or_clip(ep.tcp_poses[t], 6)
        joints = pad_or_clip(ep.joint_states[t], 12)
        ft = pad_or_clip(ep.force_torques[t], 6)
        proprio_list.append(np.concatenate([tcp, joints, ft]))

    return np.stack(proprio_list, axis=0)  # (T, 24)


# ===========================================================================
#  评估函数
# ===========================================================================


def run_evaluation(
    env_wrapper: RealRobotBasePolicyEnvWrapper,
    agent: "QAgent",
    num_episodes: int,
    device: torch.device,
    global_step: int,
    save_video: bool = False,
    run_name: str = "",
    output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    在真机上运行若干评估 episode，返回性能指标。

    注意：评估与训练共用同一物理机器人，评估时暂停数据收集。

    Returns
    -------
    metrics : dict
        eval/success_rate, eval/mean_reward, eval/mean_length,
        eval/mean_pos_error
    """
    rewards: List[float] = []
    successes: List[float] = []
    lengths: List[int] = []
    pos_errors: List[float] = []
    frames: List[List[np.ndarray]] = []

    for ep_idx in range(num_episodes):
        obs, _ = env_wrapper.reset()
        ep_reward = 0.0
        ep_success = False
        ep_frames: List[np.ndarray] = []
        step = 0

        while True:
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            obs, reward, terminated, truncated, info = env_wrapper.step(action)
            ep_reward += reward.sum().item()
            step += 1

            if save_video and output_dir is not None:
                # 从 camera_key 取图像帧
                img_key = env_wrapper.camera_key
                if img_key in obs:
                    ep_frames.append(obs[img_key][0].cpu().numpy())

            if info.get("is_success", False):
                ep_success = True

            done = (terminated | truncated).any().item()
            if done:
                # 提取 terminal 时的位置误差
                pos_errors.append(float(info.get("pos_error", 0.0)))
                break

        rewards.append(ep_reward)
        successes.append(float(ep_success))
        lengths.append(step)

        if save_video and ep_frames and output_dir is not None:
            _save_video(ep_frames, output_dir, global_step, ep_idx, run_name)

    metrics = {
        "eval/success_rate": float(np.mean(successes)),
        "eval/mean_reward": float(np.mean(rewards)),
        "eval/mean_length": float(np.mean(lengths)),
        "eval/mean_pos_error": float(np.mean(pos_errors)) if pos_errors else 0.0,
    }
    return metrics


def _save_video(
    frames: List[np.ndarray],
    output_dir: Path,
    global_step: int,
    ep_idx: int,
    run_name: str,
) -> None:
    """将评估视频帧保存为 MP4（需要 cv2）。"""
    try:
        import cv2

        fname = output_dir / f"eval_step{global_step}_ep{ep_idx}.mp4"
        if len(frames) == 0:
            return
        # frames: list of (C,H,W) uint8 numpy
        h, w = frames[0].shape[1], frames[0].shape[2]
        writer = cv2.VideoWriter(
            str(fname),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (w, h),
        )
        for f in frames:
            bgr = cv2.cvtColor(f.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        logger.info("视频已保存: %s", fname)
    except Exception as exc:
        logger.warning("视频保存失败: %s", exc)


# ===========================================================================
#  主训练函数
# ===========================================================================


def main(cfg: ResidualTD3RealRobotConfig) -> None:
    """
    真机 Residual TD3 主训练循环。

    流程
    ----
    1. 初始化设备 & 随机种子
    2. 加载 VLA 基础策略（OpenVLA 或自定义）
    3. 加载离线数据集，计算归一化统计量
    4. 构建真机环境 & wrapper（Online / Offline / Mixed）
    5. 构建 QAgent & replay buffers
    6. 填充离线 buffer（来自 VLA 数据集）
    7. Warm-up 阶段（随机探索填充 online buffer）
    8. 主训练循环：环境交互 + actor/critic 更新 + 定期评估
    """

    # ── 1. 设备 & 随机种子 ─────────────────────────────────────────────────
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic
    print(f"随机种子: {cfg.seed}")

    # ── 2. 加载 VLA 基础策略 ────────────────────────────────────────────────
    print(f"加载 VLA 基础策略: {cfg.vla_model_path}")
    base_policy = OpenVLAWrapper(
        model_path=cfg.vla_model_path,
        device=cfg.vla_device,
    )
    base_policy._model.eval()
    print("VLA 基础策略加载完成。")

    # ── 3. 加载离线数据集 & 归一化统计量 ────────────────────────────────────
    dataset_loader: Optional[DatasetLoader] = None
    action_scaler: SimpleActionScaler
    state_standardizer: SimpleStateStandardizer

    if cfg.offline_data.dataset_path is not None:
        print(f"加载离线数据集: {cfg.offline_data.dataset_path}")
        dataset_loader = load_dataset(
            cfg.offline_data.dataset_path,
            fmt=cfg.offline_data.dataset_fmt,
        )
        print(f"数据集加载完成，共 {len(dataset_loader)} 条 episode。")

        # 从数据集计算归一化参数
        action_scaler = SimpleActionScaler.from_dataset(
            dataset_loader,
            min_range=cfg.offline_data.min_action_range,
            device=device_str,
        )
        state_standardizer = SimpleStateStandardizer.from_dataset(
            dataset_loader,
            min_std=cfg.offline_data.min_state_std,
            device=device_str,
        )
    else:
        # 无离线数据集：使用恒等变换
        print("未提供离线数据集，使用恒等归一化。")
        action_scaler = SimpleActionScaler.identity(dim=7, device=device_str)
        state_standardizer = SimpleStateStandardizer.identity(dim=24, device=device_str)

    # 任务语言嵌入（用于离线 buffer 填充时的 VLA 推断）
    lang_embed = np.zeros(512, dtype=np.float32)  # 占位；VLAWrapper 内部处理语言

    # ── 4. 构建真机环境 ─────────────────────────────────────────────────────
    safety_dict = cfg.safety.to_dict()
    target_pose = np.array(cfg.target_pose, dtype=np.float32)

    # 在线训练环境（真机交互）
    train_env_raw = RealRobotEnv(
        task_language=cfg.task_language,
        target_pose=target_pose,
        mode=EnvMode.ONLINE,
        max_steps=cfg.max_steps,
        control_freq_hz=cfg.robot.control_freq_hz,
        safety_config=safety_dict,
    )
    env = RealRobotBasePolicyEnvWrapper(
        env=train_env_raw,
        base_policy=base_policy,
        camera_key=cfg.rl_camera,
        state_standardizer=state_standardizer,
        device=device_str,
    )
    env.seed(cfg.seed)

    # ── 5. 从 env 推断维度信息 ──────────────────────────────────────────────
    image_keys: List[str] = (
        [cfg.rl_camera] if isinstance(cfg.rl_camera, str) else list(cfg.rl_camera)
    )
    lowdim_keys = ["observation.state", "observation.base_action"]

    lowdim_dim = env.observation_space["observation.state"].shape[1]    # 24
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]  # 3,224,224
    action_dim = env.action_space.shape[1]                                 # 7
    horizon = env.vec_env.metadata["horizon"]

    print(
        f"环境维度: lowdim={lowdim_dim}, img={img_c}×{img_h}×{img_w}, "
        f"action={action_dim}, horizon={horizon}"
    )

    # ── 6. 构建 QAgent ──────────────────────────────────────────────────────
    cfg.actor_name = "residual_vla"
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
    )

    actor_updates = 0
    if cfg.algo.actor_lr_warmup_steps > 0:
        print(
            f"Actor LR warmup: 0 → {cfg.agent.actor_lr:.2e} "
            f"（{cfg.algo.actor_lr_warmup_steps} 步）"
        )

    # ── 7. 构建 Replay Buffers ──────────────────────────────────────────────
    alpha = (
        cfg.algo.priority_alpha
        if cfg.algo.sampling_strategy == "prioritized_replay"
        else 0.0
    )
    beta = (
        cfg.algo.priority_beta
        if cfg.algo.sampling_strategy == "prioritized_replay"
        else 0.0
    )

    online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
    offline_batch_size = int(cfg.algo.batch_size * cfg.algo.offline_fraction)

    if cfg.algo.offline_fraction == 0.0:
        print("纯在线训练模式 (offline_fraction=0.0)")
    elif cfg.algo.offline_fraction == 1.0:
        print("纯离线训练模式 (offline_fraction=1.0)")
    else:
        print(
            f"混合训练模式 (offline_fraction={cfg.algo.offline_fraction:.2f}，"
            f"online_batch={online_batch_size}, offline_batch={offline_batch_size})"
        )

    online_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.algo.buffer_size, device="cpu"),
        alpha=alpha,
        beta=beta,
        eps=1e-6,
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,
        batch_size=max(online_batch_size, 1),
    )

    # ── Online buffer 缓存（可选，加速重启）──────────────────────────────
    online_cache_meta = {
        "task_language": cfg.task_language,
        "image_keys": image_keys,
        "n_step": cfg.algo.n_step,
        "gamma": cfg.algo.gamma,
        "horizon": horizon,
        "size": cfg.algo.learning_starts,
        "sampling_strategy": cfg.algo.sampling_strategy,
        "buffer_size": cfg.algo.buffer_size,
        "batch_size": online_batch_size,
        "random_action_noise_scale": cfg.algo.random_action_noise_scale,
    }
    _online_meta_str = json.dumps(online_cache_meta, sort_keys=True)
    online_cache_hash = hashlib.sha256(_online_meta_str.encode()).hexdigest()[:8]
    online_cache_dir = ONLINE_CACHE_DIR / online_cache_hash

    dl_dir = None
    if ONLINE_HF_REPO is not None:
        dl_dir = _hf_download_buffer(ONLINE_HF_REPO, online_cache_hash, ONLINE_CACHE_DIR)
    if dl_dir is not None:
        online_cache_dir = dl_dir

    loaded_online_from_cache = False
    if online_cache_dir.exists():
        print(f"从缓存加载 online buffer: {online_cache_dir}")
        online_rb.sampler._empty()
        optimized_replay_buffer_loads(online_rb, online_cache_dir)
        loaded_online_from_cache = True
        print(f"Online buffer 缓存加载完成 (size={len(online_rb)})")

    # ── 8. 离线 Buffer ────────────────────────────────────────────────────
    # 估算 offline buffer 容量
    if dataset_loader is not None and cfg.algo.offline_fraction > 0.0:
        n_eps = (
            min(cfg.offline_data.num_episodes, len(dataset_loader))
            if cfg.offline_data.num_episodes is not None
            else len(dataset_loader)
        )
        max_offline_transitions = sum(
            max(dataset_loader.get_episode(i).length - 1, 0) for i in range(n_eps)
        )
    else:
        max_offline_transitions = 1  # 在线训练模式：创建最小 buffer（不使用）

    offline_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=max_offline_transitions, device="cpu"),
        alpha=alpha,
        beta=beta,
        eps=1e-6,
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,
        batch_size=max(offline_batch_size, 1),
    )

    # ── 9. 填充离线 Buffer ────────────────────────────────────────────────
    if cfg.algo.offline_fraction > 0.0 and dataset_loader is not None:
        # 构建缓存哈希
        offline_cache_meta = {
            "task_language": cfg.task_language,
            "dataset_path": cfg.offline_data.dataset_path,
            "num_episodes": cfg.offline_data.num_episodes,
            "use_base_policy": cfg.offline_data.use_base_policy_for_base_actions,
            "min_action_range": cfg.offline_data.min_action_range,
            "min_state_std": cfg.offline_data.min_state_std,
            "image_keys": image_keys,
            "n_step": cfg.algo.n_step,
            "gamma": cfg.algo.gamma,
            "sampling_strategy": cfg.algo.sampling_strategy,
            "batch_size": offline_batch_size,
        }
        meta_str = json.dumps(offline_cache_meta, sort_keys=True)
        cache_hash = hashlib.sha256(meta_str.encode()).hexdigest()[:8]
        cache_dir = OFFLINE_CACHE_DIR / cache_hash

        downloaded_dir = None
        if OFFLINE_HF_REPO is not None:
            downloaded_dir = _hf_download_buffer(OFFLINE_HF_REPO, cache_hash, OFFLINE_CACHE_DIR)
        if downloaded_dir is not None:
            cache_dir = downloaded_dir

        loaded_from_cache = False
        if cache_dir.exists():
            print(f"从缓存加载 offline buffer: {cache_dir}")
            offline_rb.sampler._empty()
            optimized_replay_buffer_loads(offline_rb, cache_dir)
            loaded_from_cache = True
            print(f"Offline buffer 缓存加载完成 (size={len(offline_rb)})")

        if not loaded_from_cache:
            added = _populate_offline_buffer(
                dataset_loader=dataset_loader,
                rb=offline_rb,
                image_keys=image_keys,
                camera_key=cfg.rl_camera,
                action_scaler=action_scaler,
                state_standardizer=state_standardizer,
                lang_embed=lang_embed,
                num_episodes=cfg.offline_data.num_episodes,
                use_base_policy=cfg.offline_data.use_base_policy_for_base_actions,
                base_policy=(
                    base_policy
                    if cfg.offline_data.use_base_policy_for_base_actions
                    else None
                ),
            )
            print(f"Offline buffer 已填充 {added} 条 transitions (size={len(offline_rb)})")

            # 持久化到本地
            cache_dir.mkdir(parents=True, exist_ok=True)
            optimized_replay_buffer_dumps(offline_rb, cache_dir)
            with open(cache_dir / "user_metadata.json", "w") as f:
                json.dump(offline_cache_meta, f, indent=2)
            if OFFLINE_HF_REPO is not None:
                _hf_upload_buffer(OFFLINE_HF_REPO, cache_dir, cache_hash)
    else:
        print("跳过 offline buffer 填充（纯在线训练模式 或 未提供数据集）")

    # ── 10. Warm-up 阶段（随机探索填充 online buffer）────────────────────
    if len(online_rb) < cfg.algo.learning_starts and not loaded_online_from_cache:
        needed = cfg.algo.learning_starts - len(online_rb)
        print(f"Warm-up：收集 {needed} 步随机探索数据…")
        obs, _ = env.reset()
        next_log = 1000

        while len(online_rb) < cfg.algo.learning_starts:
            if cfg.algo.use_base_policy_for_warmup:
                # 以小幅随机噪声作为残差（基础策略引导探索）
                rand_residual = (
                    torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1
                ) * cfg.algo.random_action_noise_scale
            else:
                # 纯随机：抵消 base_action，使 combined 均匀分布
                base_action = obs["observation.base_action"]
                rand_residual = (
                    torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1
                ) * cfg.algo.random_action_noise_scale - base_action

            next_obs, reward, terminated, truncated, info = env.step(rand_residual)
            done = terminated | truncated

            combined_action = info["scaled_action"]
            _add_transitions_to_buffer(
                obs=obs,
                next_obs=next_obs,
                actions=combined_action,
                reward=reward,
                done=done,
                info=info,
                device=device,
                image_keys=image_keys,
                lowdim_keys=lowdim_keys,
                num_envs=cfg.num_envs,
                online_rb=online_rb,
            )

            if len(online_rb) >= next_log:
                print(
                    f"[Warm-up] {len(online_rb)} / {cfg.algo.learning_starts} "
                    f"transitions 收集完成"
                )
                next_log += 1000

            obs = next_obs

        # 持久化 warm-up 数据
        online_cache_dir.mkdir(parents=True, exist_ok=True)
        optimized_replay_buffer_dumps(online_rb, online_cache_dir)
        with open(online_cache_dir / "user_metadata.json", "w") as f:
            json.dump(online_cache_meta, f, indent=2)
        if ONLINE_HF_REPO is not None:
            _hf_upload_buffer(ONLINE_HF_REPO, online_cache_dir, online_cache_hash)
        print(f"Warm-up 完成，online buffer size = {len(online_rb)}")
        loaded_online_from_cache = True

    # ── 11. Critic Warmup（可选）─────────────────────────────────────────
    if cfg.algo.critic_warmup_steps > 0:
        print(f"Critic warmup：运行 {cfg.algo.critic_warmup_steps} 步纯 critic 更新…")
        for i in range(cfg.algo.critic_warmup_steps):
            online_batch = online_rb.sample(max(online_batch_size, 1)).to(
                device, non_blocking=True
            )
            if cfg.algo.offline_fraction > 0.0 and len(offline_rb) > 0:
                offline_batch = offline_rb.sample(offline_batch_size).to(
                    device, non_blocking=True
                )
                batch = torch.cat([online_batch, offline_batch], dim=0)
            else:
                batch = online_batch
            metrics = agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)
            # ref_agent=agent: during critic warmup the target network is initialized from
            # the same agent weights; QAgent uses this to set up EMA / target policy.
            if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                batch["_priority"] = metrics["_td_errors"]
                online_rb.update_tensordict_priority(batch[:online_batch_size])
                if cfg.algo.offline_fraction > 0.0 and online_batch_size < len(batch):
                    offline_rb.update_tensordict_priority(batch[online_batch_size:])
            if i % 100 == 0:
                print(
                    f"[Critic warmup] {i}/{cfg.algo.critic_warmup_steps} "
                    f"critic_loss={metrics['train/critic_loss']:.4f}"
                )
        print("Critic warmup 完成。")

    # ── 12. 构建 WandB 运行名 ─────────────────────────────────────────────
    hp_parts = [
        cfg.task_language[:20].replace(" ", "_"),
        f"n{cfg.algo.n_step}",
        f"utd{cfg.algo.num_updates_per_iteration}",
        f"buf{cfg.algo.buffer_size}",
        f"off{cfg.algo.offline_fraction:.2f}",
        f"lr{cfg.agent.actor_lr:.0e}",
    ]
    if cfg.agent.clip_q_target_to_reward_range:
        hp_parts.append("clipT")
    hp_str = "_".join(hp_parts)
    run_name = (
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{hp_str}__seed{cfg.seed}"
    )
    if cfg.wandb.name is not None:
        run_name = f"{cfg.wandb.name}__{run_name}"

    import dataclasses
    _wandb_config = dataclasses.asdict(cfg)
    _wandb_config["wandb"].pop("notes", None)

    pprint.pprint(_wandb_config)
    wandb.init(
        id=cfg.wandb.continue_run_id,
        resume=None if cfg.wandb.continue_run_id is None else "allow",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=_wandb_config,
        name=run_name,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
        notes=cfg.wandb.notes,
        group=cfg.wandb.group,
    )
    wandb.summary["environment/horizon"] = horizon

    # ── 13. 创建输出目录 ───────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_cache_dir = _CACHE_ROOT / f"run_{timestamp}_{run_name}"
    model_save_dir = run_cache_dir / "models"
    outputs_dir = run_cache_dir / "outputs"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ── 14. 主训练循环 ─────────────────────────────────────────────────────
    obs, _ = env.reset()
    global_step = 0
    best_eval_success_rate = 0.0
    training_cum_time = 0.0
    episode_count = 0
    train_start_time = time.time()
    training_timer = TrainingTimer()

    actor_update_cadence = max(
        cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration, 1
    )

    while global_step <= cfg.algo.total_timesteps:
        iter_start = time.time()

        # ── (1) 环境交互 ──────────────────────────────────────────────────
        with training_timer.time("env_step"):
            with torch.no_grad(), utils.eval_mode(agent):
                stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
                action = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)

            if cfg.algo.progressive_clipping_steps > 0:
                clip_factor = min(
                    1.0, global_step / cfg.algo.progressive_clipping_steps
                )
                action = action * clip_factor
            else:
                clip_factor = 1.0

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

        # ── episode 结束统计 ────────────────────────────────────────────
        if done.any():
            episode_count += done.float().sum().item()
            final_info = info["final_info"]
            episode_steps = final_info["episode_steps"]
            episode_indices = final_info["_episode_steps"]

            if episode_indices.any():
                discount_factor = cfg.algo.gamma ** episode_steps[episode_indices]
                episode_rewards = reward.cpu().numpy()[episode_indices]
                episode_return = float(np.mean(discount_factor * episode_rewards))
            else:
                episode_return = 0.0

            wandb.log(
                {
                    "training/episode_return": episode_return,
                    "training/episode_steps": int(np.max(episode_steps)),
                    "training/episode_count": episode_count,
                },
                step=global_step,
            )

        # ── 添加到 online replay buffer ──────────────────────────────────
        combined_action = info["scaled_action"]
        _add_transitions_to_buffer(
            obs=obs,
            next_obs=next_obs,
            actions=combined_action,
            reward=reward,
            done=done,
            info=info,
            device=device,
            image_keys=image_keys,
            lowdim_keys=lowdim_keys,
            num_envs=cfg.num_envs,
            online_rb=online_rb,
        )
        obs = next_obs

        # ── (2) 定期评估 ───────────────────────────────────────────────────
        if global_step % cfg.eval_interval_every_steps == 0 and (
            cfg.eval_first or global_step > 0
        ):
            with training_timer.time("evaluation"):
                eval_metrics = run_evaluation(
                    env_wrapper=env,
                    agent=agent,
                    num_episodes=cfg.eval_num_episodes,
                    device=device,
                    global_step=global_step,
                    save_video=cfg.save_video,
                    run_name=run_name,
                    output_dir=outputs_dir,
                )
                wandb.log(eval_metrics, step=global_step)
                print(
                    f"[Eval @ {global_step}] "
                    f"success={eval_metrics['eval/success_rate']:.3f} "
                    f"reward={eval_metrics['eval/mean_reward']:.2f} "
                    f"length={eval_metrics['eval/mean_length']:.1f}"
                )

                current_success_rate = eval_metrics["eval/success_rate"]
                if current_success_rate > best_eval_success_rate:
                    best_eval_success_rate = current_success_rate
                    torch.save(
                        agent.state_dict(),
                        model_save_dir / f"best_model_step{global_step}.pt",
                    )
                    print(f"🎉 新最佳成功率: {current_success_rate:.4f}")

            # 评估后重新开始 obs（env 在 step 内部已自动 reset）
            obs, _ = env.reset()

        global_step += cfg.num_envs

        # ── (3) 网络更新 ───────────────────────────────────────────────────
        if global_step % cfg.algo.update_every_n_steps == 0 or global_step == cfg.num_envs:
            i = 0
            while i < cfg.algo.num_updates_per_iteration:
                with training_timer.time("batch_sampling"):
                    online_batch = online_rb.sample(max(online_batch_size, 1)).to(
                        device, non_blocking=True
                    )
                    if cfg.algo.offline_fraction > 0.0 and len(offline_rb) > 0:
                        offline_batch = offline_rb.sample(offline_batch_size).to(
                            device, non_blocking=True
                        )
                        batch = torch.cat([online_batch, offline_batch], dim=0)
                    else:
                        batch = online_batch

                update_actor = (i + 1) % actor_update_cadence == 0

                if update_actor and cfg.algo.actor_lr_warmup_steps > 0:
                    warmup_progress = min(
                        1.0, actor_updates / cfg.algo.actor_lr_warmup_steps
                    )
                    current_lr = cfg.agent.actor_lr * warmup_progress
                    for pg in agent.actor_opt.param_groups:
                        pg["lr"] = current_lr

                if update_actor:
                    actor_updates += 1

                with training_timer.time("gradient_update"):
                    # ref_agent=agent: QAgent uses this for EMA target-network updates;
                    # passing the same agent is consistent with the original training loop.
                    metrics = agent.update(
                        batch, stddev, update_actor, bc_batch=None, ref_agent=agent
                    )

                # 更新优先级（PER）
                if (
                    cfg.algo.sampling_strategy == "prioritized_replay"
                    and "_td_errors" in metrics
                ):
                    batch["_priority"] = metrics["_td_errors"]
                    if cfg.algo.offline_fraction > 0.0:
                        online_batch_size_actual = int(
                            cfg.algo.batch_size * (1 - cfg.algo.offline_fraction)
                        )
                        if online_batch_size_actual > 0:
                            online_rb.update_tensordict_priority(
                                batch[:online_batch_size_actual]
                            )
                        if online_batch_size_actual < len(batch):
                            offline_rb.update_tensordict_priority(
                                batch[online_batch_size_actual:]
                            )
                    else:
                        online_rb.update_tensordict_priority(batch)

                metrics["data/batch_terminal_R"] = (
                    batch["next"]["reward"][~batch["nonterminal"]].mean()
                )
                metrics["data/terminal_share"] = (
                    (~batch["nonterminal"]).float().mean()
                )
                i += 1

        training_cum_time += time.time() - iter_start

        # ── (4) 日志 ───────────────────────────────────────────────────────
        if global_step % cfg.log_freq == 0:
            sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0
            log_dict = {
                "training/SPS": sps,
                "training/global_step": global_step,
                "buffer/online_size": len(online_rb),
                "buffer/offline_size": len(offline_rb),
                "timing/training_total_time": time.time() - train_start_time,
                "training/actor_lr": agent.actor_opt.param_groups[0]["lr"],
            }
            log_dict.update(training_timer.get_timing_stats())
            filtered = {k: v for k, v in metrics.items() if not k.startswith("_")}
            log_dict.update(filtered)

            if "_actions" in metrics:
                acts = metrics["_actions"]
                log_dict["train/residual_l1_magnitude"] = torch.mean(
                    torch.abs(acts)
                ).item()
                log_dict["train/residual_l2_magnitude"] = torch.mean(
                    torch.square(acts)
                ).item()
                log_dict["histograms/residual_actions"] = wandb.Histogram(
                    acts.cpu().numpy().reshape(-1)
                )
            if "_target_q" in metrics:
                log_dict["histograms/critic_qt"] = wandb.Histogram(
                    metrics["_target_q"].cpu().numpy().reshape(-1)
                )
            if cfg.algo.progressive_clipping_steps > 0:
                log_dict["training/progressive_clipping_factor"] = clip_factor

            wandb.log(log_dict, step=global_step)

            # 控制台输出
            current_actor_lr = agent.actor_opt.param_groups[0]["lr"]
            if "train/actor_loss_base" in metrics:
                msg = (
                    f"[{global_step}] actor_loss={metrics['train/actor_loss_base']:.4f} "
                    f"critic_loss={metrics['train/critic_loss']:.4f} "
                    f"actor_lr={current_actor_lr:.2e}"
                )
            else:
                msg = (
                    f"[{global_step}] critic_loss={metrics['train/critic_loss']:.4f} "
                    f"actor_lr={current_actor_lr:.2e}"
                )
            timing = training_timer.get_timing_stats()
            if timing:
                msg += (
                    f" | Time%: "
                    f"env={timing.get('timing/env_step_percentage', 0):.1f} "
                    f"grad={timing.get('timing/gradient_update_percentage', 0):.1f}"
                )
            print(msg)

    # ── 15. 训练结束 ───────────────────────────────────────────────────────
    total_time = time.time() - train_start_time
    print(f"训练完成，总用时 {total_time:.2f} 秒。")
    env.close()

    if run_cache_dir.exists():
        print(f"清理运行目录: {run_cache_dir}")
        shutil.rmtree(run_cache_dir)


# ===========================================================================
#  入口点
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="真机 Residual TD3 训练")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON 配置文件路径（可选，覆盖默认参数）",
    )
    parser.add_argument(
        "--task_language",
        type=str,
        default=None,
        help="任务语言描述",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="离线数据集路径",
    )
    parser.add_argument(
        "--vla_model_path",
        type=str,
        default=None,
        help="VLA 基础策略路径（本地目录或 HuggingFace Hub ID）",
    )
    parser.add_argument(
        "--offline_fraction",
        type=float,
        default=None,
        help="离线数据比例 (0.0~1.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式（禁用 wandb，减少步数）",
    )
    args = parser.parse_args()

    cfg = ResidualTD3RealRobotConfig()

    # 从 JSON 文件加载配置覆盖
    if args.config is not None:
        import dataclasses

        with open(args.config) as f:
            overrides = _json.load(f)
        for key, val in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)

    # 命令行参数覆盖
    if args.task_language is not None:
        cfg.task_language = args.task_language
    if args.dataset_path is not None:
        cfg.offline_data.dataset_path = args.dataset_path
    if args.vla_model_path is not None:
        cfg.vla_model_path = args.vla_model_path
    if args.offline_fraction is not None:
        cfg.algo.offline_fraction = args.offline_fraction
    if args.debug:
        cfg.debug = True
        cfg.algo.total_timesteps = 200
        cfg.algo.learning_starts = 10
        cfg.eval_num_episodes = 1
        cfg.eval_interval_every_steps = 100

    main(cfg)
