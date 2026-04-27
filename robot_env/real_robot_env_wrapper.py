"""
real_robot_env_wrapper.py  ——  RealRobotEnv 到 Residual TD3 训练接口的适配层

核心功能
--------
* 将单个 RealRobotEnv（Online 或 Offline 模式）包装为与
  train_real_robot_td3.py 兼容的类向量化环境接口（batch_size = 1）
* 每次 step/reset 后自动调用 VLA 基础策略计算 base_action，并注入到
  观测字典中作为 "observation.base_action"
* 观测键重命名：image → <camera_key>，proprio → "observation.state"
* 图像格式转换：HWC uint8 → CHW uint8 Tensor
* 为所有张量添加 batch 维度 (dim-0 = 1)
* 计算 combined_action = clip(base_action + residual_action, -1, 1)
  通过 info["scaled_action"] 返回，供 replay buffer 存储
* 支持 StateStandardizer 对本体感觉状态进行标准化（可选）

接口规范（与 BasePolicyVecEnvWrapper 兼容）
------------------------------------------
reset() → (obs_dict, info)
    obs_dict 键：
        <camera_key>          : (1, C, H, W)  uint8  Tensor
        "observation.state"   : (1, 24)       float32 Tensor
        "observation.base_action": (1, 7)     float32 Tensor

step(residual_action: (1, 7)) → (obs_dict, reward, terminated, truncated, info)
    info["scaled_action"]   : (1, 7) float32 Tensor — 实际执行的 combined action
    info["final_obs"]       : list[dict | None]      — episode 结束时的最终状态
    info["final_info"]["episode_steps"]  : np.ndarray — 当前 episode 步数
    info["final_info"]["_episode_steps"] : np.ndarray — done 掩码（bool）

observation_space[key].shape : (1, *feature_shape)
action_space.shape           : (1, 7)
vec_env.metadata["horizon"]  : int  — 最大步数
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces

from env import RealRobotEnv, IMG_C, IMG_H, IMG_W
from vla_interface import VLAModelWrapper

logger = logging.getLogger(__name__)


class RealRobotBasePolicyEnvWrapper:
    """
    将单个 RealRobotEnv 包装为 Residual TD3 训练脚本兼容的接口。

    Parameters
    ----------
    env              : RealRobotEnv 实例（Online 或 Offline 模式）
    base_policy      : VLAModelWrapper 实例，负责生成 base_action
    camera_key       : 观测字典中相机图像使用的键名
    state_standardizer : 可选的状态标准化器，需实现 standardize(tensor) → tensor
    device           : 张量设备（"cpu" 或 "cuda"）
    """

    def __init__(
        self,
        env: RealRobotEnv,
        base_policy: VLAModelWrapper,
        camera_key: str = "observation.images.top",
        state_standardizer=None,
        device: str = "cpu",
    ):
        self.env = env
        self.base_policy = base_policy
        self.camera_key = camera_key
        self.state_standardizer = state_standardizer
        self.device = device

        # 当前 episode 的内部状态
        self._current_base_action: np.ndarray = np.zeros(7, dtype=np.float32)
        self._episode_step: int = 0

        # ── 观测/动作空间（含 batch 维度 = 1）─────────────────
        self.observation_space = spaces.Dict(
            {
                camera_key: spaces.Box(
                    0,
                    255,
                    shape=(1, IMG_C, IMG_H, IMG_W),
                    dtype=np.uint8,
                ),
                "observation.state": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(1, 24),
                    dtype=np.float32,
                ),
                "observation.base_action": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(1, 7),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1, 7), dtype=np.float32
        )

        # metadata：训练脚本通过 env.vec_env.metadata["horizon"] 访问
        self.metadata = dict(env.metadata)
        self.metadata["horizon"] = env.max_steps

        # 让 env.vec_env.metadata 可访问（训练脚本原生接口）
        self.vec_env = self

    # ===========================================================
    #  主要 Gymnasium 接口
    # ===========================================================

    def reset(self, **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        重置环境，返回初始观测。

        Returns
        -------
        obs  : dict of Tensors，各 shape 含 batch 维度 (1, ...)
        info : 空字典（与 Gymnasium 规范一致）
        """
        obs, info = self.env.reset(**kwargs)
        self._episode_step = 0
        self._current_base_action = self._compute_base_action(obs)
        return self._wrap_obs(obs, self._current_base_action), info

    def step(
        self,
        residual_action: torch.Tensor,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict,
    ]:
        """
        执行一步残差动作。

        Parameters
        ----------
        residual_action : (1, 7) float32 Tensor
            残差策略输出，已在 [-1, 1] 内

        Returns
        -------
        obs         : dict，各 shape (1, ...)
        reward      : (1,) float32 Tensor
        terminated  : (1,) bool Tensor
        truncated   : (1,) bool Tensor
        info        : dict（含 scaled_action / final_obs / final_info）
        """
        # ── 1. 计算 combined_action ───────────────────────────
        residual_np = (
            residual_action[0].cpu().numpy()
            if isinstance(residual_action, torch.Tensor)
            else np.asarray(residual_action[0])
        ).astype(np.float32)

        combined = np.clip(
            self._current_base_action + residual_np, -1.0, 1.0
        ).astype(np.float32)

        # ── 2. 执行真机动作 ───────────────────────────────────
        obs, reward, terminated, truncated, step_info = self.env.step(combined)
        done = terminated or truncated
        self._episode_step += 1

        # ── 3. 构建包装后的 obs ───────────────────────────────
        if done:
            # episode 结束时 base_action 置零（避免对无效帧进行 VLA 推断）
            new_base_action = np.zeros(7, dtype=np.float32)
        else:
            new_base_action = self._compute_base_action(obs)

        wrapped_obs = self._wrap_obs(obs, new_base_action)

        # ── 4. 处理 episode 结束信息 ──────────────────────────
        info: Dict[str, Any] = dict(step_info)

        if done:
            # final_obs：terminal 时刻的真实状态，用于 replay buffer 的 next_obs
            # shape 为无 batch 维度的 numpy 数组字典
            info["final_obs"] = [
                {
                    k: v[0].cpu().numpy()
                    if isinstance(v, torch.Tensor)
                    else (v[0] if isinstance(v, np.ndarray) else np.asarray(v[0]))
                    for k, v in wrapped_obs.items()
                }
            ]
            info["final_info"] = {
                "episode_steps": np.array([self._episode_step]),
                "_episode_steps": np.array([True]),
            }

            # 重置环境，更新 wrapped_obs 为新 episode 初始观测
            reset_obs, _ = self.env.reset()
            self._episode_step = 0
            new_base_action = self._compute_base_action(reset_obs)
            wrapped_obs = self._wrap_obs(reset_obs, new_base_action)
        else:
            info["final_obs"] = [None]
            info["final_info"] = {
                "episode_steps": np.array([0]),
                "_episode_steps": np.array([False]),
            }

        # ── 5. 记录组合动作供 replay buffer 存储 ─────────────
        info["scaled_action"] = torch.tensor(
            combined, dtype=torch.float32
        ).unsqueeze(0)  # (1, 7)

        # ── 6. 更新内部状态 ───────────────────────────────────
        self._current_base_action = new_base_action

        # ── 7. 转换标量为张量 ─────────────────────────────────
        reward_t = torch.tensor([reward], dtype=torch.float32)
        terminated_t = torch.tensor([terminated], dtype=torch.bool)
        truncated_t = torch.tensor([truncated], dtype=torch.bool)

        return wrapped_obs, reward_t, terminated_t, truncated_t, info

    def seed(self, seed: int) -> None:
        """设置随机种子（传递给底层 env）。"""
        self.env.np_random = np.random.default_rng(seed)

    def close(self) -> None:
        """关闭底层环境（真机模式下停止机器人）。"""
        self.env.close()

    @property
    def num_envs(self) -> int:
        return 1

    # ===========================================================
    #  内部工具方法
    # ===========================================================

    def _compute_base_action(self, obs: Dict) -> np.ndarray:
        """
        调用 VLA 基础策略推断 base_action。

        若推断失败（如模型异常），返回零向量并记录警告，
        保证训练循环不中断。
        """
        try:
            action = self.base_policy.predict_action(obs)
            return np.clip(action, -1.0, 1.0).astype(np.float32)
        except Exception as exc:
            logger.warning(
                "base_policy.predict_action 失败，base_action 置零: %s", exc
            )
            return np.zeros(7, dtype=np.float32)

    def _wrap_obs(
        self, obs: Dict, base_action: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """
        将 RealRobotEnv 原始观测转换为训练格式。

        转换规则
        --------
        image (H,W,C) uint8        → <camera_key>: (1,C,H,W) uint8 Tensor
        proprio (24,) float32      → "observation.state": (1,24) float32 Tensor
                                     （可选：经 state_standardizer 标准化）
        base_action (7,) float32   → "observation.base_action": (1,7) float32 Tensor
        """
        # 图像：HWC → CHW，添加 batch 维
        img_chw = obs["image"].transpose(2, 0, 1)  # (C, H, W)
        img_t = torch.as_tensor(img_chw, dtype=torch.uint8).unsqueeze(0)  # (1,C,H,W)

        # 本体感觉状态（可选标准化）
        proprio_t = torch.as_tensor(obs["proprio"], dtype=torch.float32)
        if self.state_standardizer is not None:
            proprio_t = self.state_standardizer.standardize(proprio_t)
        proprio_t = proprio_t.unsqueeze(0)  # (1, 24)

        # 基础动作
        base_t = torch.as_tensor(base_action, dtype=torch.float32).unsqueeze(0)  # (1, 7)

        return {
            self.camera_key: img_t,
            "observation.state": proprio_t,
            "observation.base_action": base_t,
        }
