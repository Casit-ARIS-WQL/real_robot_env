import gymnasium as gym
import numpy as np
from gymnasium import spaces
from enum import Enum
from typing import Callable, Iterator, Optional, Dict, Any, Tuple
import logging

from dataset_utils import DatasetLoader, EpisodeData, load_dataset

logger = logging.getLogger(__name__)


class EnvMode(Enum):
    ONLINE  = "online"   # 真机在线交互
    OFFLINE = "offline"  # VLA数据集离线回放


# ============================================================
#  多模态观测空间定义 (图像 + 本体感觉 + 语言Token)
# ============================================================
IMG_H, IMG_W, IMG_C = 224, 224, 3   # 与VLA预训练保持一致
LANG_TOKEN_DIM      = 512            # 语言编码维度 (e.g. CLIP text embedding)
PROPRIO_DIM         = 24             # 关节位置+速度+末端位姿+力矩


class RealRobotEnv(gym.Env):
    """
    统一的真机 Online / VLA Offline 强化学习环境。

    Observation (Dict):
        image    : (H, W, 3)  uint8  —— 相机 RGB
        proprio  : (24,)      float32 —— 本体感觉 [tcp(6) | joint_pos+vel(12) | ft(6)]
        language : (512,)     float32 —— 任务语言嵌入

    Action:
        (7,) float32  —— [dx,dy,dz, drx,dry,drz, gripper]  末端增量控制

    新增特性 (相较原版)
    ------------------
    * 支持多格式 VLA 数据集 (HDF5 / LeRobot / RLDS) 通过 dataset_utils.load_dataset
    * 修复 Offline 模式时间步错位 Bug：step(t) 现在正确返回执行 action[t] 后的 next_state
    * info 字典在 Offline 模式下包含 expert_action，便于 IQL/CQL/TD3-BC 等离线算法使用
    * episode_batch_iter() 方法：直接从数据集生成 (s,a,r,s',done) mini-batch，无需额外 wrapper
    * 支持多相机：extra_cameras 参数，obs 中额外返回 image_<name> 键
    * 支持自定义奖励函数注入 reward_fn，离线 RL 奖励重标注
    * 支持动作归一化器 action_normalizer，统一 VLA 输出与真机控制量的尺度
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task_language: str,
        target_pose: np.ndarray,
        mode: EnvMode = EnvMode.ONLINE,
        dataset_path: Optional[str] = None,    # Offline 模式下 VLA 数据集路径
        dataset_fmt: str = "auto",             # 数据集格式: "auto"|"hdf5"|"lerobot"|"rlds"
        language_encoder=None,                 # 语言编码器 (e.g. CLIPTextEncoder)
        robot_interface=None,                  # Online 模式下真机接口
        camera_interface=None,                 # Online 模式下主相机接口
        extra_cameras: Optional[Dict[str, Any]] = None,  # 额外相机 {name: interface}
        max_steps: int = 500,
        control_freq_hz: float = 10.0,
        safety_config: Optional[Dict] = None,
        action_normalizer=None,                # ActionNormalizer，用于 VLA 动作反归一化
        reward_fn: Optional[Callable] = None,  # 自定义奖励函数 fn(pose, action, raw_obs) -> float
    ):
        super().__init__()
        self.mode             = mode
        self.task_language    = task_language
        self.target_pose      = np.array(target_pose, dtype=np.float32)
        self.max_steps        = max_steps
        self.control_freq_hz  = control_freq_hz
        self.current_step     = 0
        self.prev_action      = np.zeros(7, dtype=np.float32)
        self.action_normalizer = action_normalizer
        self._reward_fn       = reward_fn
        self._extra_cameras   = extra_cameras or {}

        # ---------- 安全配置 ----------
        self.safety_cfg = safety_config or {
            "max_force_norm":    50.0,   # N
            "min_z":             0.05,   # m  防撞桌
            "max_joint_vel":     1.5,    # rad/s
            "max_action_delta":  0.1,    # 单步最大增量 (rad/m)
        }

        # ---------- 空间定义 ----------
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        obs_space: Dict[str, spaces.Space] = {
            "image":    spaces.Box(0, 255, (IMG_H, IMG_W, IMG_C), dtype=np.uint8),
            "proprio":  spaces.Box(-np.inf, np.inf, (PROPRIO_DIM,), dtype=np.float32),
            "language": spaces.Box(-np.inf, np.inf, (LANG_TOKEN_DIM,), dtype=np.float32),
        }
        for cam_name in self._extra_cameras:
            obs_space[f"image_{cam_name}"] = spaces.Box(
                0, 255, (IMG_H, IMG_W, IMG_C), dtype=np.uint8
            )
        self.observation_space = spaces.Dict(obs_space)

        # ---------- 语言编码 ----------
        self._lang_encoder = language_encoder
        self._lang_embed   = self._encode_language(task_language)

        # ---------- Online 接口 ----------
        self.robot_interface  = robot_interface
        self.camera_interface = camera_interface

        # ---------- Offline 数据集 ----------
        self._dataset_path  = dataset_path
        self._dataset_fmt   = dataset_fmt
        self._dataset_loader: Optional[DatasetLoader] = None
        self._episode_data: Optional[EpisodeData] = None
        self._episode_idx   = 0

        if mode == EnvMode.OFFLINE:
            self._load_dataset(dataset_path, dataset_fmt)

    # ===========================================================
    #  公共接口
    # ===========================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_action  = np.zeros(7, dtype=np.float32)

        if self.mode == EnvMode.ONLINE:
            obs = self._reset_online()
        else:
            obs = self._reset_offline()

        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        # ── 阶段1: 动作安全处理 ──────────────────────────────
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 增量幅度限制，防止真机单步暴走
        delta = action - self.prev_action
        delta = np.clip(delta, -self.safety_cfg["max_action_delta"],
                                self.safety_cfg["max_action_delta"])
        safe_action = self.prev_action + delta

        # ── 阶段2: 执行 & 获取下一状态观测 ───────────────────
        if self.mode == EnvMode.ONLINE:
            raw_obs = self._step_online(safe_action)
        else:
            # 修复：执行 action[t] 后应返回 t+1 时刻的状态
            raw_obs = self._step_offline(self.current_step + 1)

        self.current_step += 1

        current_pose  = raw_obs["tcp_pose"]
        joint_states  = raw_obs["joint_states"]
        force_torque  = raw_obs["force_torque"]

        # ── 阶段3: 终止 & 奖励判定 ───────────────────────────
        terminated, truncated = False, False
        is_success = False
        info: Dict[str, Any] = {}

        # 3.1 安全违规 (最高优先级)
        if self._is_safety_violation(joint_states, force_torque, current_pose):
            terminated = True
            reward     = -100.0
            info["termination_reason"] = "safety_violation"
            if self.mode == EnvMode.ONLINE:
                self._emergency_stop()
            logger.warning("Safety violation triggered!")

        # 3.2 任务成功
        elif self._is_task_successful(current_pose):
            terminated = True
            is_success = True
            reward     = 100.0
            info["termination_reason"] = "task_success"

        # 3.3 常规稠密奖励 + 超时截断
        else:
            if self._reward_fn is not None:
                reward = float(self._reward_fn(current_pose, safe_action, raw_obs))
            else:
                reward = self._compute_dense_reward(current_pose, safe_action, raw_obs)
            if self.current_step >= self.max_steps:
                truncated = True
                info["termination_reason"] = "time_limit"

        # ── 阶段4: 状态更新 & info 填充 ──────────────────────
        self.prev_action   = safe_action
        info["is_success"] = is_success
        info["step"]       = self.current_step
        info["pos_error"]  = float(np.linalg.norm(
                                 current_pose[:3] - self.target_pose[:3]))

        # Offline 模式：在 info 中暴露专家动作，供离线 RL 算法（IQL/CQL/TD3-BC）使用
        if self.mode == EnvMode.OFFLINE and self._episode_data is not None:
            t_expert = min(self.current_step - 1, self._episode_data.length - 1)
            info["expert_action"] = self._episode_data.actions[t_expert].copy()

        obs = self._build_obs(raw_obs)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.mode == EnvMode.ONLINE and self.camera_interface:
            return self.camera_interface.get_image()
        if self._episode_data is not None:
            idx = min(self.current_step, self._episode_data.length - 1)
            return self._episode_data.images[idx]
        return np.zeros((IMG_H, IMG_W, IMG_C), dtype=np.uint8)

    def close(self):
        if self.mode == EnvMode.ONLINE and self.robot_interface:
            self.robot_interface.stop()

    # ===========================================================
    #  离线 RL 批量迭代接口
    # ===========================================================

    def episode_batch_iter(
        self,
        batch_size: int = 256,
        shuffle: bool = True,
    ) -> Iterator[Dict[str, np.ndarray]]:
        """
        从 Offline 数据集生成 (s, a, r, s', done) mini-batch，
        直接用于离线 RL 算法（CQL / IQL / TD3-BC）训练。

        每个 batch 包含以下键：
            obs_image        : (B, H, W, 3)  uint8
            obs_proprio      : (B, 24)       float32
            actions          : (B, 7)        float32
            next_obs_image   : (B, H, W, 3)  uint8
            next_obs_proprio : (B, 24)       float32
            dones            : (B,)          float32   episode末步=1

        Parameters
        ----------
        batch_size : mini-batch 大小
        shuffle    : 是否打乱顺序
        """
        assert self.mode == EnvMode.OFFLINE, "episode_batch_iter 仅在 OFFLINE 模式下可用"
        assert self._dataset_loader is not None, "数据集未加载"
        yield from self._dataset_loader.transition_iter(
            batch_size=batch_size,
            shuffle=shuffle,
            rng=self.np_random if hasattr(self, "np_random") and self.np_random is not None
                else np.random.default_rng(),
        )

    # ===========================================================
    #  Online 模式实现
    # ===========================================================

    def _reset_online(self) -> Dict:
        """回到 Home 位，等待稳定后获取初始观测。"""
        assert self.robot_interface is not None, "需要提供 robot_interface"
        self.robot_interface.move_to_home()
        raw_obs = self._get_online_raw_obs()
        return self._build_obs(raw_obs)

    def _step_online(self, action: np.ndarray) -> Dict:
        """下发末端增量指令并同步获取状态。"""
        self.robot_interface.send_eef_delta_command(
            action[:6], gripper=action[6],
            freq_hz=self.control_freq_hz
        )
        return self._get_online_raw_obs()

    def _get_online_raw_obs(self) -> Dict:
        state = self.robot_interface.get_latest_observation()
        image = self.camera_interface.get_image()     # (H,W,3) uint8
        raw: Dict[str, Any] = {
            "tcp_pose":     np.array(state["tcp_pose"],     dtype=np.float32),
            "joint_states": np.array(state["joint_states"], dtype=np.float32),
            "force_torque": np.array(state["force_torque"], dtype=np.float32),
            "image":        image,
        }
        # 多相机
        for cam_name, cam_iface in self._extra_cameras.items():
            raw[f"image_{cam_name}"] = cam_iface.get_image()
        return raw

    # ===========================================================
    #  Offline 模式实现 (VLA数据集回放)
    # ===========================================================

    def _load_dataset(self, dataset_path: str, fmt: str = "auto"):
        """
        通过 dataset_utils.load_dataset 加载 VLA 数据集，
        支持 HDF5 / LeRobot / RLDS 三种格式自动检测。
        """
        if dataset_path is None:
            raise ValueError("Offline 模式需要提供 dataset_path")
        self._dataset_loader = load_dataset(dataset_path, fmt=fmt)
        logger.info("VLA 数据集加载成功: %d 条轨迹", len(self._dataset_loader))

    def _reset_offline(self) -> Dict:
        """从数据集中随机采样一条 episode，重置至起始帧。"""
        assert self._dataset_loader is not None, "请先调用 _load_dataset"
        self._episode_idx  = int(self.np_random.integers(0, len(self._dataset_loader)))
        self._episode_data = self._dataset_loader.get_episode(self._episode_idx)

        # 若数据集含语言标注则更新嵌入
        if self._episode_data.language:
            self._lang_embed = self._encode_language(self._episode_data.language)

        raw_obs = self._get_offline_raw_obs(0)
        return self._build_obs(raw_obs)

    def _step_offline(self, t: int) -> Dict:
        """按时间步索引回放数据集帧（自动截断至末帧）。"""
        t = min(t, self._episode_data.length - 1)
        return self._get_offline_raw_obs(t)

    def _get_offline_raw_obs(self, t: int) -> Dict:
        ep = self._episode_data
        raw: Dict[str, Any] = {
            "tcp_pose":     ep.tcp_poses[t].astype(np.float32),
            "joint_states": ep.joint_states[t].astype(np.float32),
            "force_torque": ep.force_torques[t].astype(np.float32),
            "image":        ep.images[t],
        }
        # 额外相机（存储在 EpisodeData.extra 中）
        for cam_name in self._extra_cameras:
            key = f"image_{cam_name}"
            if key in ep.extra:
                raw[key] = ep.extra[key][t]
        return raw

    # ===========================================================
    #  观测构建
    # ===========================================================

    def _build_obs(self, raw_obs: Dict) -> Dict:
        """将原始传感器数据组装为标准多模态观测字典。"""
        tcp    = raw_obs["tcp_pose"]       # (6,)
        joints = raw_obs["joint_states"]   # (12,)
        ft     = raw_obs["force_torque"]   # (6,)

        # 对齐至固定维度（数据集字段维度可能不同）
        tcp    = _pad_or_clip(tcp,    6)
        joints = _pad_or_clip(joints, 12)
        ft     = _pad_or_clip(ft,     6)
        proprio = np.concatenate([tcp, joints, ft]).astype(np.float32)  # (24,)

        # 图像尺寸对齐
        img = raw_obs["image"]
        if img.shape[:2] != (IMG_H, IMG_W):
            img = self._resize_image(img, IMG_H, IMG_W)

        obs: Dict[str, Any] = {
            "image":    img.astype(np.uint8),
            "proprio":  proprio,
            "language": self._lang_embed.astype(np.float32),
        }

        # 多相机图像
        for cam_name in self._extra_cameras:
            key = f"image_{cam_name}"
            if key in raw_obs:
                extra_img = raw_obs[key]
                if extra_img.shape[:2] != (IMG_H, IMG_W):
                    extra_img = self._resize_image(extra_img, IMG_H, IMG_W)
                obs[key] = extra_img.astype(np.uint8)

        return obs

    # ===========================================================
    #  奖励与终止
    # ===========================================================

    def _is_safety_violation(self, joint_states, force_torque, pose) -> bool:
        if np.linalg.norm(force_torque[:3]) > self.safety_cfg["max_force_norm"]:
            return True
        if pose[2] < self.safety_cfg["min_z"]:
            return True
        # 关节速度越限 (joint_states后半段为速度)
        n = len(joint_states) // 2
        if np.any(np.abs(joint_states[n:]) > self.safety_cfg["max_joint_vel"]):
            return True
        return False

    def _is_task_successful(self, pose) -> bool:
        pos_error = np.linalg.norm(pose[:3] - self.target_pose[:3])
        ori_error = np.linalg.norm(pose[3:] - self.target_pose[3:])
        return pos_error < 0.005 and ori_error < 0.02

    def _compute_dense_reward(
        self, pose: np.ndarray, action: np.ndarray, raw_obs: Dict
    ) -> float:
        pos_error = np.linalg.norm(pose[:3] - self.target_pose[:3])

        # 距离奖励 (指数衰减势能场)
        r_dist = np.exp(-5.0 * pos_error)

        # 姿态奖励
        ori_error = np.linalg.norm(pose[3:] - self.target_pose[3:])
        r_ori = np.exp(-2.0 * ori_error)

        # 动作平滑惩罚 (防止电机高频震荡)
        action_diff = np.linalg.norm(action - self.prev_action)
        p_smooth = 0.05 * (action_diff ** 2)

        # 力矩安全裕量奖励 (鼓励轻柔接触)
        ft_norm = np.linalg.norm(raw_obs["force_torque"][:3])
        p_force = 0.01 * max(0.0, ft_norm - 5.0)  # 超过5N才惩罚

        return 0.6 * r_dist + 0.4 * r_ori - p_smooth - p_force

    # ===========================================================
    #  工具方法
    # ===========================================================

    def _encode_language(self, text: str) -> np.ndarray:
        if self._lang_encoder is not None:
            return self._lang_encoder.encode(text)
        # 无编码器时返回零向量 (占位，便于后续替换)
        logger.warning("未提供语言编码器，language embedding 将为零向量")
        return np.zeros(LANG_TOKEN_DIM, dtype=np.float32)

    def _emergency_stop(self):
        logger.error("触发急停!")
        if self.robot_interface:
            self.robot_interface.emergency_stop()

    @staticmethod
    def _resize_image(img: np.ndarray, h: int, w: int) -> np.ndarray:
        try:
            import cv2
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            # fallback: 简单裁剪/填充
            return img[:h, :w]


# ===========================================================
#  模块级工具函数
# ===========================================================

def _pad_or_clip(arr: np.ndarray, target_len: int) -> np.ndarray:
    """将一维数组裁剪或零填充至目标长度。"""
    arr = arr.flatten()
    if len(arr) >= target_len:
        return arr[:target_len].astype(np.float32)
    return np.pad(arr, (0, target_len - len(arr))).astype(np.float32)
