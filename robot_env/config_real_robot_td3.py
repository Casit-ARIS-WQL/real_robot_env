"""
config_real_robot_td3.py  ——  真机 Residual TD3 训练配置

所有超参数通过 Python dataclass 管理，便于在脚本中直接实例化或通过 JSON/YAML 覆盖。

快速入门
--------
    from config_real_robot_td3 import ResidualTD3RealRobotConfig

    cfg = ResidualTD3RealRobotConfig()
    cfg.task_language = "pick up the red block"
    cfg.offline_data.dataset_path = "/data/robot_demos.h5"
    cfg.algo.offline_fraction = 0.5   # online+offline 混合训练
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ===========================================================================
#  硬件与安全配置
# ===========================================================================


@dataclass
class RobotConfig:
    """真机硬件连接配置。"""

    ip: str = "192.168.1.100"
    port: int = 50051
    control_freq_hz: float = 10.0
    # 回 home 位时各关节目标角度（None=使用机器人默认 home 位）
    home_position: Optional[List[float]] = None


@dataclass
class CameraConfig:
    """主相机配置。"""

    camera_type: str = "realsense"   # "realsense" | "usb" | "virtual"
    width: int = 224
    height: int = 224
    fps: int = 30
    # 在观测字典中的键名（需与 rl_camera 保持一致）
    obs_key: str = "observation.images.top"


@dataclass
class SafetyConfig:
    """
    真机安全边界配置。
    与 RealRobotEnv 的 safety_config 字段一一对应。
    """

    max_force_norm: float = 50.0     # 末端合力上限 (N)
    min_z: float = 0.05              # 末端最低高度 (m)，防碰桌面
    max_joint_vel: float = 1.5       # 关节角速度上限 (rad/s)
    max_action_delta: float = 0.1    # 单步动作增量上限 (rad 或 m)

    def to_dict(self) -> dict:
        return {
            "max_force_norm": self.max_force_norm,
            "min_z": self.min_z,
            "max_joint_vel": self.max_joint_vel,
            "max_action_delta": self.max_action_delta,
        }


# ===========================================================================
#  数据集与归一化配置
# ===========================================================================


@dataclass
class OfflineDataConfig:
    """离线数据集配置。"""

    dataset_path: Optional[str] = None
    # 数据格式："auto"=自动检测; "hdf5" | "lerobot" | "rlds"
    dataset_fmt: str = "auto"
    # None=加载全部；正整数=仅加载前 N 条 episode
    num_episodes: Optional[int] = None
    # True=用 VLA 模型推断 base_action；False=用专家 GT 动作作为 base_action
    use_base_policy_for_base_actions: bool = False
    # 动作归一化：每维度最小量程，防止除零
    min_action_range: float = 0.01
    # 状态标准化：每维度标准差下限
    min_state_std: float = 0.01


# ===========================================================================
#  强化学习算法超参数
# ===========================================================================


@dataclass
class AlgoConfig:
    """Residual TD3 算法超参数。"""

    total_timesteps: int = 100_000
    batch_size: int = 256
    # 每个 mini-batch 中来自离线 buffer 的比例
    # 0.0 = 纯在线；1.0 = 纯离线；(0, 1) = 混合
    offline_fraction: float = 0.5
    # warm-up 阶段随机动作收集的步数（填充 online buffer）
    learning_starts: int = 1_000
    # online replay buffer 最大容量（步数）
    buffer_size: int = 100_000

    # TD3 核心超参
    gamma: float = 0.99
    n_step: int = 1
    update_every_n_steps: int = 1
    num_updates_per_iteration: int = 1
    actor_updates_per_iteration: int = 1
    # stddev 调度：格式 "linear(start, end, steps)" 或 "constant(v)"
    stddev_schedule: str = "linear(0.3,0.1,50000)"

    # 学习率预热
    actor_lr_warmup_steps: int = 0
    critic_warmup_steps: int = 0

    # 渐进动作裁剪（防止早期大幅残差破坏机器人）
    progressive_clipping_steps: int = 0

    # warm-up 探索配置
    random_action_noise_scale: float = 0.3
    use_base_policy_for_warmup: bool = True

    # Replay buffer 优化
    prefetch_batches: int = 2
    # "uniform" | "prioritized_replay"
    sampling_strategy: str = "uniform"
    priority_alpha: float = 0.6
    priority_beta: float = 0.4


# ===========================================================================
#  网络与 Agent 配置
# ===========================================================================


@dataclass
class ActorNetConfig:
    """Actor 网络结构配置（传递给 QAgent）。"""

    action_scale: float = 1.0
    hidden_dim: int = 256
    num_layers: int = 3


@dataclass
class AgentConfig:
    """QAgent 整体配置。"""

    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    clip_q_target_to_reward_range: bool = False
    actor: ActorNetConfig = field(default_factory=ActorNetConfig)


# ===========================================================================
#  日志配置
# ===========================================================================


@dataclass
class WandbConfig:
    """Weights & Biases 日志配置。"""

    project: str = "real-robot-residual-td3"
    entity: Optional[str] = None
    name: Optional[str] = None
    # "online" | "offline" | "disabled"
    mode: str = "online"
    notes: Optional[str] = None
    group: Optional[str] = None
    continue_run_id: Optional[str] = None


# ===========================================================================
#  顶层配置
# ===========================================================================


@dataclass
class ResidualTD3RealRobotConfig:
    """
    真机 Residual TD3 完整配置。

    三种训练模式（通过 algo.offline_fraction 控制）
    -----------------------------------------------
    纯离线 (Offline-only)  : offline_fraction = 1.0
        仅从 VLA 数据集中采样 batch，不进行真机交互。
        适用于算法验证阶段。

    纯在线 (Online-only)   : offline_fraction = 0.0
        仅使用真机交互数据。需要先完成 warm-up 探索。

    混合 (Mixed, 推荐)    : 0 < offline_fraction < 1.0
        先用 VLA 数据集预填充离线 buffer，再用真机交互填充在线 buffer，
        按比例混合采样更新 QAgent。
    """

    # ── 任务描述 ─────────────────────────────────────────────────
    task_language: str = "pick up the object and place it on the target"
    # 目标末端位姿 [x, y, z, rx, ry, rz]（单位：m, rad）
    target_pose: List[float] = field(
        default_factory=lambda: [0.5, 0.0, 0.3, 0.0, 0.0, 0.0]
    )

    # ── 硬件接口 ──────────────────────────────────────────────────
    robot: RobotConfig = field(default_factory=RobotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # ── VLA 基础策略 ──────────────────────────────────────────────
    # HuggingFace Hub 路径或本地目录，例如 "openvla/openvla-7b"
    vla_model_path: str = "openvla/openvla-7b"
    vla_device: str = "cuda"
    # 用于区分不同算法变体的运行标识符
    actor_name: str = "residual_vla"

    # ── RL 训练 ───────────────────────────────────────────────────
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    offline_data: OfflineDataConfig = field(default_factory=OfflineDataConfig)

    # ── 环境 ──────────────────────────────────────────────────────
    max_steps: int = 500        # 每个 episode 最大交互步数
    num_envs: int = 1           # 真机固定为 1（不支持向量化并行）
    # 用于 QAgent 的相机观测键（需与 camera.obs_key 保持一致）
    rl_camera: str = "observation.images.top"

    # ── 评估 ──────────────────────────────────────────────────────
    eval_num_episodes: int = 5
    eval_interval_every_steps: int = 5_000
    # True=训练第 0 步时立即评估
    eval_first: bool = False
    save_video: bool = False

    # ── 日志 ──────────────────────────────────────────────────────
    log_freq: int = 100
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # ── 其他 ──────────────────────────────────────────────────────
    seed: Optional[int] = None
    torch_deterministic: bool = False
    debug: bool = False
