"""
test_train_td3_lerobot.py — 使用模拟 LeRobot 数据测试 train_real_robot_td3.py 中的强化学习算法

测试范围
--------
1. 生成符合 LeRobot 格式的合成数据集
2. SimpleStateStandardizer / SimpleActionScaler 归一化工具
3. _build_proprio_from_episode 本体感觉向量构建
4. _populate_offline_buffer 离线 buffer 填充（使用 Mock replay buffer）
5. RealRobotBasePolicyEnvWrapper + Mock VLA 策略（reset / step）
6. 简化版主训练循环（warm-up 阶段 + 多步网络更新）
7. run_evaluation 评估函数

运行方式
--------
    cd /home/runner/work/real_robot_env/real_robot_env
    python test_train_td3_lerobot.py

依赖（均为轻量级或标准库）
--------
    numpy pandas pillow torch
    tensordict torchrl  （可选；缺失时使用 Mock replay buffer 跳过相关测试）
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# ---------------------------------------------------------------------------
#  把仓库目录加入 sys.path
# ---------------------------------------------------------------------------
REPO_DIR = Path(__file__).parent
sys.path.insert(0, str(REPO_DIR))

# ---------------------------------------------------------------------------
#  预注入 Mock 模块：在 import train_real_robot_td3 前拦截重型依赖
#
#  train_real_robot_td3.py 在模块顶层 import 了以下可能不存在的包：
#    tensordict, torchrl, wandb, resfit.*
#  通过向 sys.modules 注入轻量 stub，使 import 不会失败，
#  同时允许测试脚本直接使用这些函数中无需重型依赖的部分。
# ---------------------------------------------------------------------------
def _install_mock_modules() -> None:
    """将缺失的重型依赖替换为最小可用 stub。"""

    # ---- tensordict --------------------------------------------------------
    if "tensordict" not in sys.modules:
        td_mod = types.ModuleType("tensordict")

        class _TensorDict(dict):
            """极简 TensorDict stub（继承 dict，支持 [] 访问 & unsqueeze）。"""

            def __init__(self, data=None, batch_size=None):
                super().__init__(data or {})
                self._batch_size = batch_size or []

            def unsqueeze(self, dim):
                return self  # no-op for stub

            def to(self, *args, **kwargs):
                return self

            def __getitem__(self, key):
                return super().__getitem__(key)

            def __setitem__(self, key, value):
                super().__setitem__(key, value)

            @property
            def batch_size(self):
                return self._batch_size

        td_mod.TensorDict = _TensorDict
        sys.modules["tensordict"] = td_mod

    # ---- torchrl -----------------------------------------------------------
    if "torchrl" not in sys.modules:
        torchrl_mod  = types.ModuleType("torchrl")
        torchrl_data = types.ModuleType("torchrl.data")

        class _LazyTensorStorage:
            def __init__(self, max_size=1000, device="cpu"):
                self.max_size = max_size
                self._data: list = []

        class _TensorDictPrioritizedReplayBuffer:
            """最小 replay buffer stub，接口与真实类兼容。"""
            def __init__(self, storage=None, alpha=0.0, beta=0.0, eps=1e-6,
                         priority_key="_priority", transform=None,
                         pin_memory=False, prefetch=0, batch_size=32):
                self._storage = storage or _LazyTensorStorage()
                self._batch_size = batch_size
                self._data: list = []

            def add(self, td):
                self._data.append(td)
                if hasattr(self._storage, "_data"):
                    self._storage._data.append(td)

            def __len__(self):
                return len(self._data)

            def sample(self, batch_size=None):
                bs = batch_size or self._batch_size
                n  = max(len(self._data), 1)
                idxs = np.random.randint(0, n, size=min(bs, n))
                return [self._data[i] for i in idxs]

            def update_tensordict_priority(self, batch):
                pass

            def sampler(self):
                class _S:
                    def _empty(self): pass
                return _S()

            def state_dict(self):
                return {}

            def load_state_dict(self, sd):
                pass

        torchrl_data.LazyTensorStorage = _LazyTensorStorage
        torchrl_data.TensorDictPrioritizedReplayBuffer = _TensorDictPrioritizedReplayBuffer

        torchrl_mod.data = torchrl_data
        sys.modules["torchrl"]      = torchrl_mod
        sys.modules["torchrl.data"] = torchrl_data

    # ---- wandb -------------------------------------------------------------
    if "wandb" not in sys.modules:
        wandb_mod = types.ModuleType("wandb")
        wandb_mod.init    = lambda *a, **kw: None
        wandb_mod.log     = lambda *a, **kw: None
        wandb_mod.finish  = lambda *a, **kw: None
        wandb_mod.summary = {}
        sys.modules["wandb"] = wandb_mod

    # ---- resfit (多级子包) -----------------------------------------------
    _resfit_pkgs = [
        "resfit",
        "resfit.rl_finetuning",
        "resfit.rl_finetuning.off_policy",
        "resfit.rl_finetuning.off_policy.common_utils",
        "resfit.rl_finetuning.off_policy.common_utils.utils",
        "resfit.rl_finetuning.off_policy.rl",
        "resfit.rl_finetuning.off_policy.rl.q_agent",
        "resfit.rl_finetuning.utils",
        "resfit.rl_finetuning.utils.rb_transforms",
        "resfit.rl_finetuning.utils.hugging_face",
    ]
    for pkg in _resfit_pkgs:
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    # utils.schedule / utils.eval_mode
    utils_mod = sys.modules["resfit.rl_finetuning.off_policy.common_utils.utils"]
    if not hasattr(utils_mod, "schedule"):
        utils_mod.schedule = lambda schedule_str, step: 0.1
    if not hasattr(utils_mod, "eval_mode"):
        import contextlib

        @contextlib.contextmanager
        def _eval_mode(model):
            was_train = getattr(model, "_train_mode", True)
            if hasattr(model, "eval"):
                model.eval()
            try:
                yield
            finally:
                if was_train and hasattr(model, "train"):
                    model.train()

        utils_mod.eval_mode = _eval_mode

    # MultiStepTransform stub
    rb_transforms_mod = sys.modules["resfit.rl_finetuning.utils.rb_transforms"]
    if not hasattr(rb_transforms_mod, "MultiStepTransform"):
        class _MultiStepTransform:
            def __init__(self, n_steps=1, gamma=0.99): pass
        rb_transforms_mod.MultiStepTransform = _MultiStepTransform

    # QAgent stub（占位，让 train_real_robot_td3 import 时不报错）
    q_agent_mod = sys.modules["resfit.rl_finetuning.off_policy.rl.q_agent"]
    if not hasattr(q_agent_mod, "QAgent"):
        class _QAgentStub:
            def __init__(self, *a, **kw): pass
        q_agent_mod.QAgent = _QAgentStub

    # hugging_face stubs
    hf_mod = sys.modules["resfit.rl_finetuning.utils.hugging_face"]
    if not hasattr(hf_mod, "_hf_download_buffer"):
        hf_mod._hf_download_buffer          = lambda *a, **kw: None
        hf_mod._hf_upload_buffer            = lambda *a, **kw: None
        hf_mod.optimized_replay_buffer_dumps = lambda rb, path: None
        hf_mod.optimized_replay_buffer_loads = lambda rb, path: None


_install_mock_modules()

# ---------------------------------------------------------------------------
#  合成数据集参数
# ---------------------------------------------------------------------------
N_EPISODES  = 4     # 生成的 episode 数量
EPISODE_LEN = 25    # 每条 episode 的帧数
IMG_H, IMG_W = 224, 224
ACTION_DIM   = 7
STATE_DIM    = 18   # tcp(6) + joint_pos+vel(12)

TASK_TEXTS = [
    "pick up the red cube",
    "place the bottle on the shelf",
    "push the block to the left",
    "grasp the green bottle",
]


# ===========================================================================
#  工具：生成合成 LeRobot 数据集
# ===========================================================================

def _make_fake_image_bytes(h: int = IMG_H, w: int = IMG_W) -> bytes:
    """生成随机 PNG 图像的 bytes（用于 parquet 序列化）。"""
    import io
    from PIL import Image
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_lerobot_dataset(root: Path) -> None:
    """
    在 root 下生成 N_EPISODES 条轨迹，写成 LeRobot 标准目录结构。

    目录结构
    --------
    root/
      meta/episodes.json
      meta/info.json
      data/chunk-000/episode_000000.parquet
      ...
    """
    import pandas as pd

    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    episode_meta = []
    for ep_id in range(N_EPISODES):
        T   = EPISODE_LEN
        rng = np.random.default_rng(seed=ep_id + 100)

        states  = rng.uniform(-1.0, 1.0, (T, STATE_DIM)).astype(np.float32)
        actions = rng.uniform(-0.1, 0.1, (T, ACTION_DIM)).astype(np.float32)

        rows = []
        for t in range(T):
            rows.append({
                "frame_index":       t,
                "episode_index":     ep_id,
                "timestamp":         t / 10.0,
                "observation.state": states[t].tolist(),
                "action":            actions[t].tolist(),
                "observation.image": _make_fake_image_bytes(),
            })

        df = pd.DataFrame(rows)
        parquet_path = root / "data" / "chunk-000" / f"episode_{ep_id:06d}.parquet"
        df.to_parquet(parquet_path, index=False)

        episode_meta.append({
            "episode_index": ep_id,
            "task":          TASK_TEXTS[ep_id % len(TASK_TEXTS)],
            "length":        T,
        })

    with open(root / "meta" / "episodes.json", "w") as f:
        json.dump(episode_meta, f, indent=2)

    info = {
        "fps": 10,
        "n_episodes": N_EPISODES,
        "total_frames": N_EPISODES * EPISODE_LEN,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
            "action":            {"dtype": "float32", "shape": [ACTION_DIM]},
            "observation.image": {"dtype": "image"},
        },
    }
    with open(root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"[数据生成] {N_EPISODES} 条轨迹 × {EPISODE_LEN} 帧 → {root}")


# ===========================================================================
#  Mock 接口：模拟真机硬件
# ===========================================================================

class MockRobotInterface:
    """模拟机器人：返回安全固定观测，不触发安全保护。"""

    def __init__(self, tcp_pose=None):
        self._tcp_pose = np.array(
            tcp_pose if tcp_pose is not None else [0.5, 0.0, 0.3, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self._stopped = False

    def move_to_home(self):
        pass

    def send_eef_delta_command(self, delta, gripper=0.0, freq_hz=10.0):
        self._tcp_pose = self._tcp_pose + np.asarray(delta, dtype=np.float32)[:6] * 0.001

    def get_latest_observation(self):
        return {
            "tcp_pose":     self._tcp_pose.copy(),
            "joint_states": np.zeros(12, dtype=np.float32),
            "force_torque": np.zeros(6,  dtype=np.float32),
        }

    def stop(self):
        self._stopped = True

    def emergency_stop(self):
        self._stopped = True


class MockCameraInterface:
    """模拟相机：返回随机 uint8 RGB 图像。"""

    def __init__(self, h: int = IMG_H, w: int = IMG_W):
        self._h, self._w = h, w
        self._rng = np.random.default_rng(seed=0)

    def get_image(self) -> np.ndarray:
        return self._rng.integers(0, 255, (self._h, self._w, 3), dtype=np.uint8)


# ===========================================================================
#  Mock replay buffer（在 torchrl 不可用时替代）
# ===========================================================================

class MockReplayBuffer:
    """简单 list-based replay buffer，兼容 TensorDictPrioritizedReplayBuffer 接口。"""

    def __init__(self, batch_size: int = 32):
        self._storage: List[Any] = []
        self._batch_size = batch_size

    def add(self, td):
        self._storage.append(td)

    def __len__(self):
        return len(self._storage)

    def sample(self, batch_size: Optional[int] = None):
        bs = batch_size or self._batch_size
        idxs = np.random.randint(0, max(len(self._storage), 1), size=min(bs, max(len(self._storage), 1)))
        return [self._storage[i] for i in idxs]

    def update_tensordict_priority(self, batch):
        pass

    def sampler(self):
        pass


# ===========================================================================
#  Mock VLA 策略
# ===========================================================================

class MockVLAPolicy:
    """模拟 VLA 基础策略，随机生成 7 维动作，不需要 GPU/模型权重。"""

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed=seed)

    def predict_action(self, obs: Dict, task_text: str = "") -> np.ndarray:
        return self._rng.uniform(-0.05, 0.05, (ACTION_DIM,)).astype(np.float32)

    def batch_predict(self, obs_list: List[Dict], task_text: str = "") -> np.ndarray:
        return np.stack([self.predict_action(o, task_text) for o in obs_list])


# ===========================================================================
#  Mock QAgent（替换 resfit.QAgent，支持轻量测试）
# ===========================================================================

class MockQAgent:
    """
    轻量版 QAgent：
    - actor / critic 均为简单线性层，能进行真实梯度更新
    - 接口与 resfit.rl_finetuning.off_policy.rl.q_agent.QAgent 兼容
    """

    def __init__(self, action_dim: int = 7, state_dim: int = 24, device: str = "cpu"):
        self.action_dim = action_dim
        self.state_dim  = state_dim
        self.device     = torch.device(device)

        # 极简 actor / critic（MLP）
        self.actor  = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim),
            torch.nn.Tanh(),
        ).to(self.device)
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        ).to(self.device)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self._train_mode = True

    # ── Gymnasium eval_mode 上下文兼容 ──────────────────────────────
    def train(self):
        self._train_mode = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        self._train_mode = False
        self.actor.eval()
        self.critic.eval()

    def state_dict(self) -> Dict:
        return {
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    # ── act ─────────────────────────────────────────────────────────
    def act(
        self,
        obs: Dict[str, torch.Tensor],
        eval_mode: bool = False,
        stddev: float = 0.1,
        cpu: bool = False,
    ) -> torch.Tensor:
        """
        从 obs 提取 state 特征，通过 actor 生成残差动作（1, 7）。
        """
        # 取 observation.state，可能为 (1, 24) 或 (B, 24)
        state = obs.get("observation.state", None)
        if state is None:
            # 无 state 时退化为零残差
            action = torch.zeros(1, self.action_dim, device=self.device)
        else:
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float().to(self.device)
            else:
                state = state.float().to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            B = state.shape[0]
            # 拼接随机 base_action（测试阶段无真实 base）
            dummy_base = torch.zeros(B, self.action_dim, device=self.device)
            feat = torch.cat([state, dummy_base], dim=-1)  # (B, state+action)
            action = self.actor(feat)

        if not eval_mode and stddev > 0:
            noise  = torch.randn_like(action) * stddev
            action = torch.clamp(action + noise, -1.0, 1.0)

        if cpu:
            action = action.cpu()
        return action

    # ── update ──────────────────────────────────────────────────────
    def update(
        self,
        batch,
        stddev: float = 0.1,
        update_actor: bool = True,
        bc_batch=None,
        ref_agent=None,
    ) -> Dict[str, float]:
        """
        一步 TD3 风格更新（简化版，仅用于接口验证）。

        batch 可以是：
        - List[TensorDict]（MockReplayBuffer 返回）
        - TensorDict（torchrl 返回）
        """
        metrics: Dict[str, float] = {}

        # ── critic 更新 ──────────────────────────────────────────
        self.critic_opt.zero_grad()
        critic_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        # 对每条 transition 计算简单 MSE-to-zero 损失（仅测试梯度流动）
        if isinstance(batch, list) and len(batch) > 0:
            for td in batch:
                try:
                    obs_state = _extract_state_from_td(td, self.device)
                    if obs_state is not None:
                        dummy = self.critic(
                            torch.cat([obs_state, torch.zeros(obs_state.shape[0],
                                                              self.action_dim * 2,
                                                              device=self.device)], dim=-1)
                        )
                        critic_loss = critic_loss + dummy.mean() * 0.0  # 梯度流动
                except Exception:
                    pass
        critic_loss.backward()
        self.critic_opt.step()
        metrics["train/critic_loss"] = float(critic_loss.item())

        # ── actor 更新 ──────────────────────────────────────────
        if update_actor:
            self.actor_opt.zero_grad()
            dummy_state = torch.zeros(1, self.state_dim, device=self.device)
            dummy_base  = torch.zeros(1, self.action_dim, device=self.device)
            feat   = torch.cat([dummy_state, dummy_base], dim=-1)
            action = self.actor(feat)
            actor_loss = -action.abs().mean() * 0.0  # 梯度流动
            actor_loss.backward()
            self.actor_opt.step()
            metrics["train/actor_loss"] = float(actor_loss.item())

        return metrics


def _extract_state_from_td(td, device):
    """从 TensorDict 或 dict 中安全提取 observation.state。"""
    try:
        if hasattr(td, "__getitem__"):
            obs = td["obs"]
            if hasattr(obs, "__getitem__"):
                state = obs["observation.state"]
                if isinstance(state, torch.Tensor):
                    s = state.float().to(device)
                    if s.dim() == 1:
                        s = s.unsqueeze(0)
                    return s
    except Exception:
        pass
    return None


# ===========================================================================
#  测试 1：归一化工具  SimpleStateStandardizer / SimpleActionScaler
# ===========================================================================

def test_normalizers(dataset_loader) -> None:
    print("\n===== Test 1: SimpleStateStandardizer / SimpleActionScaler =====")

    # 动态 import（依赖 torch 即可，无需 resfit）
    from train_real_robot_td3 import SimpleActionScaler, SimpleStateStandardizer

    # ---- SimpleStateStandardizer ----
    std_from_ds = SimpleStateStandardizer.from_dataset(dataset_loader, device="cpu")
    assert std_from_ds.mean.shape == (12,), f"standardizer mean 形状错: {std_from_ds.mean.shape}"

    dummy_state = torch.ones(12, dtype=torch.float32)
    normed = std_from_ds.standardize(dummy_state)
    assert normed.shape == (12,), f"standardize 输出形状错: {normed.shape}"
    print(f"  Standardizer 均值范围: [{std_from_ds.mean.min():.4f}, {std_from_ds.mean.max():.4f}]")

    identity_std = SimpleStateStandardizer.identity(dim=24, device="cpu")
    vec = torch.tensor(np.ones(24, dtype=np.float32))
    assert torch.allclose(identity_std.standardize(vec), vec), "恒等标准化应返回原值"

    # ---- SimpleActionScaler ----
    scaler_from_ds = SimpleActionScaler.from_dataset(dataset_loader, device="cpu")
    assert scaler_from_ds.bias.shape == (ACTION_DIM,), f"scaler bias 形状错: {scaler_from_ds.bias.shape}"

    raw_action = torch.tensor(np.zeros(ACTION_DIM, dtype=np.float32))
    scaled = scaler_from_ds.scale(raw_action)
    unscaled = scaler_from_ds.unscale(scaled)
    assert torch.allclose(unscaled, raw_action, atol=1e-5), "缩放-反缩放应为恒等变换"
    print(f"  ActionScaler bias: {scaler_from_ds.bias.numpy()}")

    identity_scaler = SimpleActionScaler.identity(dim=ACTION_DIM, device="cpu")
    v = torch.tensor(np.zeros(ACTION_DIM, dtype=np.float32))
    assert torch.allclose(identity_scaler.scale(v), v), "恒等缩放应返回原值"

    print("  [PASS]")


# ===========================================================================
#  测试 2：_build_proprio_from_episode
# ===========================================================================

def test_build_proprio(dataset_loader) -> None:
    print("\n===== Test 2: _build_proprio_from_episode =====")

    from train_real_robot_td3 import _build_proprio_from_episode

    ep = dataset_loader.get_episode(0)
    proprio = _build_proprio_from_episode(ep)

    assert proprio.ndim == 2,  f"应为 2D (T, 24)，实为 {proprio.shape}"
    assert proprio.shape[0] == ep.length, f"时间步不匹配: {proprio.shape[0]} != {ep.length}"
    assert proprio.shape[1] == 24, f"proprio 维度应为 24，实为 {proprio.shape[1]}"
    assert proprio.dtype == np.float32, f"dtype 应为 float32，实为 {proprio.dtype}"

    print(f"  proprio shape: {proprio.shape}  dtype: {proprio.dtype}")
    print(f"  proprio[0] = {proprio[0]}")
    print("  [PASS]")


# ===========================================================================
#  测试 3：_populate_offline_buffer（使用 Mock replay buffer）
# ===========================================================================

def test_populate_offline_buffer(dataset_loader) -> None:
    print("\n===== Test 3: _populate_offline_buffer =====")

    from train_real_robot_td3 import (
        SimpleActionScaler,
        SimpleStateStandardizer,
        _populate_offline_buffer,
    )

    # 检查 torchrl / tensordict 是否可用
    try:
        from tensordict import TensorDict
        from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
        from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform

        # 构建真实 replay buffer
        n_transitions = N_EPISODES * (EPISODE_LEN - 1)
        rb = TensorDictPrioritizedReplayBuffer(
            storage=LazyTensorStorage(max_size=n_transitions, device="cpu"),
            alpha=0.0,
            beta=0.0,
            eps=1e-6,
            priority_key="_priority",
            transform=MultiStepTransform(n_steps=1, gamma=0.99),
            pin_memory=False,
            prefetch=0,
            batch_size=32,
        )
        using_real_rb = True
        print("  使用真实 TensorDictPrioritizedReplayBuffer")
    except ImportError:
        # 回退到 Mock replay buffer
        rb = MockReplayBuffer(batch_size=32)
        using_real_rb = False
        print("  torchrl/resfit 不可用，使用 MockReplayBuffer")

    action_scaler     = SimpleActionScaler.identity(dim=ACTION_DIM, device="cpu")
    state_standardizer = SimpleStateStandardizer.identity(dim=24, device="cpu")
    lang_embed        = np.zeros(512, dtype=np.float32)
    camera_key        = "observation.images.top"
    image_keys        = [camera_key]

    added = _populate_offline_buffer(
        dataset_loader=dataset_loader,
        rb=rb,
        image_keys=image_keys,
        camera_key=camera_key,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        lang_embed=lang_embed,
        num_episodes=None,
        use_base_policy=False,
        base_policy=None,
    )

    expected = N_EPISODES * (EPISODE_LEN - 1)
    assert added == expected, f"transitions 数量不符: {added} != {expected}"
    assert len(rb) == expected, f"buffer 大小不符: {len(rb)} != {expected}"
    print(f"  填充 transitions 数量: {added}  （期望 {expected}）")

    # 验证 buffer 中的数据结构
    if using_real_rb:
        sample = rb.sample(8)
        assert sample is not None
        print(f"  抽样 batch 成功，类型: {type(sample)}")
    else:
        sample = rb.sample(8)
        assert len(sample) > 0
        td0 = sample[0]
        assert "obs"    in td0, "缺少 obs 键"
        assert "action" in td0, "缺少 action 键"
        assert "next"   in td0, "缺少 next 键"
        print(f"  抽样 transition keys: {list(td0.keys())}")

    print("  [PASS]")


# ===========================================================================
#  测试 4：RealRobotBasePolicyEnvWrapper reset / step（Online 模式）
# ===========================================================================

def test_env_wrapper_online() -> None:
    print("\n===== Test 4: RealRobotBasePolicyEnvWrapper (ONLINE) =====")

    from env import EnvMode, RealRobotEnv
    from real_robot_env_wrapper import RealRobotBasePolicyEnvWrapper
    from train_real_robot_td3 import SimpleStateStandardizer

    target_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    robot  = MockRobotInterface(tcp_pose=target_pose)
    camera = MockCameraInterface()

    raw_env = RealRobotEnv(
        task_language="pick up the red cube",
        target_pose=target_pose,
        mode=EnvMode.ONLINE,
        robot_interface=robot,
        camera_interface=camera,
        max_steps=20,
    )

    mock_policy     = MockVLAPolicy()
    state_std       = SimpleStateStandardizer.identity(dim=24, device="cpu")
    camera_key      = "observation.images.top"

    # GenericVLAWrapper 包装 MockVLAPolicy
    from vla_interface import GenericVLAWrapper
    vla_wrapper = GenericVLAWrapper(
        model=None,
        predict_fn=lambda proc_obs, task_text: mock_policy.predict_action({}, task_text),
    )

    env = RealRobotBasePolicyEnvWrapper(
        env=raw_env,
        base_policy=vla_wrapper,
        camera_key=camera_key,
        state_standardizer=state_std,
        device="cpu",
    )
    env.seed(0)

    # ---- reset ----
    obs, info = env.reset()
    _check_obs(obs, camera_key)
    print(f"  reset() obs keys: {list(obs.keys())}")
    print(f"  image shape:  {obs[camera_key].shape}")
    print(f"  state shape:  {obs['observation.state'].shape}")
    print(f"  base_action shape: {obs['observation.base_action'].shape}")

    # ---- step × 10 ----
    total_reward = 0.0
    step_count   = 0
    for _ in range(10):
        residual = torch.zeros(1, ACTION_DIM)
        obs, reward, terminated, truncated, info = env.step(residual)
        _check_obs(obs, camera_key)
        assert "scaled_action" in info, "info 缺少 scaled_action"
        total_reward += reward.item()
        step_count   += 1
        done = (terminated | truncated).item()
        if done:
            break

    print(f"  完成 {step_count} 步，总奖励: {total_reward:.4f}")
    env.close()
    print("  [PASS]")


def _check_obs(obs: Dict[str, torch.Tensor], camera_key: str) -> None:
    assert camera_key in obs,               f"obs 缺少 {camera_key}"
    assert "observation.state"       in obs, "obs 缺少 observation.state"
    assert "observation.base_action" in obs, "obs 缺少 observation.base_action"
    assert obs[camera_key].shape      == (1, 3, IMG_H, IMG_W),  \
        f"image shape 错误: {obs[camera_key].shape}"
    assert obs["observation.state"].shape   == (1, 24), \
        f"state shape 错误: {obs['observation.state'].shape}"
    assert obs["observation.base_action"].shape == (1, ACTION_DIM), \
        f"base_action shape 错误: {obs['observation.base_action'].shape}"


# ===========================================================================
#  测试 5：RealRobotBasePolicyEnvWrapper（OFFLINE 模式）
# ===========================================================================

def test_env_wrapper_offline(dataset_root: Path) -> None:
    print("\n===== Test 5: RealRobotBasePolicyEnvWrapper (OFFLINE) =====")

    from env import EnvMode, RealRobotEnv
    from real_robot_env_wrapper import RealRobotBasePolicyEnvWrapper
    from vla_interface import GenericVLAWrapper
    from train_real_robot_td3 import SimpleStateStandardizer

    target_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    raw_env = RealRobotEnv(
        task_language="pick up the red cube",
        target_pose=target_pose,
        mode=EnvMode.OFFLINE,
        dataset_path=str(dataset_root),
        dataset_fmt="lerobot",
        max_steps=EPISODE_LEN,
    )

    mock_policy = MockVLAPolicy()
    vla_wrapper = GenericVLAWrapper(
        model=None,
        predict_fn=lambda proc_obs, task_text: mock_policy.predict_action({}, task_text),
    )
    state_std  = SimpleStateStandardizer.identity(dim=24, device="cpu")
    camera_key = "observation.images.top"

    env = RealRobotBasePolicyEnvWrapper(
        env=raw_env,
        base_policy=vla_wrapper,
        camera_key=camera_key,
        state_standardizer=state_std,
        device="cpu",
    )
    env.seed(42)

    obs, info = env.reset()
    _check_obs(obs, camera_key)
    print(f"  reset() obs keys: {list(obs.keys())}")

    step_count = 0
    for _ in range(EPISODE_LEN):
        residual = torch.zeros(1, ACTION_DIM)
        obs, reward, terminated, truncated, info = env.step(residual)
        _check_obs(obs, camera_key)
        step_count += 1
        if (terminated | truncated).item():
            break

    print(f"  完成 {step_count} 步")
    env.close()
    print("  [PASS]")


# ===========================================================================
#  测试 6：简化版主训练循环（Mock Agent + Mock 环境）
# ===========================================================================

def test_training_loop(dataset_root: Path) -> None:
    """
    不依赖 resfit / wandb，使用 MockQAgent 模拟完整的训练循环：
        warm-up → 填充离线 buffer → 多步 critic/actor 更新
    """
    print("\n===== Test 6: 简化版主训练循环 =====")

    from env import EnvMode, RealRobotEnv
    from real_robot_env_wrapper import RealRobotBasePolicyEnvWrapper
    from vla_interface import GenericVLAWrapper
    from train_real_robot_td3 import (
        SimpleActionScaler,
        SimpleStateStandardizer,
        _add_transitions_to_buffer,
        _populate_offline_buffer,
    )
    from dataset_utils import load_dataset

    device    = torch.device("cpu")
    device_str = "cpu"
    camera_key = "observation.images.top"
    image_keys = [camera_key]
    lowdim_keys = ["observation.state", "observation.base_action"]
    action_dim  = ACTION_DIM
    seed = 0

    # ── 1. 数据集 & 归一化 ──────────────────────────────────────────────
    dataset_loader = load_dataset(str(dataset_root), fmt="lerobot")
    action_scaler  = SimpleActionScaler.identity(dim=action_dim, device=device_str)
    state_std      = SimpleStateStandardizer.identity(dim=24, device=device_str)
    lang_embed     = np.zeros(512, dtype=np.float32)

    # ── 2. 构建 Online 环境 ─────────────────────────────────────────────
    target_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    robot  = MockRobotInterface(tcp_pose=target_pose)
    camera = MockCameraInterface()
    raw_env = RealRobotEnv(
        task_language="pick up the red cube",
        target_pose=target_pose,
        mode=EnvMode.ONLINE,
        robot_interface=robot,
        camera_interface=camera,
        max_steps=20,
    )

    mock_policy = MockVLAPolicy(seed=seed)
    vla_wrapper = GenericVLAWrapper(
        model=None,
        predict_fn=lambda proc_obs, task_text: mock_policy.predict_action({}, task_text),
    )
    env = RealRobotBasePolicyEnvWrapper(
        env=raw_env,
        base_policy=vla_wrapper,
        camera_key=camera_key,
        state_standardizer=state_std,
        device=device_str,
    )
    env.seed(seed)

    # ── 3. Mock replay buffers ─────────────────────────────────────────
    online_rb  = MockReplayBuffer(batch_size=8)
    offline_rb = MockReplayBuffer(batch_size=8)

    # ── 4. 填充 offline buffer ─────────────────────────────────────────
    added = _populate_offline_buffer(
        dataset_loader=dataset_loader,
        rb=offline_rb,
        image_keys=image_keys,
        camera_key=camera_key,
        action_scaler=action_scaler,
        state_standardizer=state_std,
        lang_embed=lang_embed,
        num_episodes=None,
        use_base_policy=False,
        base_policy=None,
    )
    print(f"  Offline buffer 填充完成: {added} transitions")
    assert added == N_EPISODES * (EPISODE_LEN - 1)

    # ── 5. Warm-up：收集 online 数据 ───────────────────────────────────
    WARMUP_STEPS = 15
    obs, _ = env.reset()
    for step in range(WARMUP_STEPS):
        residual = torch.randn(1, action_dim) * 0.1
        next_obs, reward, terminated, truncated, info = env.step(residual)
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
            num_envs=1,
            online_rb=online_rb,
        )
        obs = next_obs

    print(f"  Warm-up 完成，online buffer size = {len(online_rb)}")
    assert len(online_rb) == WARMUP_STEPS, f"online buffer 大小不符: {len(online_rb)}"

    # ── 6. 构建 MockQAgent ─────────────────────────────────────────────
    agent = MockQAgent(action_dim=action_dim, state_dim=24, device=device_str)

    # ── 7. 主训练循环（简化：10 步环境交互 + 10 次 agent 更新）──────────
    TRAIN_STEPS    = 10
    AGENT_UPDATES  = 10
    OFFLINE_FRAC   = 0.5
    BATCH_SIZE     = 8
    online_bs  = max(int(BATCH_SIZE * (1 - OFFLINE_FRAC)), 1)
    offline_bs = max(int(BATCH_SIZE * OFFLINE_FRAC),       1)

    obs, _ = env.reset()
    episode_rewards = []
    ep_reward = 0.0

    for step in range(TRAIN_STEPS):
        # 环境交互
        with torch.no_grad():
            agent.eval()
            residual = agent.act(obs, eval_mode=False, stddev=0.1)
            agent.train()

        next_obs, reward, terminated, truncated, info = env.step(residual)
        done = terminated | truncated
        ep_reward += reward.item()

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
            num_envs=1,
            online_rb=online_rb,
        )
        obs = next_obs

        if done.item():
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()

    print(f"  训练环境交互 {TRAIN_STEPS} 步，完成 {len(episode_rewards)} 个 episode")

    for upd in range(AGENT_UPDATES):
        online_batch  = online_rb.sample(online_bs)
        offline_batch = offline_rb.sample(offline_bs)
        merged_batch  = online_batch + offline_batch

        update_actor = (upd % 2 == 0)
        metrics = agent.update(merged_batch, stddev=0.1, update_actor=update_actor)

        if upd == 0:
            print(f"  首次更新 metrics: { {k: f'{v:.4f}' for k, v in metrics.items()} }")

    print(f"  Agent 更新 {AGENT_UPDATES} 次完成")
    env.close()
    print("  [PASS]")


# ===========================================================================
#  测试 7：run_evaluation（使用 Mock QAgent + Mock 环境）
# ===========================================================================

def test_run_evaluation() -> None:
    print("\n===== Test 7: run_evaluation =====")

    from env import EnvMode, RealRobotEnv
    from real_robot_env_wrapper import RealRobotBasePolicyEnvWrapper
    from vla_interface import GenericVLAWrapper
    from train_real_robot_td3 import SimpleStateStandardizer, run_evaluation

    target_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    robot  = MockRobotInterface(tcp_pose=target_pose)
    camera = MockCameraInterface()
    raw_env = RealRobotEnv(
        task_language="pick up the red cube",
        target_pose=target_pose,
        mode=EnvMode.ONLINE,
        robot_interface=robot,
        camera_interface=camera,
        max_steps=10,
    )

    mock_policy = MockVLAPolicy()
    vla_wrapper = GenericVLAWrapper(
        model=None,
        predict_fn=lambda proc_obs, task_text: mock_policy.predict_action({}, task_text),
    )
    state_std  = SimpleStateStandardizer.identity(dim=24, device="cpu")
    camera_key = "observation.images.top"

    env = RealRobotBasePolicyEnvWrapper(
        env=raw_env,
        base_policy=vla_wrapper,
        camera_key=camera_key,
        state_standardizer=state_std,
        device="cpu",
    )
    env.seed(1)

    agent  = MockQAgent(action_dim=ACTION_DIM, state_dim=24, device="cpu")

    # 给 agent 加上 utils.eval_mode 兼容的上下文管理器
    # run_evaluation 使用 `utils.eval_mode(agent)` —— 我们注入 Mock 实现
    import contextlib

    @contextlib.contextmanager
    def _mock_eval_mode(model):
        was_training = model._train_mode
        model.eval()
        try:
            yield
        finally:
            if was_training:
                model.train()

    # Patch utils.eval_mode
    try:
        from resfit.rl_finetuning.off_policy.common_utils import utils as resfit_utils
        original_eval_mode = resfit_utils.eval_mode
        resfit_utils.eval_mode = _mock_eval_mode
    except ImportError:
        # resfit 不可用：用 patch 注入
        with patch("train_real_robot_td3.utils") as mock_utils:
            mock_utils.eval_mode = _mock_eval_mode
            mock_utils.schedule.return_value = 0.1

            metrics = run_evaluation(
                env_wrapper=env,
                agent=agent,
                num_episodes=2,
                device=torch.device("cpu"),
                global_step=0,
                save_video=False,
                run_name="test",
                output_dir=None,
            )
            _check_eval_metrics(metrics)
            env.close()
            print("  [PASS] (resfit 已 mock)")
            return

    metrics = run_evaluation(
        env_wrapper=env,
        agent=agent,
        num_episodes=2,
        device=torch.device("cpu"),
        global_step=0,
        save_video=False,
        run_name="test",
        output_dir=None,
    )
    _check_eval_metrics(metrics)
    resfit_utils.eval_mode = original_eval_mode
    env.close()
    print("  [PASS]")


def _check_eval_metrics(metrics: Dict[str, float]) -> None:
    expected_keys = ("eval/success_rate", "eval/mean_reward",
                     "eval/mean_length", "eval/mean_pos_error")
    for k in expected_keys:
        assert k in metrics, f"eval metrics 缺少键: {k}"
    print(f"  eval metrics: { {k: f'{v:.4f}' for k, v in metrics.items()} }")


# ===========================================================================
#  主入口
# ===========================================================================

def main() -> None:
    print("=" * 65)
    print("  train_real_robot_td3.py  ←  模拟 LeRobot 数据  RL 算法测试")
    print("=" * 65)

    with tempfile.TemporaryDirectory(prefix="td3_lerobot_test_") as tmp:
        tmp_dir    = Path(tmp)
        ds_root    = tmp_dir / "lerobot_dataset"
        output_dir = tmp_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成合成 LeRobot 数据集
        create_lerobot_dataset(ds_root)

        # 加载 DatasetLoader
        from dataset_utils import LeRobotDatasetLoader
        loader = LeRobotDatasetLoader(str(ds_root))
        print(f"数据集加载完成：{len(loader)} 条轨迹 × {EPISODE_LEN} 帧")

        # 逐项测试
        test_normalizers(loader)
        test_build_proprio(loader)
        test_populate_offline_buffer(loader)
        test_env_wrapper_online()
        test_env_wrapper_offline(ds_root)
        test_training_loop(ds_root)
        test_run_evaluation()

    print("\n" + "=" * 65)
    print("  全部测试通过 ✓")
    print("=" * 65)


if __name__ == "__main__":
    main()
