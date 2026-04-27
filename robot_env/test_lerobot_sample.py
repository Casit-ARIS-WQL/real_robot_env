"""
test_lerobot_sample.py — 使用合成 LeRobot 样例数据测试 RealRobotEnv 离线/在线流程

测试范围
--------
1. 生成符合 LeRobot 格式的合成数据（meta/episodes.json + parquet 文件）
2. LeRobotDatasetLoader 加载 & EpisodeData 字段校验
3. RealRobotEnv (OFFLINE 模式) reset / step / render
4. episode_batch_iter() mini-batch 迭代
5. compute_dataset_stats() 统计工具
6. RealRobotEnv (ONLINE 模式) reset / step（使用 Mock 接口模拟真机）
7. RolloutCollector (ONLINE 模式) 使用 Mock VLA 策略收集轨迹

运行方式
--------
    cd /home/runner/work/real_robot_env/real_robot_env
    python test_lerobot_sample.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# 把仓库目录加入 sys.path，确保能直接 import
sys.path.insert(0, str(Path(__file__).parent))

from dataset_utils import (
    LeRobotDatasetLoader,
    compute_dataset_stats,
    load_dataset,
)
from env import EnvMode, RealRobotEnv
from vla_interface import GenericVLAWrapper, RolloutCollector

# ---------------------------------------------------------------------------
#  可调参数
# ---------------------------------------------------------------------------
N_EPISODES   = 3    # 生成几条轨迹
EPISODE_LEN  = 20   # 每条轨迹帧数
IMG_H, IMG_W = 224, 224   # 与 env.py 期望一致，避免依赖 cv2 做 resize
STATE_DIM    = 18   # tcp(6) + joint(12)
ACTION_DIM   = 7    # [Δx,Δy,Δz,Δrx,Δry,Δrz,gripper]
# JOINT_DIM: 关节位置 6 个 + 关节速度 6 个 = 12 (与 env.py PROPRIO_DIM 拆分对应)
JOINT_DIM    = STATE_DIM - 6   # == 12
PROPRIO_DIM  = 24  # tcp(6) + joint_pos+vel(12) + ft(6)，与 env.py 定义一致
LANG_DIM     = 512 # 语言嵌入维度，与 env.py LANG_TOKEN_DIM 一致

TASK_TEXTS = [
    "pick up the red cube",
    "place the bottle on the shelf",
    "push the block to the left",
]


# ---------------------------------------------------------------------------
#  工具：生成合成 LeRobot 格式数据集
# ---------------------------------------------------------------------------

def _make_fake_image_bytes(h: int = IMG_H, w: int = IMG_W) -> bytes:
    """生成随机 PNG 图像的 bytes。"""
    import io
    from PIL import Image  # 需要 Pillow
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_fake_image_array(h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    """返回随机 uint8 RGB 数组 (H, W, 3)。"""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def create_lerobot_sample_dataset(root: Path, use_image_bytes: bool = False) -> None:
    """
    在 root 目录下生成 N_EPISODES 条轨迹，写成 LeRobot 标准格式。

    目录结构
    --------
    root/
      meta/
        episodes.json
        info.json
      data/
        chunk-000/
          episode_000000.parquet
          episode_000001.parquet
          ...
    """
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    episode_meta = []

    for ep_id in range(N_EPISODES):
        T = EPISODE_LEN
        rng = np.random.default_rng(seed=ep_id)

        # ---- 构造每帧数据 ----
        states  = rng.uniform(-1.0, 1.0, (T, STATE_DIM)).astype(np.float32)
        actions = rng.uniform(-0.1, 0.1, (T, ACTION_DIM)).astype(np.float32)

        rows = []
        for t in range(T):
            row = {
                "frame_index":       t,
                "episode_index":     ep_id,
                "timestamp":         t / 10.0,         # 10 Hz
                "observation.state": states[t].tolist(),
                "action":            actions[t].tolist(),
            }
            # 图像列：存为 bytes（PNG）以确保 parquet 序列化/反序列化的一致性
            if use_image_bytes:
                row["observation.image"] = _make_fake_image_bytes()
            else:
                # 默认也使用 bytes，parquet 不能直接存储多维 numpy 数组
                row["observation.image"] = _make_fake_image_bytes()

            rows.append(row)

        df = pd.DataFrame(rows)
        parquet_path = root / "data" / "chunk-000" / f"episode_{ep_id:06d}.parquet"
        df.to_parquet(parquet_path, index=False)

        episode_meta.append({
            "episode_index": ep_id,
            "task":          TASK_TEXTS[ep_id % len(TASK_TEXTS)],
            "length":        T,
        })

    # ---- meta/episodes.json ----
    with open(root / "meta" / "episodes.json", "w") as f:
        json.dump(episode_meta, f, indent=2)

    # ---- meta/info.json（可选元信息）----
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


# ---------------------------------------------------------------------------
#  测试 1: DatasetLoader 加载与字段校验
# ---------------------------------------------------------------------------

def test_loader(root: Path) -> LeRobotDatasetLoader:
    print("\n===== Test 1: LeRobotDatasetLoader =====")
    loader = LeRobotDatasetLoader(str(root))
    print(f"  轨迹数量: {len(loader)}  (期望 {N_EPISODES})")
    assert len(loader) == N_EPISODES, f"轨迹数量不符: {len(loader)} != {N_EPISODES}"

    ep0 = loader.get_episode(0)
    print(f"  episode[0] 长度: {ep0.length}  (期望 {EPISODE_LEN})")
    assert ep0.length == EPISODE_LEN

    print(f"  images shape      : {ep0.images.shape}       dtype={ep0.images.dtype}")
    print(f"  tcp_poses shape   : {ep0.tcp_poses.shape}    dtype={ep0.tcp_poses.dtype}")
    print(f"  joint_states shape: {ep0.joint_states.shape} dtype={ep0.joint_states.dtype}")
    print(f"  actions shape     : {ep0.actions.shape}      dtype={ep0.actions.dtype}")
    print(f"  language          : '{ep0.language}'")

    assert ep0.images.ndim       == 4,  "images 应为 4D (T,H,W,3)"
    assert ep0.tcp_poses.shape   == (EPISODE_LEN, 6)
    assert ep0.joint_states.shape[0] == EPISODE_LEN
    assert ep0.actions.shape     == (EPISODE_LEN, ACTION_DIM)
    assert isinstance(ep0.language, str) and len(ep0.language) > 0

    # load_dataset 自动检测
    loader2 = load_dataset(str(root), fmt="auto")
    assert len(loader2) == N_EPISODES
    print("  load_dataset(fmt='auto') 自动检测正常")

    print("  [PASS]")
    return loader


# ---------------------------------------------------------------------------
#  测试 2: compute_dataset_stats
# ---------------------------------------------------------------------------

def test_stats(loader: LeRobotDatasetLoader) -> None:
    print("\n===== Test 2: compute_dataset_stats =====")
    stats = compute_dataset_stats(loader)
    expected_keys = ("action_mean", "action_std", "action_min", "action_max",
                     "proprio_mean", "proprio_std")
    for k in expected_keys:
        assert k in stats, f"缺少统计键: {k}"
        print(f"  {k}: shape={stats[k].shape}")

    assert stats["action_mean"].shape == (ACTION_DIM,), \
        f"action_mean 形状错误: {stats['action_mean'].shape}"
    assert stats["proprio_mean"].shape == (24,), \
        f"proprio_mean 形状错误: {stats['proprio_mean'].shape}"
    print("  [PASS]")


# ---------------------------------------------------------------------------
#  测试 3: RealRobotEnv 离线模式 reset / step
# ---------------------------------------------------------------------------

def test_env(root: Path) -> None:
    print("\n===== Test 3: RealRobotEnv (OFFLINE) reset / step =====")

    target_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)

    env = RealRobotEnv(
        task_language="pick up the red cube",
        target_pose=target_pose,
        mode=EnvMode.OFFLINE,
        dataset_path=str(root),
        dataset_fmt="lerobot",
        max_steps=EPISODE_LEN,
    )

    # --- reset ---
    obs, info = env.reset(seed=42)
    print(f"  reset() obs keys: {list(obs.keys())}")
    assert "image"    in obs, "obs 缺少 image 键"
    assert "proprio"  in obs, "obs 缺少 proprio 键"
    assert "language" in obs, "obs 缺少 language 键"
    assert obs["image"].shape   == (224, 224, 3),  f"image shape 错误: {obs['image'].shape}"
    assert obs["proprio"].shape == (24,),           f"proprio shape 错误: {obs['proprio'].shape}"
    assert obs["language"].shape== (512,),          f"language shape 错误: {obs['language'].shape}"
    print(f"  image shape: {obs['image'].shape}  proprio shape: {obs['proprio'].shape}")

    # --- step ---
    total_reward = 0.0
    step_count   = 0
    for _ in range(EPISODE_LEN):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count   += 1

        assert obs["image"].shape    == (224, 224, 3)
        assert obs["proprio"].shape  == (24,)
        assert "expert_action"       in info, "离线模式 info 缺少 expert_action"
        assert info["expert_action"].shape == (ACTION_DIM,)

        if terminated or truncated:
            break

    print(f"  完成 {step_count} 步，总奖励: {total_reward:.4f}")
    print(f"  最后 info: is_success={info['is_success']}  step={info['step']}")

    # --- render ---
    frame = env.render()
    assert frame is not None and frame.ndim == 3, "render() 返回格式错误"
    print(f"  render() 返回帧 shape: {frame.shape}")

    env.close()
    print("  [PASS]")


# ---------------------------------------------------------------------------
#  测试 4: episode_batch_iter
# ---------------------------------------------------------------------------

def test_batch_iter(root: Path) -> None:
    print("\n===== Test 4: episode_batch_iter =====")

    target_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    env = RealRobotEnv(
        task_language="pick up the red cube",
        target_pose=target_pose,
        mode=EnvMode.OFFLINE,
        dataset_path=str(root),
        dataset_fmt="lerobot",
    )

    BATCH_SIZE = 8
    batch_count = 0
    for batch in env.episode_batch_iter(batch_size=BATCH_SIZE, shuffle=True):
        expected_keys = ("obs_image", "obs_proprio", "actions",
                         "next_obs_image", "next_obs_proprio", "dones")
        for k in expected_keys:
            assert k in batch, f"batch 缺少键: {k}"

        B = batch["actions"].shape[0]
        assert B <= BATCH_SIZE
        assert batch["obs_image"].ndim   == 4,  "obs_image 应为 4D"
        assert batch["obs_proprio"].ndim == 2,  "obs_proprio 应为 2D"
        assert batch["actions"].shape[1] == ACTION_DIM
        assert batch["dones"].ndim       == 1

        batch_count += 1
        if batch_count == 1:
            print(f"  第1批: obs_image={batch['obs_image'].shape}  "
                  f"actions={batch['actions'].shape}  dones={batch['dones'].shape}")

    total_frames = N_EPISODES * EPISODE_LEN
    print(f"  共迭代 {batch_count} 批，覆盖 {total_frames} 个转换对")
    env.close()
    print("  [PASS]")


# ---------------------------------------------------------------------------
#  测试 5: bytes 格式图像解码
# ---------------------------------------------------------------------------

def test_bytes_image(tmp_dir: Path) -> None:
    print("\n===== Test 5: bytes 图像解码 =====")
    # 检查 Pillow 是否可用
    try:
        import PIL  # noqa: F401
    except ImportError:
        print("  [SKIP] PIL 未安装，跳过 bytes 图像测试")
        return

    bytes_root = tmp_dir / "lerobot_bytes"
    create_lerobot_sample_dataset(bytes_root, use_image_bytes=True)
    loader = LeRobotDatasetLoader(str(bytes_root))
    ep = loader.get_episode(0)
    print(f"  bytes 解码后 images shape: {ep.images.shape}  dtype={ep.images.dtype}")
    assert ep.images.ndim == 4 and ep.images.dtype == np.uint8
    print("  [PASS]")


# ---------------------------------------------------------------------------
#  Mock 接口：模拟真机硬件，用于 Online 模式测试
# ---------------------------------------------------------------------------

class MockRobotInterface:
    """模拟机器人接口，返回安全的固定观测，不触发安全保护。"""

    def __init__(self, tcp_pose=None):
        self._tcp_pose = np.array(
            tcp_pose if tcp_pose is not None else [0.5, 0.0, 0.3, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self._stopped = False

    def move_to_home(self):
        pass

    def send_eef_delta_command(self, delta, gripper=0.0, freq_hz=10.0):
        # delta 为 6 维末端增量 (env.py 传入 action[:6])，模拟微小位移
        self._tcp_pose = self._tcp_pose + np.asarray(delta, dtype=np.float32)[:6] * 0.01

    def get_latest_observation(self):
        return {
            "tcp_pose":     self._tcp_pose.copy(),
            # JOINT_DIM=12: 前6位为关节位置，后6位为关节速度（均为0，不触发速度越限）
            "joint_states": np.zeros(JOINT_DIM, dtype=np.float32),
            "force_torque": np.zeros(6, dtype=np.float32),   # 无外力
        }

    def stop(self):
        self._stopped = True

    def emergency_stop(self):
        self._stopped = True


class MockCameraInterface:
    """模拟相机接口，返回随机 uint8 RGB 图像。"""

    def __init__(self, h: int = IMG_H, w: int = IMG_W):
        self._h = h
        self._w = w
        self._rng = np.random.default_rng(seed=0)

    def get_image(self) -> np.ndarray:
        return self._rng.integers(0, 255, (self._h, self._w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
#  测试 6: RealRobotEnv Online 模式 reset / step
# ---------------------------------------------------------------------------

def test_online_env() -> None:
    print("\n===== Test 6: RealRobotEnv (ONLINE) reset / step =====")

    target_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    robot  = MockRobotInterface(tcp_pose=target_pose)
    camera = MockCameraInterface()

    env = RealRobotEnv(
        task_language="pick up the red cube",
        target_pose=target_pose,
        mode=EnvMode.ONLINE,
        robot_interface=robot,
        camera_interface=camera,
        max_steps=10,
    )

    # --- reset ---
    obs, info = env.reset(seed=0)
    print(f"  reset() obs keys: {list(obs.keys())}")
    assert "image"    in obs, "obs 缺少 image 键"
    assert "proprio"  in obs, "obs 缺少 proprio 键"
    assert "language" in obs, "obs 缺少 language 键"
    assert obs["image"].shape    == (IMG_H, IMG_W, 3),  f"image shape 错误: {obs['image'].shape}"
    assert obs["proprio"].shape  == (PROPRIO_DIM,),     f"proprio shape 错误: {obs['proprio'].shape}"
    assert obs["language"].shape == (LANG_DIM,),        f"language shape 错误: {obs['language'].shape}"
    print(f"  image shape: {obs['image'].shape}  proprio shape: {obs['proprio'].shape}")

    # --- step ---
    step_count  = 0
    total_reward = 0.0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count   += 1

        assert obs["image"].shape   == (IMG_H, IMG_W, 3)
        assert obs["proprio"].shape == (PROPRIO_DIM,)
        assert "is_success" in info
        assert "step"       in info

        if terminated or truncated:
            break

    print(f"  完成 {step_count} 步，总奖励: {total_reward:.4f}")
    print(f"  最后 info: is_success={info['is_success']}  step={info['step']}")

    # --- render ---
    frame = env.render()
    assert frame is not None and frame.ndim == 3, "render() 返回格式错误"
    print(f"  render() 返回帧 shape: {frame.shape}")

    env.close()
    assert robot._stopped, "env.close() 应调用 robot_interface.stop()"
    print("  [PASS]")


# ---------------------------------------------------------------------------
#  测试 7: RolloutCollector (Online 模式) 使用 Mock VLA 策略收集轨迹
# ---------------------------------------------------------------------------

def test_online_rollout_collector() -> None:
    print("\n===== Test 7: RolloutCollector (ONLINE) =====")

    target_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    robot  = MockRobotInterface(tcp_pose=target_pose)
    camera = MockCameraInterface()

    env = RealRobotEnv(
        task_language="pick up the red cube",
        target_pose=target_pose,
        mode=EnvMode.ONLINE,
        robot_interface=robot,
        camera_interface=camera,
        max_steps=15,
    )

    # Mock VLA 模型：随机采样动作，不依赖 GPU / 真实权重
    rng_policy = np.random.default_rng(seed=42)

    def _mock_predict(processed_obs, task_text):
        return rng_policy.uniform(-0.05, 0.05, (7,)).astype(np.float32)

    policy = GenericVLAWrapper(model=None, predict_fn=_mock_predict)
    collector = RolloutCollector(env, policy, task_text="pick up the red cube")

    # --- collect_episode ---
    episode = collector.collect_episode(max_steps=15)
    print(f"  episode length      : {episode['length']}")
    print(f"  total_reward        : {episode['total_reward']:.4f}")
    print(f"  is_success          : {episode['is_success']}")
    print(f"  actions.shape       : {episode['actions'].shape}")
    print(f"  rewards.shape       : {episode['rewards'].shape}")
    print(f"  dones.shape         : {episode['dones'].shape}")

    assert episode["length"] > 0,                      "episode 长度应 > 0"
    assert episode["actions"].ndim  == 2,              "actions 应为 2D (T,7)"
    assert episode["actions"].shape[1] == ACTION_DIM,  "actions 维度不符"
    assert episode["rewards"].ndim  == 1,              "rewards 应为 1D"
    assert episode["dones"].ndim    == 1,              "dones 应为 1D"
    assert len(episode["observations"]) == episode["length"]

    # --- collect_n_steps ---
    robot2  = MockRobotInterface(tcp_pose=target_pose)
    camera2 = MockCameraInterface()
    env2 = RealRobotEnv(
        task_language="pick up the red cube",
        target_pose=target_pose,
        mode=EnvMode.ONLINE,
        robot_interface=robot2,
        camera_interface=camera2,
        max_steps=15,
    )
    collector2 = RolloutCollector(env2, policy, task_text="pick up the red cube")
    N_STEPS = 12
    transitions = collector2.collect_n_steps(N_STEPS)
    print(f"  collect_n_steps({N_STEPS}) → {len(transitions)} 条转换对")
    assert len(transitions) == N_STEPS, f"转换对数量不符: {len(transitions)} != {N_STEPS}"
    for t in transitions:
        assert "obs"    in t
        assert "action" in t
        assert "reward" in t
        assert "done"   in t

    env2.close()
    print("  [PASS]")


# ---------------------------------------------------------------------------
#  主入口
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  LeRobot 样例数据 → RealRobotEnv 离线/在线测试")
    print("=" * 60)

    with tempfile.TemporaryDirectory(prefix="lerobot_test_") as tmp:
        tmp_dir = Path(tmp)
        root = tmp_dir / "lerobot_dataset"

        # 生成合成数据
        create_lerobot_sample_dataset(root, use_image_bytes=False)

        # 离线测试
        loader = test_loader(root)
        test_stats(loader)
        test_env(root)
        test_batch_iter(root)
        test_bytes_image(tmp_dir)

    # 在线测试（无需数据集文件）
    test_online_env()
    test_online_rollout_collector()

    print("\n" + "=" * 60)
    print("  全部测试通过 ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
