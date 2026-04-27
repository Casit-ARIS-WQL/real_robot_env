"""
vla_interface.py  ——  VLA 模型对接接口

功能模块
--------
* VLAObsProcessor   —— 将 env 观测预处理为 VLA 模型输入格式
* ActionNormalizer  —— 动作归一化 / 反归一化（VLA 输出 [-1,1] → 真机控制量）
* VLAModelWrapper   —— 抽象基类，屏蔽不同 VLA 模型的实现差异
* OpenVLAWrapper    —— OpenVLA (HuggingFace) 对接
* GenericVLAWrapper —— 通用包装器，适配自定义模型
* RolloutCollector  —— 在线 RL 滚动数据收集器

快速入门
--------
    from vla_interface import OpenVLAWrapper, RolloutCollector
    from env import RealRobotEnv, EnvMode

    policy = OpenVLAWrapper("openvla/openvla-7b", device="cuda")
    env    = RealRobotEnv(..., mode=EnvMode.ONLINE)
    collector = RolloutCollector(env, policy, task_text="pick up the cube")

    episode = collector.collect_episode()
    print(episode["total_reward"], episode["length"])
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  观测预处理
# ---------------------------------------------------------------------------

class VLAObsProcessor:
    """
    将 env.step/reset 返回的观测字典转换为 VLA 模型所需的输入格式。

    支持
    ----
    * 图像归一化（ImageNet 均值/方差）
    * 本体感觉归一化
    * 语言分词（HuggingFace Tokenizer / AutoProcessor）

    Parameters
    ----------
    image_mean     : (3,) RGB 均值，默认 ImageNet
    image_std      : (3,) RGB 标准差，默认 ImageNet
    proprio_mean   : (D,) 本体感觉均值（None=不归一化）
    proprio_std    : (D,) 本体感觉标准差（None=不归一化）
    tokenizer      : HuggingFace Tokenizer（与 processor 二选一）
    processor      : HuggingFace AutoProcessor（优先级高于 tokenizer）
    image_format   : "CHW" 或 "HWC"（模型期望的图像维度顺序）
    """

    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        image_mean:   Optional[np.ndarray] = None,
        image_std:    Optional[np.ndarray] = None,
        proprio_mean: Optional[np.ndarray] = None,
        proprio_std:  Optional[np.ndarray] = None,
        tokenizer=None,
        processor=None,
        image_format: str = "CHW",
    ):
        self.image_mean   = image_mean   if image_mean   is not None else self._IMAGENET_MEAN
        self.image_std    = image_std    if image_std    is not None else self._IMAGENET_STD
        self.proprio_mean = proprio_mean
        self.proprio_std  = proprio_std
        self.tokenizer    = tokenizer
        self.processor    = processor
        self.image_format = image_format.upper()

    # ---- 子组件 ----

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """uint8 (H,W,3) -> float32，归一化后转换维度顺序。

        Returns
        -------
        (3,H,W) if image_format=="CHW" else (H,W,3)
        """
        img = image.astype(np.float32) / 255.0
        img = (img - self.image_mean) / self.image_std       # HWC, broadcast on last dim
        if self.image_format == "CHW":
            img = img.transpose(2, 0, 1)
        return img

    def process_proprio(self, proprio: np.ndarray) -> np.ndarray:
        """本体感觉归一化（如未提供统计量则直接返回）。"""
        if self.proprio_mean is not None and self.proprio_std is not None:
            return ((proprio - self.proprio_mean) / (self.proprio_std + 1e-8)).astype(np.float32)
        return proprio.astype(np.float32)

    def tokenize_language(self, text: str) -> Dict[str, Any]:
        """将语言描述转换为 token ids（依赖已传入的 tokenizer/processor）。"""
        if self.processor is not None:
            return self.processor(text=text, return_tensors="pt",
                                  padding=True, truncation=True, max_length=128)
        if self.tokenizer is not None:
            return self.tokenizer(text, return_tensors="pt",
                                  padding=True, truncation=True, max_length=128)
        return {"text": text}

    # ---- 完整流程 ----

    def __call__(self, obs: Dict, task_text: str = "") -> Dict:
        """
        完整预处理。

        Parameters
        ----------
        obs       : env.step/reset 返回的观测字典
                    {"image": uint8(H,W,3), "proprio": (D,), "language": (512,)}
        task_text : 任务自然语言（用于实时分词的模型）

        Returns
        -------
        processed : dict
            pixel_values  : (3,H,W) float32  （CHW 格式时）或 (H,W,3)
            proprio       : (D,)    float32
            lang_embed    : (512,)  float32   （预计算嵌入向量）
            lang_tokens   : dict              （task_text 非空时额外输出）
        """
        processed: Dict[str, Any] = {
            "pixel_values": self.process_image(obs["image"]),
            "proprio":      self.process_proprio(obs["proprio"]),
            "lang_embed":   obs["language"].astype(np.float32),
        }
        if task_text:
            processed["lang_tokens"] = self.tokenize_language(task_text)
        return processed


# ---------------------------------------------------------------------------
#  动作归一化
# ---------------------------------------------------------------------------

class ActionNormalizer:
    """
    双向动作归一化工具。

    VLA 模型通常输出 [-1, 1] 归一化动作，需要映射回真机控制范围。

    Parameters
    ----------
    action_low  : (7,) 真机动作下界
    action_high : (7,) 真机动作上界
    """

    def __init__(self, action_low: np.ndarray, action_high: np.ndarray):
        self.low   = np.array(action_low,  dtype=np.float32)
        self.high  = np.array(action_high, dtype=np.float32)
        self.scale = (self.high - self.low) / 2.0
        self.bias  = (self.high + self.low) / 2.0

    def normalize(self, action: np.ndarray) -> np.ndarray:
        """真机动作 → [-1, 1]"""
        return ((action - self.bias) / (self.scale + 1e-8)).astype(np.float32)

    def denormalize(self, action: np.ndarray) -> np.ndarray:
        """[-1, 1] → 真机动作"""
        return (action * self.scale + self.bias).astype(np.float32)

    @classmethod
    def from_dataset_stats(
        cls,
        dataset_actions: np.ndarray,
        percentile: float = 1.0,
    ) -> "ActionNormalizer":
        """从数据集动作样本自动估计范围（去除尾部异常值）。"""
        low  = np.percentile(dataset_actions, percentile,       axis=0)
        high = np.percentile(dataset_actions, 100 - percentile, axis=0)
        return cls(low, high)


# ---------------------------------------------------------------------------
#  VLA 模型抽象接口
# ---------------------------------------------------------------------------

class VLAModelWrapper(ABC):
    """
    VLA 模型的统一抽象接口。

    屏蔽不同 VLA 模型（OpenVLA / RT-2 / π0 / Octo 等）的实现差异，
    使 RL 训练循环只需调用 predict_action()。

    Parameters
    ----------
    obs_processor     : VLAObsProcessor 实例（None=使用默认配置）
    action_normalizer : ActionNormalizer 实例（None=不做反归一化）
    """

    def __init__(
        self,
        obs_processor:     Optional[VLAObsProcessor]  = None,
        action_normalizer: Optional[ActionNormalizer]  = None,
    ):
        self.obs_processor     = obs_processor or VLAObsProcessor()
        self.action_normalizer = action_normalizer

    @abstractmethod
    def _model_predict(self, processed_obs: Dict, task_text: str) -> np.ndarray:
        """
        模型推理核心（子类实现）。

        Parameters
        ----------
        processed_obs : VLAObsProcessor 输出的预处理观测
        task_text     : 原始任务语言描述

        Returns
        -------
        norm_action : (7,) float32，在 [-1, 1] 的归一化动作
        """

    def predict_action(self, obs: Dict, task_text: str = "") -> np.ndarray:
        """
        端到端推理：原始 env 观测 → 真机可执行动作。

        Parameters
        ----------
        obs       : env.step/reset 返回的观测字典
        task_text : 任务语言描述（自然语言）

        Returns
        -------
        action : (7,) float32，已反归一化至真机控制范围
        """
        processed   = self.obs_processor(obs, task_text)
        norm_action = self._model_predict(processed, task_text)
        if self.action_normalizer is not None:
            return self.action_normalizer.denormalize(norm_action)
        return norm_action.astype(np.float32)

    def batch_predict(self, obs_list: List[Dict], task_text: str = "") -> np.ndarray:
        """批量推理，用于离线策略评估。

        Returns
        -------
        actions : (N, 7) float32
        """
        return np.stack(
            [self.predict_action(obs, task_text) for obs in obs_list], axis=0
        )


# ---------------------------------------------------------------------------
#  OpenVLA 对接
# ---------------------------------------------------------------------------

class OpenVLAWrapper(VLAModelWrapper):
    """
    OpenVLA 模型对接（https://github.com/openvla/openvla）。

    依赖
    ----
        pip install transformers accelerate torch pillow

    Parameters
    ----------
    model_path    : HuggingFace Hub 路径或本地目录，如 "openvla/openvla-7b"
    device        : "cuda" / "cpu" / "cuda:0"
    torch_dtype   : torch.dtype（默认 bfloat16 on CUDA，float32 on CPU）
    unnorm_key    : OpenVLA 动作反归一化键名（None=模型内部决定）
    obs_processor : 自定义 VLAObsProcessor（None=使用默认 ImageNet 均值方差）
    """

    def __init__(
        self,
        model_path:        str,
        device:            str = "cuda",
        torch_dtype=None,
        unnorm_key:        Optional[str] = None,
        obs_processor:     Optional[VLAObsProcessor]  = None,
        action_normalizer: Optional[ActionNormalizer]  = None,
    ):
        super().__init__(obs_processor, action_normalizer)
        self._unnorm_key = unnorm_key
        self._device     = device
        self._load_model(model_path, torch_dtype)

    def _load_model(self, model_path: str, torch_dtype):
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            raise ImportError("OpenVLA 需要: pip install transformers torch accelerate")

        dtype = torch_dtype
        if dtype is None:
            dtype = torch.bfloat16 if (
                torch.cuda.is_available() and "cuda" in self._device
            ) else torch.float32

        self._hf_processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()
        logger.info("OpenVLA 模型加载成功: %s", model_path)

    def _model_predict(self, processed_obs: Dict, task_text: str) -> np.ndarray:
        import torch
        from PIL import Image as PILImage

        # OpenVLA 期望 PIL Image：将归一化图像反变换回 uint8
        img_chw = processed_obs["pixel_values"]   # (3,H,W) float32
        mean = self.obs_processor.image_mean.reshape(3, 1, 1)
        std  = self.obs_processor.image_std.reshape(3, 1, 1)
        img_hwc = np.clip((img_chw * std + mean) * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        pil_image = PILImage.fromarray(img_hwc)

        inputs = self._hf_processor(
            text=task_text,
            images=pil_image,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            action = self._model.predict_action(
                **inputs, unnorm_key=self._unnorm_key
            )

        return np.array(action.cpu().float()).flatten()[:7].astype(np.float32)


# ---------------------------------------------------------------------------
#  通用包装器
# ---------------------------------------------------------------------------

class GenericVLAWrapper(VLAModelWrapper):
    """
    通用 VLA 包装器，适配任意实现了 predict(obs, text) 接口的模型。

    Parameters
    ----------
    model          : 任意模型对象
    predict_fn     : 推理函数，签名 (processed_obs: dict, task_text: str) -> array-like (7,)
                     若为 None，则在 model 上查找 predict_fn_name 方法。
    predict_fn_name: 当 predict_fn=None 时，从 model 查找的方法名
    """

    def __init__(
        self,
        model,
        predict_fn=None,
        predict_fn_name:   str = "predict_action",
        obs_processor:     Optional[VLAObsProcessor]  = None,
        action_normalizer: Optional[ActionNormalizer]  = None,
    ):
        super().__init__(obs_processor, action_normalizer)
        self._model      = model
        self._predict_fn = predict_fn or getattr(model, predict_fn_name)

    def _model_predict(self, processed_obs: Dict, task_text: str) -> np.ndarray:
        result = self._predict_fn(processed_obs, task_text)
        if hasattr(result, "cpu"):       # torch.Tensor
            result = result.cpu().numpy()
        elif hasattr(result, "numpy"):   # tf.Tensor / jax.Array
            result = result.numpy()
        return np.array(result, dtype=np.float32).flatten()[:7]


# ---------------------------------------------------------------------------
#  在线 RL 数据收集器
# ---------------------------------------------------------------------------

class RolloutCollector:
    """
    将 VLA 策略与 RealRobotEnv 配合，收集 (s, a, r, s', done) 转换对。

    用途
    ----
    * 在线强化学习（PPO / SAC / TD3 等）
    * VLA 策略的在线评估与调优

    Parameters
    ----------
    env       : RealRobotEnv 实例
    policy    : VLAModelWrapper 实例
    task_text : 任务语言描述（自然语言），覆盖 env 中的 task_language
    """

    def __init__(self, env, policy: VLAModelWrapper, task_text: str = ""):
        self.env       = env
        self.policy    = policy
        self.task_text = task_text

    def collect_episode(self, max_steps: Optional[int] = None) -> Dict:
        """
        运行一个完整 episode，返回轨迹数据。

        Returns
        -------
        episode : dict
            observations      : List[Dict]         长度 T
            actions           : np.ndarray (T, 7)
            rewards           : np.ndarray (T,)
            next_observations : List[Dict]         长度 T
            dones             : np.ndarray (T,)
            infos             : List[Dict]         长度 T
            total_reward      : float
            length            : int
            is_success        : bool
        """
        max_steps = max_steps or getattr(self.env, "max_steps", 500)
        obs, _ = self.env.reset()

        observations: List[Dict] = []
        actions:      List[np.ndarray] = []
        rewards:      List[float] = []
        next_obs_list: List[Dict] = []
        dones:        List[float] = []
        infos:        List[Dict] = []
        total_reward  = 0.0
        is_success    = False

        for _ in range(max_steps):
            action = self.policy.predict_action(obs, self.task_text)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            next_obs_list.append(next_obs)
            dones.append(float(terminated or truncated))
            infos.append(info)

            total_reward += reward
            if info.get("is_success", False):
                is_success = True

            obs = next_obs
            if terminated or truncated:
                break

        return {
            "observations":      observations,
            "actions":           np.array(actions,  dtype=np.float32),
            "rewards":           np.array(rewards,  dtype=np.float32),
            "next_observations": next_obs_list,
            "dones":             np.array(dones,    dtype=np.float32),
            "infos":             infos,
            "total_reward":      total_reward,
            "length":            len(actions),
            "is_success":        is_success,
        }

    def collect_n_steps(self, n_steps: int) -> List[Dict]:
        """
        收集若干步转换对（跨 episode），用于 on-policy 算法缓冲区填充。

        Returns
        -------
        transitions : List[dict]，每个 dict 包含
            obs, action, reward, next_obs, done, info
        """
        transitions: List[Dict] = []
        obs, _ = self.env.reset()

        for _ in range(n_steps):
            action = self.policy.predict_action(obs, self.task_text)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            transitions.append({
                "obs":      obs,
                "action":   action,
                "reward":   float(reward),
                "next_obs": next_obs,
                "done":     float(terminated or truncated),
                "info":     info,
            })
            if terminated or truncated:
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        return transitions

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        评估策略性能，返回平均奖励、成功率等指标。

        Returns
        -------
        metrics : dict
            mean_reward    : float
            std_reward     : float
            success_rate   : float  (0.0 ~ 1.0)
            mean_length    : float
        """
        rewards  = []
        successes = []
        lengths  = []
        for _ in range(n_episodes):
            ep = self.collect_episode()
            rewards.append(ep["total_reward"])
            successes.append(float(ep["is_success"]))
            lengths.append(ep["length"])

        return {
            "mean_reward":  float(np.mean(rewards)),
            "std_reward":   float(np.std(rewards)),
            "success_rate": float(np.mean(successes)),
            "mean_length":  float(np.mean(lengths)),
        }
