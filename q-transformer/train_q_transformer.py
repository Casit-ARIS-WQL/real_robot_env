"""
train_q_transformer.py  ——  Q-Transformer Q 值函数独立训练脚本

核心流程
--------
1. 通过 ``json_dataloader.load_qtransformer_dataset`` 加载 episode JSON 数据，
   自动完成动作离散化，返回兼容 PyTorch DataLoader 的 ``QTransformerEpisodeDataset``。
2. 构建 MaxViT 视觉骨干 + ``QRoboticTransformer`` 模型（支持单/多动作、
   dueling head、dual critics、**多视角输入**等配置）。
3. 用 ``QLearner`` 封装训练循环，内置 EMA 目标网络、梯度裁剪、
   conservative regularization loss 以及定期 checkpoint 保存。

快速运行（使用仓库内置的示例数据）
------------------------------------
    cd q-transformer
    python train_q_transformer.py                         # 默认配置（单视角 right）
    python train_q_transformer.py --help                  # 查看全部参数

常用参数示例
------------
    # 使用目录中所有 .json 文件，多步 Q-learning，保存到自定义目录
    python train_q_transformer.py \\
        --data_path data/ \\
        --num_timesteps 4 \\
        --num_actions 14 \\
        --num_train_steps 20000 \\
        --checkpoint_folder ./runs/exp1

    # 只使用单臂（前 7 维动作），关闭文本条件化
    python train_q_transformer.py \\
        --num_actions 7 \\
        --condition_on_text false

    # 使用多视角输入（right + top + left 三个相机）
    python train_q_transformer.py \\
        --image_keys right,top,left \\
        --num_actions 14 \\
        --num_train_steps 20000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
#  日志配置
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_q_transformer")


# ---------------------------------------------------------------------------
#  参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Q-Transformer Q 值函数训练脚本（配合 json_dataloader.py 使用）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- 数据 ---
    data_group = parser.add_argument_group("数据")
    data_group.add_argument(
        "--data_path",
        type=str,
        default="data",
        help=(
            "数据集根目录或单个 episode JSON 文件路径。\n"
            "推荐目录结构（二级子目录）：data/1/episode_with_rewards.json + data/1/right/ + data/1/top/ + data/1/left/，\n"
            "data/2/episode_with_rewards.json + …（每个子目录代表一条完整轨迹）。\n"
            "也兼容旧版扁平结构：目录下直接存放多个 .json 文件。"
        ),
    )
    data_group.add_argument(
        "--image_keys",
        type=str,
        default="top,left,right",
        help=(
            "使用的相机视角，单视角时填写一个名称（如 'right'），"
            "多视角时填写逗号分隔的名称列表（如 'right,top,left'）。"
            "可选值: right | top | left | top_2"
        ),
    )
    data_group.add_argument(
        "--num_frames",
        type=int,
        default=6,
        help="每个状态堆叠的视频帧数 F，对应模型输入中的帧维度",
    )
    data_group.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="图像边长（正方形），模型输入分辨率",
    )
    data_group.add_argument(
        "--num_timesteps",
        type=int,
        default=1,
        help="N-step Q-learning 的步数；1 = 单步，>1 = N 步",
    )

    # --- 动作 ---
    action_group = parser.add_argument_group("动作")
    action_group.add_argument(
        "--num_actions",
        type=int,
        default=14,
        help="动作维度数（双臂 14 维；单臂 7 维）",
    )
    action_group.add_argument(
        "--action_bins",
        type=int,
        default=256,
        help="每个动作维度的离散区间数",
    )

    # --- 模型 ---
    model_group = parser.add_argument_group("模型")
    model_group.add_argument(
        "--vit_dim",
        type=int,
        default=64,
        help="MaxViT 基础通道维度",
    )
    model_group.add_argument(
        "--vit_depth",
        type=str,
        default="2,2,5,2",
        help="MaxViT 各阶段层数，逗号分隔，如 '2,2,5,2'",
    )
    model_group.add_argument(
        "--transformer_depth",
        type=int,
        default=6,
        help="Q-Transformer 主干 Transformer 层数",
    )
    model_group.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Transformer 注意力头数",
    )
    model_group.add_argument(
        "--dim_head",
        type=int,
        default=64,
        help="每个注意力头的维度",
    )
    model_group.add_argument(
        "--dueling",
        type=lambda x: x.lower() == "true",
        default=True,
        metavar="true/false",
        help="是否使用 Dueling DQN head",
    )
    model_group.add_argument(
        "--dual_critics",
        type=lambda x: x.lower() == "true",
        default=False,
        metavar="true/false",
        help="是否使用双 critic 以减小 Q 值高估偏差",
    )
    model_group.add_argument(
        "--condition_on_text",
        type=lambda x: x.lower() == "true",
        default=True,
        metavar="true/false",
        help="是否对任务语言指令做文本条件化",
    )
    model_group.add_argument(
        "--cond_drop_prob",
        type=float,
        default=0.2,
        help="文本条件 Classifier-Free Guidance 的 dropout 概率",
    )
    model_group.add_argument(
        "--num_residual_streams",
        type=int,
        default=4,
        help="Hyper-Connection 残差流数（1 = 标准残差，>1 = 多流）",
    )
    model_group.add_argument(
        "--text_model_path",
        type=str,
        default=None,
        help=(
            "本地 T5 文本编码器模型目录。"
            "若为 None（默认），则在首次运行时从 HuggingFace Hub 自动下载 google/t5-v1_1-base。"
            "在无网络环境中，请先运行 download_t5_model.py 完成下载，再通过此参数指定本地路径。"
        ),
    )

    # --- 训练 ---
    train_group = parser.add_argument_group("训练")
    train_group.add_argument(
        "--num_train_steps",
        type=int,
        default=10000,
        help="总训练步数",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="每步 batch 大小",
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="优化器学习率",
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="AdamAtan2 weight decay",
    )
    train_group.add_argument(
        "--grad_accum_every",
        type=int,
        default=1,
        help="梯度累积步数（模拟更大 batch）",
    )
    train_group.add_argument(
        "--discount_factor_gamma",
        type=float,
        default=0.98,
        help="Q-learning 折扣因子 γ",
    )
    train_group.add_argument(
        "--reward_scale",
        type=float,
        default=1.0,
        help=(
            "奖励缩放因子，将原始奖励除以该值以使 Q 目标保持在 [0,1]。"
            "对于子 episode 稀疏奖励（1~6 分），建议设为 21.0（最大累计奖励之和）"
        ),
    )
    train_group.add_argument(
        "--min_reward",
        type=float,
        default=0.0,
        help="conservative regularization loss 中的最小奖励值",
    )
    train_group.add_argument(
        "--conservative_reg_loss_weight",
        type=float,
        default=1.0,
        help="conservative regularization loss 权重；设为 0 可禁用",
    )
    train_group.add_argument(
        "--use_bce_loss",
        type=lambda x: x.lower() == "true",
        default=False,
        metavar="true/false",
        help="是否使用 HL-Gauss 分类 loss 代替 MSE regression loss",
    )
    train_group.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="梯度裁剪范数上限",
    )
    train_group.add_argument(
        "--checkpoint_folder",
        type=str,
        default="./checkpoints",
        help="checkpoint 保存目录",
    )
    train_group.add_argument(
        "--checkpoint_every",
        type=int,
        default=1000,
        help="每隔多少步保存一次 checkpoint",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
#  主函数
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 0. 确认依赖
    # ------------------------------------------------------------------
    try:
        import torch
    except ImportError:
        logger.error("未找到 PyTorch，请执行: pip install torch")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. 数据加载
    #    通过 json_dataloader.load_qtransformer_dataset 完成：
    #    - JSON 解析 → EpisodeDataWithRewards
    #    - 动作范围统计 → 线性离散化
    #    - 封装为 QTransformerEpisodeDataset（兼容 PyTorch DataLoader）
    # ------------------------------------------------------------------
    logger.info("=== 步骤 1/3：加载数据集 ===")
    logger.info("数据路径: %s", args.data_path)

    # 将当前工作目录添加到 sys.path，确保能找到 json_dataloader.py
    _repo_dir = Path(__file__).parent
    if str(_repo_dir) not in sys.path:
        sys.path.insert(0, str(_repo_dir))

    # json_dataloader.py 依赖 dataset_utils（位于 robot_env/），也加入 sys.path
    _robot_env_dir = _repo_dir.parent / "robot_env"
    if _robot_env_dir.is_dir() and str(_robot_env_dir) not in sys.path:
        sys.path.insert(0, str(_robot_env_dir))

    from json_dataloader import load_qtransformer_dataset  # noqa: E402

    # 解析多视角参数：逗号分隔 → list；单个视角也转为 list
    image_keys = [k.strip() for k in args.image_keys.split(",") if k.strip()]
    num_cameras = len(image_keys)

    dataset = load_qtransformer_dataset(
        path=args.data_path,
        image_keys=image_keys,
        num_frames=args.num_frames,
        image_size=args.image_size,
        action_bins=args.action_bins,
        num_actions=args.num_actions,
        num_timesteps=args.num_timesteps,
    )

    logger.info(
        "数据集就绪：%d 条样本  num_actions=%d  action_bins=%d  "
        "num_frames=%d  image_size=%d  num_timesteps=%d  image_keys=%s",
        len(dataset),
        args.num_actions,
        args.action_bins,
        args.num_frames,
        args.image_size,
        args.num_timesteps,
        image_keys,
    )

    if len(dataset) == 0:
        logger.error("数据集为空，请检查 --data_path 是否正确")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. 构建模型
    #    QRoboticTransformer = MaxViT（视觉骨干）
    #                        + TokenLearner
    #                        + Transformer（主干）
    #                        + QHead（单/多动作）
    # ------------------------------------------------------------------
    logger.info("=== 步骤 2/3：构建模型 ===")

    from q_transformer import QRoboticTransformer  # noqa: E402

    vit_depth = tuple(int(d) for d in args.vit_depth.split(","))

    # MaxViT 需要输入分辨率能被 window_size（7）和下采样因子整除
    # 下采样因子 = conv_stem(×2) × stages(×2^len(depth)) = 2 × 2^4 = 32
    # 默认 image_size=224 满足 224/32=7，224/7=32，均为整数
    vit_config = dict(
        num_classes=1000,
        dim_conv_stem=args.vit_dim,
        dim=args.vit_dim,
        dim_head=args.dim_head,
        depth=vit_depth,
        window_size=7,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
    )

    model = QRoboticTransformer(
        vit=vit_config,
        num_actions=args.num_actions,
        action_bins=args.action_bins,
        depth=args.transformer_depth,
        heads=args.heads,
        dim_head=args.dim_head,
        cond_drop_prob=args.cond_drop_prob,
        dueling=args.dueling,
        condition_on_text=args.condition_on_text,
        num_residual_streams=args.num_residual_streams,
        dual_critics=args.dual_critics,
        weight_tie_action_bin_embed=True,
        num_cameras=num_cameras,
        text_model_path=args.text_model_path,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "模型参数量: %.2f M  (num_actions=%d  action_bins=%d  "
        "condition_on_text=%s  dueling=%s  dual_critics=%s  num_cameras=%d)",
        num_params / 1e6,
        args.num_actions,
        args.action_bins,
        args.condition_on_text,
        args.dueling,
        args.dual_critics,
        num_cameras,
    )

    # ------------------------------------------------------------------
    # 3. 训练
    #    QLearner 内置：
    #    - EMA 目标网络（q_target_ema_kwargs 可调）
    #    - AdamAtan2 优化器
    #    - Accelerate 分布式/混合精度支持
    #    - Conservative regularization loss（离线 Q-learning 关键组件）
    #    - N-step / 单步 / 多动作自回归 Q-learning 自动分支
    # ------------------------------------------------------------------
    logger.info("=== 步骤 3/3：开始训练 ===")
    logger.info(
        "总步数=%d  batch_size=%d  lr=%.1e  γ=%.3f  "
        "reward_scale=%.1f  num_timesteps=%d",
        args.num_train_steps,
        args.batch_size,
        args.learning_rate,
        args.discount_factor_gamma,
        args.reward_scale,
        args.num_timesteps,
    )

    from q_transformer import QLearner  # noqa: E402

    learner = QLearner(
        model,
        dataset=dataset,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_accum_every=args.grad_accum_every,
        discount_factor_gamma=args.discount_factor_gamma,
        reward_scale=args.reward_scale,
        min_reward=args.min_reward,
        conservative_reg_loss_weight=args.conservative_reg_loss_weight,
        use_bce_loss=args.use_bce_loss,
        max_grad_norm=args.max_grad_norm,
        n_step_q_learning=(args.num_timesteps > 1),
        checkpoint_folder=args.checkpoint_folder,
        checkpoint_every=args.checkpoint_every,
        dataloader_kwargs=dict(
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
        ),
    )

    # 启动训练循环（内部会自动保存 checkpoint）
    learner()


# ---------------------------------------------------------------------------
#  入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()