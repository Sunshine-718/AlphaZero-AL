# 训练优化进展：O Policy 震荡分析与对策

> 日期：2026-02-23
> 前置文档：`fpu-training-diagnosis.md`、`o-policy-convergence-diagnosis.md`
> 阶段：FPU 越界 bug 修复后，针对 O policy 持续震荡的优化

---

## 目录

1. [问题现状](#1-问题现状)
2. [O Policy 震荡的定量分析](#2-o-policy-震荡的定量分析)
3. [已实现的训练优化](#3-已实现的训练优化)
4. [与开源实现的对比](#4-与开源实现的对比)
5. [Hinge Loss 可行性分析](#5-hinge-loss-可行性分析)
6. [当前超参配置汇总](#6-当前超参配置汇总)
7. [待验证实验](#7-待验证实验)

---

## 1. 问题现状

### 与前序诊断报告的关系

| 文档 | 阶段 | 发现 |
|------|------|------|
| `fpu-training-diagnosis.md` | FPU bug 未修复 | FPU 越界导致 O 搜索饿死，col3 被钉在 ~47% |
| `o-policy-convergence-diagnosis.md` | FPU bug 未修复 | NN value head 评估方向反转，MCTS 越搜越偏 |
| **本文** | **FPU bug 已修复** | O policy **不再停滞**，但在 col2/col3/col4 之间**持续震荡** |

### 当前观察

FPU 下界截断（`fpu_value = std::max(-1.0f, fpu_value)`）修复后：

- X policy：已稳定收敛到 col3（~78-88%）
- O policy：col3 从 ~47% 解锁，但出现新现象——**三路震荡**
  - col2 ≈ 29%, col3 ≈ 33%, col4 ≈ 30%，三者此消彼长
  - 不是收敛后的微小波动，而是 **大幅来回摆动**

---

## 2. O Policy 震荡的定量分析

### 2.1 Policy Chasing（策略追逐）现象

震荡的根因是 NN 与 MCTS 之间的反馈延迟形成的追逐循环：

```
Step 1: NN 学到 "col3 好" → prior(col3) 高
    ↓
Step 2: MCTS 集中访问 col3 → 自对弈数据以 col3 路线为主
    ↓
Step 3: 对手（X）也开始针对 col3 应对 → col3 路线的 outcome 变差
    ↓
Step 4: Value head 发现 "col3 似乎不好了" → 切换到 col2/col4
    ↓
Step 5: col2/col4 路线数据增多 → 但对手未针对 → 效果一般
    ↓
Step 6: 回到 col3（因为此时对手不再专门针对 col3）
    ↓
回到 Step 1...（循环往复）
```

### 2.2 震荡频率估计

从训练曲线观察：

- **震荡周期** ≈ **100 training steps**（一个完整 col3↑ → col3↓ → col3↑ 循环）
- 每个 training step 消耗约 100 × 平均棋局长度 ≈ 100 × 20 = **2000 个样本**

### 2.3 Buffer 作为低通滤波器

Replay buffer 对训练信号起到**移动平均**（低通滤波）的作用。Buffer 越大，覆盖的震荡周期越多，平均后的信号越稳定。

**关键参数：**

| 参数 | 值 | 含义 |
|------|-----|------|
| buffer 容量 | 100,000 样本（旧）/ 500,000（新） | 可容纳的历史数据量 |
| 每步新数据 | ~2,000 样本 | 一个 training step 产生的数据 |
| buffer 刷新时间 | 100,000 / 2,000 = **50 steps**（旧）| buffer 完全更新一遍的时间 |
| 震荡周期 | ~100 steps | O policy 完成一次完整震荡的时间 |
| 覆盖周期数 | 50 / 100 = **0.5 个周期**（旧）| 当前 buffer 能覆盖多少个震荡周期 |

**分析：**

- 0.5 个周期意味着 buffer 在任何时刻只包含**半个震荡相位**的数据
- 这完全不够——buffer 内的数据要么偏向 col3，要么偏向 col2/col4，无法同时覆盖两侧
- 需要至少覆盖 **2 个完整周期**（~200 steps 的数据量），才能让移动平均有效平滑信号

**结论：需要 500k-750k 的 buffer 容量**

```
需要: 2 个周期 × 100 steps/周期 × 2000 样本/step = 400,000 样本
安全裕量 (×1.5): 600,000 样本
推荐: 500,000 ~ 750,000
```

当前已将默认 buffer 调整为 **500,000**（`server.py --buf 500000`）。

### 2.4 为什么不用开局随机化

前序诊断报告推荐了开局随机化（Opening Diversification）。但经分析：

- 开局随机化会引入更多**不确定性**——随机开局产生的样本质量参差不齐
- 大 buffer + 现有 Dirichlet noise 已经提供了足够的探索
- 更倾向于通过 **buffer 大小 + 采样策略 + loss 加权** 的组合来解决

---

## 3. 已实现的训练优化

### 3.1 Noise Epsilon 衰减

**目的**：开局阶段用更高的 Dirichlet noise 促进探索（帮助发现 O-col3），随棋局进行逐步降低 noise 以提高中后盘策略精度。

**衰减公式**：

```
effective_eps = noise_eps_min + (noise_eps_init - noise_eps_min) × max(0, 1 - step / noise_steps)
```

- `noise_eps_init`：初始 noise epsilon（默认 0.25）
- `noise_eps_min`：衰减下限（默认 0.1，通过 `--noise_eps_min` 设置）
- `noise_steps`：衰减步数（默认 0 = 不衰减，通过 `--noise_steps` 设置）

**示例**：`--noise_eps 0.5 --noise_steps 12 --noise_eps_min 0.1`

| 棋步 | effective_eps |
|------|--------------|
| 0 | 0.50 |
| 3 | 0.40 |
| 6 | 0.30 |
| 9 | 0.20 |
| 12+ | 0.10 |

**关键实现细节**：

- C++ `MCTS.noise_epsilon` 是 public 成员，被 `get_ucb()` **动态读取**（不缓存在节点上）
- `BatchedMCTS` 中所有 active game 在同一 step 推进 → 单个全局 eps 即可
- `noise_steps=0` 时完全向后兼容

**修改文件（7 个）**：

| 文件 | 改动 |
|------|------|
| `src/cpp/BatchedMCTS.h` | 新增 `set_noise_epsilon(float eps)` 方法 |
| `src/cpp/bindings.cpp` | 暴露 `set_noise_epsilon` 给 Python |
| `src/MCTS_cpp.py` | Python wrapper 转发方法 |
| `src/player.py` | `BatchedAlphaZeroPlayer` 存储衰减参数 |
| `src/game.py` | `batch_self_play` 中计算衰减并设置 eps |
| `client.py` | CLI 参数 `--noise_steps`, `--noise_eps_min` |
| `server.py` | CLI 参数 + config dict 传递 |

### 3.2 Class Weight（类别加权 Value Loss）

**目的**：补偿 P2 win 样本的不足。

**问题背景**：

Connect4 是 P1 必胜游戏，训练数据中 winner 类别天然不均衡：

| 训练阶段 | P1 Win | Draw | P2 Win |
|----------|--------|------|--------|
| 早期（随机） | ~55% | ~10% | ~35% |
| 中期（P1 变强） | ~70% | ~5% | ~25% |
| 后期（P1 很强） | ~85% | ~3% | ~12% |

P2 win 样本仅占 12-18%，导致 value head（3-class NLL loss）对 P2 win 局面的梯度贡献被淹没。

**实现**：

在 `NetworkBase.py` 的 `train_step` 中，按 mini-batch 内各类频率的倒数加权：

```python
if balance_class_weight:
    counts = torch.bincount(value, minlength=3).float().clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * 3.0  # 归一化使总权重不变
    v_loss = F.nll_loss(value_pred, value, weight=weights.to(value_pred.device))
else:
    v_loss = F.nll_loss(value_pred, value)
```

**效果**：P2 win 样本自动获得更大梯度权重，迫使网络认真区分 col2/col3/col4 在 P2 视角下的价值差异。

### 3.3 分层采样（Stratified Sampling）

**目的**：在 replay buffer 采样时，确保 P1 win / Draw / P2 win 三类等量出现。

**实现**：

在 `ReplayBuffer.py` 中新增 `_stratified_sample` 方法：

```python
def _stratified_sample(self, max_samples):
    """按 winner 类别（P1 win=1, Draw=0, P2 win=-1）分层采样，每类等量。"""
    n = len(self)
    winners = self.winner[:n].view(-1)
    per_class = max_samples // 3

    idx_list = []
    for w in [1, 0, -1]:
        class_indices = torch.where(winners == w)[0]
        if len(class_indices) == 0:
            continue
        # 有放回采样：少数类自动过采样
        rand_idx = torch.from_numpy(
            np.random.randint(0, len(class_indices), per_class, dtype=np.int64))
        idx_list.append(class_indices[rand_idx])

    idx = torch.cat(idx_list)
    idx = idx[torch.randperm(len(idx))]  # 打乱顺序
    return idx
```

**关键设计**：
- **有放回采样**：当某类样本不足 `per_class` 时，同一样本会被重复抽取。这比丢弃多数类样本更好——不浪费信息
- 与 class weight 通过 **同一开关** `--balance_sampling` 控制

### 3.4 超参透传

以下超参已全部可通过 `server.py` CLI 设置，经 config dict → `setattr` 自动传递到 pipeline 及下游组件：

| 参数 | CLI flag | 默认值 | 传递路径 |
|------|----------|--------|----------|
| Replay ratio | `--replay_ratio` | 0.1 | server → config → pipeline → ReplayBuffer |
| Training epochs | `--n_epochs` | 5 | server → config → pipeline → NetworkBase.train_step |
| Noise steps | `--noise_steps` | 0 | server → config (+ client CLI) |
| Noise eps min | `--noise_eps_min` | 0.1 | server → config (+ client CLI) |
| Balance sampling | `--balance_sampling` | off | server → config → ReplayBuffer + NetworkBase |

---

## 4. 与开源实现的对比

### 4.1 Value Head 设计

| 实现 | Value Head 类型 | Loss | 分类？ |
|------|----------------|------|--------|
| AlphaGo Zero (2017) | 单标量 tanh | MSE | 回归 |
| AlphaZero (2018) | 单标量 tanh | MSE | 回归 |
| **Leela Chess Zero (lc0)** | **WDL 3-class softmax** | **Cross-Entropy** | **分类** |
| MuZero (2020) | Categorical（离散化标量为多 bins） | Cross-Entropy | 分类 |
| KataGo | 混合（scalar + 辅助任务） | MSE + 辅助 | 混合 |
| **本项目** | **3-class softmax（P1win/Draw/P2win）** | **NLL** | **分类** |

**结论**：本项目的 3-class WDL 设计与 **lc0 的 WDL head 同源**，是经过验证的设计。

### 4.2 为什么 lc0 不做 class weight？

- 国际象棋高水平对局**和棋比例极高**（~50%+），三类天然均衡
- 训练数据量巨大（数十亿局）
- 搜索预算高（800+ simulations）

### 4.3 Connect4 的特殊性

- **P1 必胜游戏** → P2 win 天然稀少，且随 P1 变强越来越少
- **搜索预算低**（n=100）→ MCTS 信号噪声大
- **类别不均衡直接影响 value head 质量** → 需要 class weight 补偿

### 4.4 主流实现的采样策略

**所有主流实现都使用均匀随机采样**，没有做分层采样。原因：

1. 多数使用回归 value head，不存在 NLL 类别不均衡问题
2. 数据量大、搜索预算高，信号质量足够
3. 分层采样会改变 state distribution，可能影响 policy head 训练

本项目的情况（3-class NLL + 低搜索预算 + P1 必胜）是一个特殊组合，分层采样的收益可能大于风险。需要实验验证。

---

## 5. Hinge Loss 可行性分析

### 结论：技术可行，但不推荐

| | Cross-Entropy (当前) | Hinge Loss |
|---|---|---|
| 目标 | 学习**校准的概率分布** | 只关心**分类边界** |
| 输出 | log_softmax → 概率有物理意义 | 原始 logits → 概率无意义 |
| 梯度行为 | 所有样本都贡献梯度 | margin 足够的样本梯度为 0 |
| class weight 支持 | ✅ `F.nll_loss(weight=...)` | ✅ `MultiMarginLoss(weight=...)` |

### 核心问题：MCTS 需要校准的 value 概率

Value head 的输出（P1win/Draw/P2win 概率）被 MCTS 用于计算期望价值：

```python
value = P(P1win) × 1 + P(Draw) × 0 + P(P2win) × (-1)
```

Cross-entropy 训练出的概率是**校准的**——P(P1win)=0.7 意味着该局面约 70% 的概率 P1 获胜。

Hinge loss 只学分类边界，即使推理时加 softmax，输出的"概率"也是**未校准的**。网络倾向于对正确类给出极高置信度，导致 MCTS value 估计过度自信，搜索质量下降。

**结论**：Cross-entropy + class weight 已经同时解决了校准问题和类别不均衡问题，无需换 loss function。

---

## 6. 当前超参配置汇总

### server.py 完整参数表

| 参数 | CLI flag | 默认值 | 说明 |
|------|----------|--------|------|
| 学习率 | `--lr` | 3e-3 | 基础学习率 |
| C_puct init | `-c` / `--c_init` | 1.4 | PUCT 探索常数 |
| C_base factor | `--c_base_factor` | 1000 | c_base = n × c_base_factor |
| FPU reduction | `--fpu_reduction` | 0.2 | FPU 悲观估计系数 |
| Noise epsilon | `--eps` | 0.25 | Dirichlet noise 混合比例 |
| Noise steps | `--noise_steps` | 0 | Noise 衰减步数（0=不衰减） |
| Noise eps min | `--noise_eps_min` | 0.1 | Noise 衰减下限 |
| Dirichlet alpha | `-a` / `--alpha` | 0.03 | Dirichlet noise 浓度参数 |
| Batch size | `-b` / `--batch_size` | 512 | 训练 batch 大小 |
| Buffer size | `--buf` | 500,000 | Replay buffer 容量 |
| Replay ratio | `--replay_ratio` | 0.1 | 每步采样比例 |
| Training epochs | `--n_epochs` | 5 | 每次 policy update 的 epoch 数 |
| Simulations | `-n` | 100 | MCTS 模拟次数 |
| Discount | `--discount` | 1.0 | 折扣因子 |
| lambda_s | `--lambda_s` | 0.1 | Steps-value 混合权重 |
| Policy LR scale | `--policy_lr_scale` | 0.1 | Policy head 学习率乘数 |
| Dropout | `--dropout` | 0.2 | Dropout 率 |
| Balance sampling | `--balance_sampling` | off | 分层采样 + class weight |

### client.py 关键参数

| 参数 | CLI flag | 默认值 |
|------|----------|--------|
| Batch size (self-play) | `-B` | 100 |
| Temperature | `-t` / `--temp` | 1.0 |
| Temp threshold | `--temp_thres` | 12 |
| Noise epsilon | `--noise_eps` | 0.25 |
| Noise steps | `--noise_steps` | 0 |
| Noise eps min | `--noise_eps_min` | 0.1 |

---

## 7. 待验证实验

### 实验矩阵

| # | 配置 | 目的 | 预期效果 |
|---|------|------|----------|
| 1 | `--buf 500000`（已设） | 大 buffer 作为低通滤波器 | 减小震荡幅度 |
| 2 | `--balance_sampling` | 分层采样 + class weight | 改善 P2 win 学习 |
| 3 | `--noise_steps 12 --noise_eps 0.5` | 开局高探索衰减 | 增加 O-col3 路线曝光 |
| 4 | 1 + 2 + 3 组合 | 全部优化叠加 | 最佳效果 |

### 评估指标

1. **O col3 概率**：目标 > 60%（理想 > 80%）
2. **O policy entropy**：应持续下降（当前 ~1.29，目标 < 0.8）
3. **震荡幅度**：col2/col3/col4 的标准差应随训练收缩
4. **Value loss**：开启 class weight 后初期可能上升（P2 win 被加权），但长期应下降
5. **F1 score**：macro F1 应提升（P2 win 类别识别改善）

### 验证命令

```bash
# 实验 1: 纯大 buffer（baseline）
python server.py --buf 500000

# 实验 2: 大 buffer + balance sampling
python server.py --buf 500000 --balance_sampling

# 实验 4: 全部优化
python server.py --buf 500000 --balance_sampling --noise_steps 12 --noise_eps_min 0.1

# 客户端（配合实验 3/4）
python client.py --noise_eps 0.5 --noise_steps 12 --noise_eps_min 0.1

# 检查 buffer 内容
python tools/inspect_buffer.py
```

---

## 附录：关键代码引用

| 文件 | 位置 | 内容 |
|------|------|------|
| `src/environments/NetworkBase.py:55-61` | train_step | class weight 计算与 NLL loss |
| `src/ReplayBuffer.py:96-119` | _stratified_sample | 分层采样实现 |
| `src/game.py:47-52` | batch_self_play | noise epsilon 衰减逻辑 |
| `src/player.py` | BatchedAlphaZeroPlayer.__init__ | noise 衰减参数存储 |
| `src/cpp/BatchedMCTS.h` | set_noise_epsilon | C++ noise setter |
| `server.py:59-60` | argparse | --balance_sampling 开关 |
| `server.py:85-86` | config dict | balance_sampling 传递 |
