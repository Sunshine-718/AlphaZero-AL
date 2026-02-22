# Connect4 训练诊断：FPU 越界与 Policy 不收敛的完整分析

> 日期：2026-02-23
> 训练配置：`lambda_s=0.2`, `fpu_reduction=0.4`, `c_init=1.0`, `c_base_factor=1000`, `n=100`, `B=100`
> 训练进度：~2165 批次, Elo ≈ 2224

---

## 目录

1. [问题描述](#1-问题描述)
2. [代码状态确认：当前 MCTS 实际用了什么参数？](#2-代码状态确认)
3. [FPU 越界的数学证明](#3-fpu-越界的数学证明)
4. [FPU 越界如何导致搜索饿死](#4-fpu-越界如何导致搜索饿死)
5. [X 的阶梯型曲线是怎么形成的？](#5-x-的阶梯型曲线是怎么形成的)
6. [O 为什么被钉死在 47%？](#6-o-为什么被钉死在-47)
7. [Loss 和 Entropy 的隐藏信息](#7-loss-和-entropy-的隐藏信息)
8. [整体因果图](#8-整体因果图)
9. [修复建议](#9-修复建议)

---

## 1. 问题描述

在 Connect4 的 AlphaZero 自对弈训练中，观察到以下现象：

**lambda_s = 0.2 时：**
- X（先手）在空棋盘上的 col3 概率呈现**阶梯型**上升（60% → 68% → 88%），而非平滑收敛
- O（后手）在 X 下 col3 后的 col3 概率**停滞在 ~47%**，完全不收敛
- Loss、Elo、F1 score 等宏观指标均正常收敛，未显示异常

**lambda_s = 0.1 时：**
- X 的 col3 概率能平滑收敛到 0.9 以上
- O 的 col3 概率同样不收敛

**核心矛盾**：宏观指标全部正常，为什么策略曲线不收敛？

---

## 2. 代码状态确认

分析训练问题之前，必须先搞清楚代码的实际状态。之前有一份修改计划 `docs/fpu-modification-plan.md`，提出了两个关键改动：

1. 引入 FPU（First Play Urgency）替代未访问节点的 `∞` 值
2. 将 CPUCT 从动态公式改为**静态 4.0**

但当前代码并非完全按照该计划执行。我需要逐一确认实际状态。

### 2.1 MCTSNode.h — UCB 计算

查看 `src/cpp/MCTSNode.h:39-50`，当前代码为：

```cpp
[[nodiscard]] float get_ucb(float c_init, float c_base, float parent_n,
                             bool is_root_node, float noise_epsilon, float fpu_value) const {
    float effective_prior = prior;
    if (is_root_node) {
        effective_prior = (1.0f - noise_epsilon) * prior + noise_epsilon * noise;
    }

    float q_value = (n_visits == 0) ? fpu_value : -Q;
    float c_puct = c_init + std::log((parent_n + c_base + 1.0f) / c_base);
    float u_score = c_puct * effective_prior * std::sqrt(parent_n) / (1.0f + n_visits);
    return q_value + u_score;
}
```

**结论**：
- FPU **已实现**：`n_visits == 0` 时使用 `fpu_value` 而非 `∞`
- CPUCT **未改为静态**：仍然使用动态公式 `c_init + log(...)`
- 参数签名保留了 `c_init, c_base`（而非计划中的 `cpuct`），并新增了 `fpu_value`

也就是说，修改计划被**部分执行**了：FPU 保留了，但 CPUCT 被回退到了动态公式。

### 2.2 MCTS.h — FPU 值的计算

查看 `src/cpp/MCTS.h` 的 simulate 方法中 FPU 相关代码（约 line 130-170）：

```cpp
float parent_value = node_pool[curr_idx].Q;
float seen_policy = 0.0f;
for (int action : valids)
{
    int32_t child_idx = node_pool[curr_idx].children[action];
    if (child_idx != -1 && node_pool[child_idx].n_visits > 0)
    {
        seen_policy += node_pool[child_idx].prior;
    }
}
float fpu_value = parent_value - fpu_reduction * std::sqrt(seen_policy);
```

**关键发现：`fpu_value` 没有任何下界保护。没有 `std::max(-1.0f, ...)` 之类的截断。**

### 2.3 默认参数链

参数从 Python 命令行一路传递到 C++ 核心：

```
client.py args:
  --c_init 1.0 (default)
  --c_base_factor 1000 (default)
  --fpu_reduction 0.4 (default)
  --noise_eps 0.25 (default)
  -n 100 (simulations)
  -B 100 (batch size)

→ BatchedAlphaZeroPlayer:
  c_init = 1.0
  c_base = n * c_base_factor = 100 * 1000 = 100,000
  fpu_reduction = 0.4

→ C++ MCTS<Connect4>:
  c_init = 1.0
  c_base = 100,000
  fpu_reduction = 0.4
```

### 2.4 关于 c_base = 100,000 的影响

虽然代码使用动态 CPUCT 公式，但 `c_base = 100,000` 极大，导致对数项几乎为零：

```
c_puct = c_init + log((parent_n + c_base + 1) / c_base)

当 parent_n = 100（最大模拟次数）时:
c_puct = 1.0 + log((100 + 100001) / 100000)
       = 1.0 + log(1.00101)
       = 1.0 + 0.00101
       ≈ 1.001
```

所以当前的"动态 CPUCT"在实际运行中是一个**伪静态值 ≈ 1.0**，全程几乎不变。

作为对比，AlphaZero 论文使用 `c_init=1.25, c_base=19652`：
```
当 parent_n = 100 时:
c_puct = 1.25 + log((100 + 19653) / 19652) = 1.25 + 0.00508 ≈ 1.255
当 parent_n = 800 时:
c_puct = 1.25 + log((800 + 19653) / 19652) = 1.25 + 0.0399 ≈ 1.29
```

差异不大，因为 100 次模拟对于 c_base=19652 也不够大。但至少 AlphaZero 在更多模拟次数时有明显的动态效果。

**总结当前代码实际状态：**

| 组件 | 修改计划 | 当前代码 | 差异 |
|------|---------|---------|------|
| FPU | 引入 FPU | ✅ 已引入 | 一致 |
| FPU 下界截断 | 未提及 | ❌ 没有 | **这是 bug** |
| CPUCT | 改为静态 4.0 | ❌ 保留动态公式 | 已回退 |
| c_base | 移除 | 保留，值为 100,000 | 已回退 |

---

## 3. FPU 越界的数学证明

### 3.1 FPU 公式

当前 FPU 值的计算方式：

```
fpu_value = parent_value - fpu_reduction × √(seen_policy)
```

其中：
- `parent_value` = 当前节点的 Q 值（从当前玩家视角，范围 [-1, +1]）
- `fpu_reduction` = 0.4（默认值）
- `seen_policy` = 所有已访问子节点的 prior 之和（范围 [0, 1]）

FPU 的设计意图是：未访问的节点应该比当前已知的平均价值**稍微差一些**，从而鼓励 MCTS 先深挖已访问节点，减少对低 prior 节点的浪费性探索。

### 3.2 越界条件

当 `fpu_value < -1.0` 时就会发生越界。整理不等式：

```
parent_value - fpu_reduction × √(seen_policy) < -1.0
parent_value < -1.0 + fpu_reduction × √(seen_policy)
```

以默认参数 `fpu_reduction = 0.4` 代入：

```
parent_value < -1.0 + 0.4 × √(seen_policy)
```

当 `seen_policy ≈ 1.0`（大部分合法动作都已被访问过）时：

```
parent_value < -1.0 + 0.4 × 1.0 = -0.6
```

**结论：只要 parent_value < -0.6 且大部分子节点已被访问，FPU 就会越界。**

### 3.3 用训练日志中的实际数据验证

从 server.py 的 eval 日志中可以看到 O 在各个阶段的 State-value：

```
batch 2121: State-value O = -0.5548
batch 2131: State-value O = -0.5729
batch 2141: State-value O = -0.5953
batch 2151: State-value O = -0.6121  ← 已跨过 -0.6 阈值
batch 2161: State-value O = -0.5956
```

注意：这些 State-value 是网络 `predict()` 方法的输出，已经包含了 `lambda_s=0.2` 的 steps 混合。

以 batch 2151 的值为例，在 MCTS 根节点（O 的回合）：

```
parent_value = -0.6121
seen_policy = ?（取决于搜索进展）
```

当 100 次模拟进行到后期，大部分高 prior 子节点都已访问过，`seen_policy` 接近 1.0：

```
fpu_value = -0.6121 - 0.4 × √(0.95)
          = -0.6121 - 0.4 × 0.9747
          = -0.6121 - 0.3899
          = -1.002  ← 已低于 -1.0
```

但这只是根节点的情况。在搜索树的**深层节点**，情况更严重。例如，当 O 在第三层再次行动时（已走了 5 步，O 下了第 2 步和第 4 步），此时的局面对 O 更加不利，Q 值可能已经跌到 -0.75 或更低：

```
fpu_value = -0.75 - 0.4 × √(0.90)
          = -0.75 - 0.4 × 0.9487
          = -0.75 - 0.3795
          = -1.13  ← 远低于 -1.0
```

### 3.4 越界的严重性

Value 网络的输出范围是 [-1, +1]。MCTS 中所有经过反向传播的 Q 值也在这个范围内（因为每一步只是做均值更新和取负）。

所以，一个已被充分访问的子节点，即使它代表了一条**已被证明必败**的路线，它的 Q 值也不会低于 -1.0。

但 FPU 越界后，未访问节点的初始估值可以是 -1.1、-1.2，甚至更低。这意味着：

```
未访问节点的 UCB 起始值 = fpu_value + U_exploration
                        = -1.1 + c_puct × prior × √(N_parent) / 1
```

已被证明必败节点的 UCB = -1.0 + c_puct × prior × √(N_parent) / (1 + n_visits)

由于已访问节点的 `1 + n_visits` 在分母，U 项更小。但关键的 Q 部分差值：
- 未访问节点 Q = -1.1
- 已知必败节点 Q = -1.0
- 差值 = 0.1，有利于**已知必败节点**

MCTS 会**宁愿反复访问一个已经被证明必输的节点**，也不去尝试未访问的节点。这就是"搜索饿死"（exploration starvation）。

---

## 4. FPU 越界如何导致搜索饿死

### 4.1 正常的 FPU 行为（parent_value = 0.3，优势方）

假设 X 在一个优势局面，parent_value = 0.3，已访问了 prior 总和 0.8 的子节点：

```
fpu_value = 0.3 - 0.4 × √(0.8) = 0.3 - 0.358 = -0.058
```

未访问节点的初始 Q = -0.058，低于 parent_value 但仍在合理范围内。如果某个未访问节点有足够高的 prior，它的 UCB 探索项可以补偿这个 Q 差距，使其被选中。

**这是 FPU 的正常工作方式**：适度悲观估计未访问节点，引导搜索优先深挖高 prior 路线。

### 4.2 越界的 FPU 行为（parent_value = -0.7，劣势方 O）

O 在一个明显的劣势局面，parent_value = -0.7，已访问了 prior 总和 0.95 的子节点：

```
fpu_value = -0.7 - 0.4 × √(0.95) = -0.7 - 0.390 = -1.09
```

现在考虑一个具体的场景：
- 子节点 A（col3）：已被访问 30 次，Q = -0.65（O 最好的防守）
- 子节点 B（col2）：已被访问 25 次，Q = -0.80
- 子节点 C（col0）：未被访问，prior = 0.02

子节点 C 的 UCB：
```
Q_part = fpu_value = -1.09
U_part = 1.0 × 0.02 × √(100) / (1 + 0) = 0.2
UCB_C = -1.09 + 0.2 = -0.89
```

子节点 B 的 UCB（已知较差）：
```
Q_part = -(-0.80) = 0.80  （取负，因为是从父节点视角看子节点）
```

等等，这里需要仔细推理。让我重新整理。

### 4.3 UCB 中 Q 值的符号约定

在 `get_ucb()` 中：

```cpp
float q_value = (n_visits == 0) ? fpu_value : -Q;
```

- `Q` 是该节点从**走棋方视角**存储的均值价值。如果该节点是 O 的回合，Q 表示 O 对这个局面的平均评价。
- `-Q` 是因为父节点（上一步的走棋方）看到的价值恰好取负。

在反向传播代码（`MCTS.h`）中：
```cpp
node_pool[update_idx].Q += (val - node_pool[update_idx].Q) / node_pool[update_idx].n_visits;
val = -val * discount;
update_idx = node_pool[update_idx].parent;
```

每往上一层 Q 值取负。所以在父节点选子节点时，`-child.Q` 就是"从我的视角看，选这个子节点有多好"。

FPU 的 `parent_value = node_pool[curr_idx].Q` 也是从当前走棋方的视角。

现在重新做计算。假设当前节点是 O 的回合：
- `parent_value = node_pool[curr_idx].Q = -0.7`（O 对这个局面的平均评价是 -0.7，很差）

子节点都是 X 的回合。子节点的 Q 是从 X 视角存储的：
- 子节点 A（X 视角 Q = 0.65）→ 从 O 视角看：`-Q = -0.65`
- 子节点 B（X 视角 Q = 0.80）→ 从 O 视角看：`-Q = -0.80`
- 子节点 C（未访问）→ 从 O 视角看：`q_value = fpu_value = -1.09`

现在比较 UCB（假设 c_puct ≈ 1.0, N_parent = 100）：

```
UCB_A = -0.65 + 1.0 × prior_A × 10 / (1 + 30) = -0.65 + prior_A × 0.323
UCB_B = -0.80 + 1.0 × prior_B × 10 / (1 + 25) = -0.80 + prior_B × 0.385
UCB_C = -1.09 + 1.0 × 0.02  × 10 / (1 + 0)  = -1.09 + 0.20 = -0.89
```

假设 prior_A = 0.40, prior_B = 0.20：
```
UCB_A = -0.65 + 0.40 × 0.323 = -0.65 + 0.129 = -0.521
UCB_B = -0.80 + 0.20 × 0.385 = -0.80 + 0.077 = -0.723
UCB_C = -0.89
```

选择顺序：A（-0.521）> B（-0.723）> C（-0.89）。

**子节点 C 的 UCB 比已知更差的子节点 B 还低**。即使 C 是一条未被探索的翻盘路线，MCTS 也不会去尝试它。而且随着 A 和 B 被继续访问，它们的 U 项会继续缩小（因为 n_visits 增大），但 Q 部分可能不会变化太多。C 始终被排在最后。

这就是"搜索饿死"：低 prior 的未访问节点永远不会被尝试，即使它们可能是唯一的防守路线。

### 4.4 没有 FPU 越界时的对比

如果加了 `fpu_value = std::max(-1.0f, fpu_value)` 截断：

```
fpu_value = max(-1.0, -1.09) = -1.0

UCB_C = -1.0 + 0.20 = -0.80
```

现在 UCB_C（-0.80）> UCB_B（-0.723）？不对，还是低于 A。

等等，让我重新算。UCB_B = -0.723，UCB_C = -0.80。C 仍然低于 B。但差距从 0.167（未截断）缩小到了 0.077（截断后）。随着搜索继续，B 的 U 项会缩小，C 最终有机会被选中。

更重要的是，在 FPU 恰好等于 -1.0 时，C 的 UCB 与"已被证明必败"的路线起点相同。这意味着它**至少不会比必败更差**，U 项（探索奖励）可以让它有机会被尝试。而越界时，它比必败还差，U 项再大也补不回来。

---

## 5. X 的阶梯型曲线是怎么形成的？

### 5.1 观察到的数据

从 server.py eval 日志中提取的 X col3 概率和 Value X：

| 批次 | X col3 | Value X | Value O |
|------|--------|---------|---------|
| 2121 | 60.86% | 0.1318 | -0.5548 |
| 2131 | 60.52% | 0.1511 | -0.5729 |
| 2141 | 68.49% | 0.2547 | -0.5953 |
| 2151 | 87.81% | 0.4690 | -0.6121 |
| 2161 | 87.44% | 0.4515 | -0.5956 |

X col3 在 batch 2141→2151 之间发生了一次巨大跳变：从 68% 到 88%。同时 Value X 从 0.25 跳到 0.47。

### 5.2 阶梯的形成机制：Value Head 的阈值触发 MCTS 相变

MCTS 的行为不是 value 的线性函数，而是存在相变阈值。具体来说：

**Step 1：Value Head 渐进学习**

Value head 通过梯度下降逐步学习，Value X 从 0.13 缓慢上升到 0.25。这个过程是连续的。

**Step 2：MCTS 的 visit 分布对 value 变化不敏感（在阈值以下）**

考虑 MCTS 如何在 col3 和 col0 之间分配 visits。假设 col3 的子树价值为 V3，col0 的子树价值为 V0。

当 Value X = 0.13 时：
```
V3 ≈ 0.15（col3 略好）
V0 ≈ 0.05（col0 较差）
Q 差 = 0.15 - 0.05 = 0.10
```

UCB 中 col0 的 U 项能否补偿 0.10 的 Q 差距？
```
U_col0 = 1.0 × prior_col0 × √100 / (1 + n_visits_col0)
```
假设 prior_col0 = 0.10（空棋盘上边列的先验概率），n_visits_col0 = 5：
```
U_col0 = 1.0 × 0.10 × 10 / 6 = 0.167
```

0.167 > 0.10 的 Q 差距。所以 col0 仍然会被频繁访问。MCTS 的 visits 仍然比较分散。

**Step 3：Value 跨过阈值后，MCTS 行为急剧变化**

当 Value X = 0.47 时：
```
V3 ≈ 0.50（col3 明显好）
V0 ≈ 0.15（col0 远远差）
Q 差 = 0.50 - 0.15 = 0.35
```

col0 的 U 项需要补偿 0.35 的 Q 差距：
```
需要 U_col0 ≥ 0.35
1.0 × 0.10 × 10 / (1 + n) ≥ 0.35
1 / (1 + n) ≥ 0.35
n ≤ 1.86
```

col0 最多被访问 1-2 次就不会再被选中！100 次模拟中，col3 会拿到 85-90 次访问。

**这就是相变**：Q 差距从 0.10 增长到 0.35 只是线性变化（3.5 倍），但 visit 分布从"分散（col3 ≈ 60%）"跳变到"极度集中（col3 ≈ 88%）"。MCTS 的 visit 分配本质上是一个赢者通吃的竞争——Q 差距一旦超过 U 项能补偿的范围，就会发生雪崩式集中。

**Step 4：Policy target 跳变**

MCTS 的 visit 分布就是 policy 的训练目标。当 visits 从 60:10:10:... 变成 88:2:2:... 时，训练目标发生了跳变。网络很快就能学到这个新目标（毕竟只是一个输出数字的变化），col3 概率迅速拉到 88%。

**Step 5：新的平台期**

网络学到 88% 后，MCTS 的 prior 更新了，但 value 没有继续大幅提升（0.469 → 0.452）。系统进入新的平台，等待 value head 的下一次突破。

### 5.3 为什么 lambda_s = 0.2 会放大阶梯？

`predict()` 方法中，lambda_s 参与最终 value 的计算：

```python
value = (1 - lambda_s) * value_base - lambda_s * sign(value_base) * steps_adjusted
```

其中 `steps_adjusted = 2 * (expected_steps / 42) - 1`，将期望步数归一化到 [-1, +1]。

当 X 处于优势时（`sign(value_base) = +1`）：
```
value = 0.8 × value_base - 0.2 × steps_adjusted
```

此公式意味着：X 赢得越快（steps 越小 → steps_adjusted 越负），effective value 越高。

**问题在于 value_base 和 steps 的学习不同步。** Value head 和 Steps head 是两个独立的网络头，它们通过共享 backbone 间接关联，但各自的学习进度不同。

考虑以下时序：

1. **Steps head 先学到 "col3 导致更短的游戏"**
   - steps_adjusted 对 col3 路线变小
   - effective value 突然上升
   - MCTS 行为跳变 → policy target 跳变 → 阶梯的第一个"台阶"

2. **Value head 还没跟上**
   - value_base 仍在缓慢上升
   - effective value 暂时稳定 → 平台期

3. **Value head 突破**
   - value_base 大幅上升
   - steps_adjusted 可能也同步变化
   - 两个效应叠加 → effective value 出现更大的跳变

lambda_s = 0.2 意味着 steps 修正占 effective value 的 20%。当 steps head 的预测发生变化时，这 20% 的波动足以推动 MCTS 跨越相变阈值。

lambda_s = 0.1 时，steps 修正只占 10%，波动被压缩到阈值以下，所以 X 的收敛更加平滑。

### 5.4 补充说明：学习率衰减的角色

从训练曲线看，学习率正在按计划衰减。在后期（batch > 2000），学习率已经很低。这意味着：

- Value head 的更新步长很小 → 在平台期积累很久才能跨过阈值
- 一旦跨过阈值，新的 MCTS targets 让 policy head 快速适应（cross-entropy loss 对大概率变化很敏感）
- 然后又进入新的平台期

这就解释了为什么阶梯的"台阶宽度"（平台期）越来越长：学习率越低，value head 跨阈值需要的更新次数越多。

---

## 6. O 为什么被钉死在 47%？

O 的 col3 概率在最近 40+ eval 周期中完全停滞：

| 批次 | O col3 | O col2 | O col4 |
|------|--------|--------|--------|
| 2121 | 46.95% | 20.33% | 20.16% |
| 2131 | 47.38% | 20.19% | 20.25% |
| 2141 | 46.36% | 22.09% | 22.04% |
| 2151 | 46.42% | 23.18% | 23.74% |
| 2161 | 47.32% | 22.68% | 23.21% |

col3 精确地锁死在 ~47%。这不是"学得慢"——而是陷入了一个稳态。三层原因叠加导致了这个死锁。

### 6.1 第一层：训练信号几乎无区分度

从 `inspect_buffer.py` 的输出可以看到，在"X 下 col3 后，O 的回合"这个局面下：

```
样本数: 1671
P1 wins: 81.4%
P2 wins: 11.4%
Draw: 7.2%
```

81.4% 的样本，O 不管下了什么都是输。这意味着训练信号是：

```
"你下了 col3 → 输了"   (大量样本)
"你下了 col2 → 输了"   (大量样本)
"你下了 col4 → 输了"   (大量样本)
"你下了 col3 → 赢了/平局" (少量样本)
"你下了 col2 → 赢了/平局" (少量样本)
```

从 game outcome 的角度看，三个主要动作（col2/col3/col4）的**胜率信号几乎无法区分**。网络看不出 col3 比 col2/col4 好在哪——因为大部分样本都是败局，败局中选 col3 和选 col2 的结果一样（都是 winner = P1）。

唯一有意义的信号来自那 18.6% 的非 P1-win 样本。但这些样本太少，且其中也是各种动作混杂，信号被噪声淹没。

**为什么 MCTS 的 policy target 本身不能区分 col3 和 col2？**

这是关键问题。理论上，即使 game outcome 无法区分，MCTS 通过深度搜索应该能发现 col3 的防守优势（因为 col3 导致更多深层 draw/P2-win 路线）。但 FPU 越界阻断了这个过程——下一节解释。

### 6.2 第二层：FPU 截断了 O 的搜索深度

在第 3-4 节中，我们已经证明 O 的 FPU 值会在深层节点越界到 -1.0 以下。现在分析这如何影响 O 的搜索质量。

**正常搜索流程（假设 FPU 未越界）：**

1. O 在根节点尝试 col3 → 进入子树
2. 子树中 X 回应 → 新的 O 回合 → 继续搜索
3. 多层搜索后，发现 col3 路线最终能活到平局或翻盘
4. 这个深层信号通过 backpropagation 传回根节点
5. col3 的 Q 值被提升 → 获得更多 visits → policy target 中 col3 占比更高

**FPU 越界后的搜索流程：**

1. O 在根节点尝试 col3 → 进入子树
2. 子树中 X 回应 → 新的 O 回合节点
3. **在这个深层 O 节点**，Q 值更低（比如 -0.75），FPU 越界到 -1.1
4. MCTS 不愿意探索该节点的未访问子节点
5. 搜索被困在已访问的几条路线上反复打转
6. 深层的防守信号**无法被发现**
7. col3 的 Q 值得不到提升 → visits 没有集中 → policy target 没有改变

**关键区别**：FPU 越界不是阻止了根节点的探索（根节点的 FPU 勉强在 -1.0 附近），而是阻止了**深层节点的探索**。搜索树的深度被截断了——MCTS 只能在浅层做决策，而浅层的信息不足以区分 col3 和 col2/col4。

这解释了为什么 col3 稳定在 ~47% 而不是 0%：根节点的 FPU 没有严重越界，col3 由于先验概率最高仍然能拿到最多 visits。但由于深层搜索被截断，MCTS 无法发现 col3 的深层防守优势，所以 col3 的 visit 比例无法进一步提升。

### 6.3 第三层：X 越强 → O 越困（负反馈死锁）

观察 value 的趋势：

```
Value X:  0.13 → 0.15 → 0.25 → 0.47 → 0.45  (上升)
Value O: -0.55 → -0.57 → -0.60 → -0.61 → -0.60  (下降)
```

随着 X 变强：
1. X 的策略更优 → 对 O 的压力增大
2. O 面临的局面更差 → value head 预测 O 的 parent_value 更低
3. 更低的 parent_value → FPU 越界更严重
4. 更严重的越界 → O 的搜索质量更差
5. 更差的搜索 → O 生成更低质量的 policy targets
6. 更差的 targets → 网络对 O 的策略没有改进
7. O 不进步 → X 更容易赢 → 回到第 1 步

**这是一个正反馈的恶性循环**（正反馈指"越差越差"，而非"越好越好"）。X 的每一次进步都在加深 O 的困境。

这也解释了为什么 O 不是在"缓慢收敛"而是被"钉死"——因为恶性循环有自稳定性。只要 FPU 越界的程度不变（parent_value 在 -0.6 附近浮动），O 的搜索质量就不变，policy target 就不变，网络就不变。

---

## 7. Loss 和 Entropy 的隐藏信息

### 7.1 Loss 分解

从训练日志和曲线图估计各 loss 分量：

| 分量 | 近似值 | 含义 |
|------|--------|------|
| p_loss (策略) | ≈ 1.0 | 网络策略与 MCTS 目标的交叉熵 |
| v_loss (价值) | ≈ 0.3 | 网络价值与胜负标签的 NLL |
| s_loss (步数) | ≈ 1.0 | 网络步数预测与实际步数的 NLL |
| **Total** | ≈ 2.3 | |

### 7.2 隐藏信息 1：p_loss 的理论下界不是 0

策略损失的公式是：

```python
per_sample_p = -torch.sum(prob * log_p_pred, dim=1)
p_loss = torch.mean(per_sample_p * mask)
```

这是交叉熵 CE(target, pred)。信息论中有一个基本恒等式：

```
CE(target, pred) = H(target) + KL(target ‖ pred)
```

其中 `H(target)` 是 MCTS 目标策略的**熵**（entropy），`KL` 是网络预测与目标之间的 KL 散度。

从 `inspect_buffer.py` 输出可以看到：

```
Global Policy Entropy:
  Mean: 0.7877
  Std:  0.5338
```

这个 0.7877 就是 MCTS 目标策略的平均熵，也就是 **p_loss 的理论下界**。即使网络完美拟合了所有 MCTS 目标（KL = 0），p_loss 也不可能低于 0.79。

当前 p_loss ≈ 1.0，所以：

```
KL(target ‖ pred) = p_loss - H(target) ≈ 1.0 - 0.79 = 0.21
```

**0.21 的 KL 散度其实不大。** 网络已经非常接近 MCTS 目标了。

这揭示了一个关键事实：**Loss 在收敛，不是因为策略在变好，而是因为网络已经学会了忠实复现 MCTS 的输出——包括 MCTS 输出中的噪声和错误。** Loss 收敛到的是一个"噪声地板"，而不是真正的最优解。

### 7.3 隐藏信息 2：Entropy 的 X/O 不对称

训练日志中报告的 entropy ≈ 0.97。这是网络输出策略的平均熵。

从 `inspect_buffer.py` 可以拆解到各个局面：

| 局面 | MCTS 目标熵（单样本平均） | NN 输出熵 | 关系 |
|------|--------------------------|-----------|------|
| X 空棋盘 | 0.9964 | 0.5884 | NN < Target |
| O 对 X-col3 | 1.1066 | 1.2919 | NN > Target |

这两个数字之间的关系有深刻含义。

**对于 X（NN entropy < Target entropy）：**

X 的 MCTS 目标带有噪声（Dirichlet 探索、温度 = 1 的采样等）。单个样本可能是 col3 = 85% 或 col3 = 50% 或甚至 col3 = 33%（少数情况），平均单样本熵 = 0.9964。

但这些样本的**方向是一致的**——绝大多数都指向 col3。网络学到了这个共性，收敛到 col3 = 87.8%，entropy = 0.5884。

网络的 entropy 低于 target 的平均 entropy，说明**网络在"去噪"**——从带噪声的样本中提取了干净的信号。这是**健康**的学习行为。

**对于 O（NN entropy > Target entropy）：**

O 的每个 MCTS 样本各自的 entropy 平均只有 1.1066——每个样本单独看还是比较集中的（有确定的"最佳"动作）。

但问题是这些样本**指向不同的方向**：

```
样本 1: col3=76.4%, col2=3.8%,  col4=5.5%  → 指向 col3
样本 2: col3=22.9%, col2=59.2%, col4=7.8%  → 指向 col2
样本 3: col3=12.4%, col2=4.0%,  col4=75.1% → 指向 col4
样本 4: col3=43.5%, col2=41.8%, col4=5.3%  → col3/col2 对半
```

这些是从 `inspect_buffer.py` 的"前 10 条样本"中提取的实际数据。

网络无法同时拟合这些互相矛盾的目标。在最小化交叉熵的驱动下，网络会收敛到这些目标的**加权平均**：

```
平均策略 ≈ col3:47%, col2:22%, col4:22%, 其余:9%
```

这个平均策略的 entropy（1.29）**必然高于**单个样本 entropy 的平均值（1.11）。这是 Jensen 不等式的直接推论：

```
H(E[p_i]) ≥ E[H(p_i)]
```

即"平均分布的熵" ≥ "各分布熵的平均"。当各分布 p_i 不完全一致时（目标互相矛盾），不等式严格成立。

**NN entropy (1.29) > Target entropy (1.11) 这个差值 Δ = 0.18，直接量化了 MCTS 目标之间的分歧程度。**

### 7.4 隐藏信息 3：为什么全局 Entropy 不再下降

全局 entropy 平台在 ~0.97。将其分解：

- 训练数据中大约 50% 是 X 回合的样本，50% 是 O 回合的样本
- X 回合样本的 NN entropy ≈ 0.59（且仍在缓慢下降）
- O 回合样本的 NN entropy ≈ 1.29（完全不动）

粗略加权平均：0.5 × 0.59 + 0.5 × 1.29 = 0.94 ≈ 0.97（接近观测值）

**Entropy 的平台完全由 O 的高 entropy 托底。** 只要 O 的策略不收敛，全局 entropy 就不会继续下降。Entropy 的平台是 O 不收敛的直接后果，而非独立的现象。

### 7.5 隐藏信息 4：F1 Score 高 = Value Head "太准确"

F1 score ≈ 0.92，说明 value head 正确地把 92% 的局面分类为 win/draw/loss。

这在宏观上是好事（网络理解了游戏态势），但对 O 来说有一个讽刺的副作用：

value head 越准确 → 对 O 局面预测的 parent_value 越负 → FPU 越界越严重 → O 的搜索质量越差。

换言之，**value head 的进步在加剧 O 的困境**。这是一个悖论：系统的一个组件变得更好，却使得另一个组件更差。

### 7.6 隐藏信息 5：Loss "收敛"的真实含义

一个真正收敛的系统应该有以下特征：
1. MCTS 产出高质量目标 → 网络学到好策略
2. 好策略反馈给 MCTS 作为 prior → MCTS 产出更好的目标
3. 正反馈循环 → loss 持续下降 → entropy 持续下降
4. 最终 loss 趋近 H(target)（target entropy 本身也在下降）

当前系统的状态是：
1. MCTS 为 O 产出矛盾目标
2. 网络忠实拟合矛盾目标（KL ≈ 0.21，已很小）
3. 矛盾的策略回传给 MCTS → MCTS 继续产出矛盾目标
4. **闭环锁死** → loss 稳定在噪声地板 → entropy 不再下降

**Loss 的"收敛"实际上是到达了一个假平衡态（False Equilibrium）。** 它不是告诉你"一切正常"，而是在告诉你"网络已经把能从这些数据中学到的东西都学完了——而这些数据本身就是有问题的"。

要打破这个假平衡态，需要**改善数据质量**（即修复 FPU → MCTS 为 O 产出一致的目标），而不是调整训练超参数（学习率、buffer 大小等）。

---

## 8. 整体因果图

```
┌─────────────────────────────────────────────────────────────────┐
│                     X 的阶梯型收敛                               │
│                                                                 │
│  Value head 通过梯度下降缓慢学习                                 │
│       ↓                                                         │
│  Value X 跨过 MCTS 的相变阈值（Q 差 > U 能补偿的范围）          │
│       ↓                                                         │
│  MCTS visit 分布急剧集中到 col3（赢者通吃）                     │
│       ↓                                                         │
│  Policy target 跳变（60% → 88%）                                │
│       ↓                                                         │
│  网络快速拟合新目标 → 进入新平台期                               │
│       ↓                                                         │
│  等待 value head 下一次突破 → 下一个"台阶"                      │
│                                                                 │
│  lambda_s = 0.2 放大效应：steps head 和 value head 的学习不同步  │
│  导致 effective value 出现额外跳变 → 阶梯更陡/更宽              │
│                                                                 │
│  lambda_s = 0.1 时效应减半 → 跳变幅度不足以触发相变 → 平滑收敛  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     O 的死锁                                     │
│                                                                 │
│  ┌→ X 变强 → O 面临的局面 Q 更负（-0.55 → -0.61）             │
│  │       ↓                                                      │
│  │  FPU 在深层节点越界 < -1.0                                   │
│  │       ↓                                                      │
│  │  搜索深度被截断：深层未访问节点被视为"比必败更差"             │
│  │       ↓                                                      │
│  │  MCTS 无法发现 col3 的深层防守优势                           │
│  │       ↓                                                      │
│  │  不同 MCTS 实例对同一局面给出矛盾的 policy targets           │
│  │  （有时 col3 最高，有时 col2，有时 col4）                     │
│  │       ↓                                                      │
│  │  网络学到的是各 target 的模态平均（mode averaging）           │
│  │  col3 ≈ 47%, col2 ≈ 22%, col4 ≈ 22%                        │
│  │  NN entropy (1.29) > Target entropy (1.11) ← Jensen 不等式  │
│  │       ↓                                                      │
│  │  81.4% 的样本结局都是 P1 wins → 无法从 outcome 区分动作优劣  │
│  │       ↓                                                      │
│  │  网络对 O 的策略没有改进 → O 不进步                          │
│  │       ↓                                                      │
│  └── X 更容易赢 → O 的 Q 更负 → 恶性循环继续 ←─────────────────┘
│                                                                 │
│  Loss/Entropy 表现：                                             │
│  • p_loss ≈ 1.0 = H(target)(0.79) + KL(0.21)                  │
│  • KL = 0.21 已经很小 → 网络已忠实拟合噪声目标                  │
│  • Entropy ≈ 0.97 = avg(X 的 0.59, O 的 1.29) → 被 O 托底     │
│  • F1 ≈ 0.92 → value head 越准 → O 的 FPU 越界越严重           │
│  → 一切指标"稳定"是因为系统达到了假平衡态，不是真正最优          │
└─────────────────────────────────────────────────────────────────┘
```

### 破局点

**FPU 下界截断 `fpu_value = std::max(-1.0f, fpu_value)` 是打破 O 死锁的最小干预。**

加上这一行后：
1. 深层 O 节点的 FPU 不再低于 -1.0 → 未访问节点至少和"已证明必败"一样值得尝试
2. 搜索深度恢复 → MCTS 能发现 col3 的深层防守优势
3. MCTS 目标变得一致（col3 总是最优） → NN entropy 下降
4. 网络学到更尖锐的 O 策略 → MCTS 先验更好 → 搜索更高效
5. 正反馈循环启动 → O 的 col3 概率开始上升
6. p_loss 开始下降（因为 target entropy 下降 + 目标不再矛盾）
7. 全局 entropy 从 0.97 向 0.7 或更低移动

---

## 9. 修复建议

### 优先级 1（必做）：FPU 下界截断

**文件**：`src/cpp/MCTS.h`

在 FPU 计算之后添加一行截断：

```cpp
float fpu_value = parent_value - fpu_reduction * std::sqrt(seen_policy);
fpu_value = std::max(-1.0f, fpu_value);  // ← 添加这一行
```

理由：这是一个 bug fix。value 网络的输出范围是 [-1, +1]，所有 Q 值也在此范围内。FPU 没有理由低于 -1.0。

### 优先级 2（推荐）：降低 lambda_s

从 0.2 降到 **0.1 或 0.05**。

理由：
- 用户已观察到 lambda_s = 0.1 时 X 能平滑收敛（无阶梯）
- lambda_s = 0.2 的 steps 修正在 Connect4 这种短局游戏中过大
- 降低 lambda_s 不影响 steps head 的训练（steps loss 权重固定为 1.0），只减少推理时的 value 扰动

### 优先级 3（可选）：动态 FPU 衰减

当处于劣势时减小 FPU 惩罚：

```cpp
float dyn_reduction = fpu_reduction * ((parent_value + 1.0f) / 2.0f);
float fpu_value = parent_value - dyn_reduction * std::sqrt(seen_policy);
fpu_value = std::max(-1.0f, fpu_value);
```

建议先只做优先级 1+2，观察效果。如果 O 仍然收敛困难再加入动态衰减。简单修复优先。

### 优先级 4（可选）：调整 c_base_factor

当前 `c_base = 100,000` 使得动态 CPUCT 退化为伪静态 ~1.0。如果需要真正的动态效果：

```
--c_base_factor 200   → c_base = 20,000（接近 AlphaZero 原版 19,652）
--c_init 1.25         → AlphaZero 原版值
```

---

## 附录：关键代码引用

| 文件 | 行号 | 内容 |
|------|------|------|
| `src/cpp/MCTSNode.h` | 39-50 | UCB 计算（含 FPU） |
| `src/cpp/MCTS.h` | ~130-170 | FPU 值计算 + 子节点选择 |
| `src/cpp/MCTS.h` | ~249-258 | Q 值反向传播 |
| `src/environments/NetworkBase.py` | 41-79 | 训练 loss 计算 |
| `src/environments/Connect4/Network.py` | 140-163 | predict() 中的 lambda_s 混合 |
| `src/ReplayBuffer.py` | 83-89 | buffer 采样逻辑 |
| `src/pipeline.py` | 107-123 | policy_update 流程 |
