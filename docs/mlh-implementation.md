# LC0 风格 Moves Left Head (MLH) 实现文档

## 背景

### 策略追逐问题 (Policy Chasing)

Connect4 是已解决的博弈，先手有必胜策略。当 AlphaZero 训练到后期，输方的所有走法 Q 值都趋近于 -1，导致 PUCT 选择退化为纯策略先验驱动：

```
UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
```

当所有 `Q(s,a) ≈ -1` 时，`Q` 项无法区分走法，选择完全由 `P(s,a)` 主导。网络学习的策略又被这种搜索反馈强化，形成自我强化的循环——即"策略追逐"。

### 旧方案：backprop 时修改 value

旧版 MLH 在 `backprop()` 中直接修改 value：

```python
# 旧方案（已废弃）
if |value| > threshold:
    activation = (|value| - threshold) / (1 - threshold)
    bonus = mlh_factor * (expected_steps / 42.0) * activation
    value += bonus   (if losing)
    value -= bonus   (if winning)
```

问题：
1. 使用 `expected_steps / 42.0` 做归一化——**绝对值**归一化依赖于游戏最大步数
2. 在 backprop 时一次性调整 value——只作用于叶节点，不随搜索积累
3. 需要 `mlh_threshold` 硬阈值来门控——多一个需要调参的超参数

### LC0 的做法：选择时使用 M_utility

Leela Chess Zero 在 **PUCT 选择阶段** 添加一个额外的 M_utility 项，利用子节点与父节点的**相对步数差**来指导搜索：

```
UCB = Q + U + M_utility
M_utility = clamp(slope * (child_M - parent_M), -cap, cap) * FastSign(-q)
```

核心思想：
- 每个节点维护一个 moves-left 的 **running average（M）**
- 通过 `child_M - parent_M` 得到**相对差值**，而非绝对步数
- 用 clamp 限制最大影响幅度
- 用 Q 值的符号来决定方向（赢着偏好快赢，输着偏好慢输）

---

## 新方案：本项目的 LC0 风格实现

### 核心公式

```
M_utility = clamp(slope × (child_M - parent_M), -cap, cap) × Q
```

其中：
- `child_M`：子节点的 M（预期剩余步数的 running average）
- `parent_M`：父节点的 M
- `slope`：控制 M 差值的灵敏度
- `cap`：限制 M_utility 的最大幅度
- `Q`：子节点自身视角的 Q 值（关键改进，见下文）

### 与 LC0 的关键区别：用 Q 代替 sign(-q)

LC0 原始公式用 `FastSign(-q)`（q 为父节点视角），输出 `{-1, 0, +1}`。本项目改用子节点视角的 **原始 Q 值**：

| 方面 | LC0 | 本项目 |
|------|-----|--------|
| 符号函数 | `FastSign(-q)` 硬切换 | `Q` 连续值 |
| Q 约定 | q 为父节点视角 | Q 为子节点视角 |
| 均势处理 | 需要 threshold 阈值参数 | Q ≈ 0 时自动无效 |
| 效果渐变 | 阶梯式 {-1, 0, +1} | 平滑过渡 [−1, +1] |

使用 `Q` 代替 `sign` 的优势：
1. **天然 Q-gating**：均势（Q ≈ 0）时 M_utility 自动趋近零，无需 threshold 参数
2. **平滑过渡**：微弱优势时轻微影响，大优势时强影响
3. **减少超参数**：从 3 个（slope, cap, threshold）减为 2 个（slope, cap）

### Q 值符号约定与 sign bug 复盘

本次实现中曾出现一个符号错误，根源在于两套不同的 Q 值存储约定：

| | 本项目 | LC0 |
|---|---|---|
| **Q 存储视角** | 子节点自身（child's perspective） | 父节点（parent's perspective） |
| **父方赢着时** | Q < 0 | q > 0 |
| **父方输着时** | Q > 0 | q < 0 |

**错误写法**：`sign(-Q)` — 误以为 `-Q` 对应 LC0 的 `q`，则 `sign(-Q)` 对应 `FastSign(-q)`。

**实际对应关系**：
- 我们的 `-Q` ≡ LC0 的 `q`（都是父节点视角）
- 因此 LC0 的 `-q` ≡ 我们的 `Q`（都是子节点视角）
- 所以 `FastSign(-q)` ≡ `sign(Q)`，**不是** `sign(-Q)`

`sign(-Q)` 比正确的 `sign(Q)` 多了一个负号，导致快赢被 penalty、慢赢被 bonus，完全反向。

**最终修正**：直接使用 `Q`（而非 `sign(Q)`），既修正了符号，又获得了天然的 Q-gating。

### M 的反向传播

每个节点维护 M 的 running average，与 Q 类似：

```
// 在 backprop 中
float ml = is_terminal ? 0.0f : moves_left;  // 终局 M=0，非终局用 NN 预测值
while (update_idx != -1) {
    node.n_visits++;
    node.Q += (val - node.Q) / node.n_visits;   // Q 的增量更新
    node.M += (ml - node.M) / node.n_visits;     // M 的增量更新
    val = -val * discount;
    ml += 1.0f;   // 父节点比子节点多一步
    update_idx = node.parent;
}
```

关键点：
- `ml += 1.0f`：沿路径向上回传时，每上升一层加 1（父节点离游戏结束更远）
- 终局节点 `ml = 0`（游戏已结束，剩余 0 步）
- 非终局节点用 NN 的 moves_left head 预测值作为初始 ml

### Network.predict() 输出

移除了旧版的 `/42.0` 归一化，直接输出原始 expected_steps：

```python
steps_prob = log_steps.exp()
idx = torch.arange(43, dtype=torch.float32, device=self.device)
expected_steps = (steps_prob * idx).sum(dim=1)

return (log_prob.exp().cpu().numpy(),
        value_base.cpu().view(-1, 1).numpy(),
        expected_steps.cpu().view(-1, 1).numpy())   # 原始步数，不再 /42.0
```

原因：新方案使用 `child_M - parent_M` 的**相对差值**，归一化在 `slope` 参数中隐含处理，无需显式除以最大步数。

---

## 参数说明

| 参数 | 类型 | 默认值 | 推荐值 (Connect4) | 说明 |
|------|------|--------|-------------------|------|
| `mlh_slope` | float | 0.0 | 0.02 ~ 0.03 | M 差值的灵敏度，0 = 禁用 |
| `mlh_cap` | float | 0.2 | 0.10 ~ 0.15 | M_utility 最大幅度上限 |

当 `mlh_slope=0.0` 时 M_utility 恒为零，行为与未启用 MLH 完全一致（向后兼容）。

### 为什么 slope 这么小？

`child_M - parent_M` 的值可以很大。在 Connect4 中：
- 典型 M 差值范围：±5 ~ ±15 步
- slope = 0.03 时，`0.03 × 10 = 0.3`（被 cap 截断为 0.15）
- 最终 M_utility = 0.15 × Q，在 |Q| = 0.5 时约为 ±0.075

对比 PUCT 的 U 项通常在 0.1 ~ 0.5 范围，M_utility 起到适度的辅助调整作用。slope 过大会导致 M 差值主导搜索，破坏正常的 Q + U 平衡。

### 安全值建议

| slope | 效果 |
|-------|------|
| 0.01 | 非常保守，几乎无影响 |
| 0.02 | 轻微倾向快赢/慢输 |
| 0.03 | 适度影响，推荐起始值 |
| 0.05+ | 较激进，可能干扰正常搜索 |

---

## 架构与参数流

```
server.py / client.py / play.py / gui_play.py
    --mlh_slope  --mlh_cap
         │
    config dict / argparse args
         │
    pipeline.py (setattr → self.mlh_slope, self.mlh_cap)
         │
    ┌────┴──────────────────┐
    ▼                       ▼
AlphaZeroPlayer         BatchedAlphaZeroPlayer
(src/player.py)         (src/player.py)
    │                       │
    ▼                       ▼
MCTS_AZ (Python)        BatchedMCTS (src/MCTS_cpp.py)
(src/MCTS.py)               │
  .mlh_slope                 ▼
  .mlh_cap              C++ BatchedMCTS (src/cpp/BatchedMCTS.h)
                             │
                             ▼
                         C++ MCTS (src/cpp/MCTS.h)
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
               simulate()        backprop()
           parent_M 传给         M running avg
           get_ucb()             ml += 1.0f
```

---

## 改动文件详解

### 1. C++ 核心层

#### `src/cpp/MCTSNode.h` — 节点结构

新增 `float M = 0.0f` 字段，存储 moves-left 的 running average。

`get_ucb()` 增加 M_utility 计算：

```cpp
float get_ucb(float c_init, float c_base, float parent_n,
              bool is_root_node, float noise_epsilon, float fpu_value,
              float parent_M, float mlh_slope, float mlh_cap) const {
    // ... q_value, u_score 计算不变 ...

    float m_utility = 0.0f;
    if (mlh_slope > 0.0f && n_visits > 0) {
        float m_diff = M - parent_M;
        m_utility = std::clamp(mlh_slope * m_diff, -mlh_cap, mlh_cap) * Q;
    }
    return q_value + u_score + m_utility;
}
```

#### `src/cpp/MCTS.h` — 搜索引擎

**成员变量**：`mlh_factor/mlh_threshold` → `mlh_slope/mlh_cap`

**simulate()**：从父节点取 `parent_M`，传给 `get_ucb()`：

```cpp
float parent_M = node_pool[curr_idx].M;
for (int action : valids) {
    float score = node_pool[child_idx].get_ucb(
        c_init, c_base, p_n, is_root, noise_epsilon, fpu_value,
        parent_M, mlh_slope, mlh_cap);
    // ...
}
```

**backprop()**：移除旧的 value 修改逻辑，新增 M 的 running average 累积：

```cpp
void backprop(std::span<const float> policy_logits, float value,
              float moves_left, bool is_terminal) {
    // ... 展开逻辑不变 ...

    int32_t update_idx = current_leaf_idx;
    float val = value;
    float ml = is_terminal ? 0.0f : moves_left;
    while (update_idx != -1) {
        node_pool[update_idx].n_visits++;
        node_pool[update_idx].Q += (val - node_pool[update_idx].Q) / node_pool[update_idx].n_visits;
        node_pool[update_idx].M += (ml - node_pool[update_idx].M) / node_pool[update_idx].n_visits;
        val = -val * discount;
        ml += 1.0f;  // 父节点离游戏结束多一步
        update_idx = node_pool[update_idx].parent;
    }
}
```

#### `src/cpp/BatchedMCTS.h` — 批量 MCTS 包装器

构造函数参数：`mlh_slope`, `mlh_cap`

```cpp
void set_mlh_params(float slope, float cap) {
    for (auto &m : mcts_envs) {
        m->mlh_slope = slope;
        m->mlh_cap = cap;
    }
}
```

#### `src/cpp/bindings.cpp` — Pybind11 绑定

构造函数和 `set_mlh_params` 绑定均更新为 `slope`, `cap` 参数名。

### 2. Python Network 层

#### `src/environments/Connect4/Network.py`

`predict()` 返回 3 元组 `(policy, value, expected_steps)`，**不再做 /42.0 归一化**：

```python
expected_steps = (steps_prob * idx).sum(dim=1)  # 原始步数

return (log_prob.exp().cpu().numpy(),
        value_base.cpu().view(-1, 1).numpy(),
        expected_steps.cpu().view(-1, 1).numpy())
```

### 3. Python MCTS 层

#### `src/MCTS.py`

**TreeNode**：新增 `self.M = 0.0`

**update()**：增加 M 的 running average 更新：
```python
def update(self, leaf_value, moves_left=0.0):
    if self.parent:
        self.parent.update(-leaf_value * self.discount, moves_left + 1.0)
    self.n_visits += 1
    self.Q += (leaf_value - self.Q) / self.n_visits
    self.M += (moves_left - self.M) / self.n_visits
```

**PUCT()**：添加 M_utility 项：
```python
m_utility = 0.0
if mlh_slope > 0 and self.n_visits > 0:
    m_diff = self.M - self.parent.M
    m_utility = max(-mlh_cap, min(mlh_cap, mlh_slope * m_diff)) * self.Q
return q_value + self.u + m_utility
```

#### `src/MCTS_cpp.py`

参数透传：`mlh_slope`, `mlh_cap`

### 4. Player / Pipeline / 入口层

#### `src/player.py`

`AlphaZeroPlayer` 和 `BatchedAlphaZeroPlayer` 参数重命名：`mlh_slope`, `mlh_cap`

#### `src/pipeline.py`

- 移除旧 warmup 机制（`_mlh_slope_target`, `_mlh_warmup`, `_mlh_warmup_loss`, `_mlh_activated` 等）
- 直接使用 `mlh_slope` 和 `mlh_cap` 初始化

#### `server.py`

```python
g_mlh.add_argument('--mlh_slope', type=float, default=0.0,
                    help='MLH slope for MCTS (0=disabled)')
g_mlh.add_argument('--mlh_cap', type=float, default=0.2,
                    help='MLH max effect cap')
```

移除旧的 `--mlh_warmup`, `--mlh_warmup_loss` 参数。

#### `client.py`

移除 HTTP header 同步机制（`X-MLH-Slope`），不再需要 warmup 动态调整。

#### `play.py` / `gui_play.py`

参数重命名为 `mlh_slope`, `mlh_cap`。

---

## 移除的功能

### Warmup 机制

旧版维护了一套 warmup 逻辑：训练初期 steps head 不准确时，slope 从 0 缓慢升到目标值，需要通过 HTTP header 同步给 client。

新方案不再需要 warmup，原因：
1. `cap` 参数已经限制了 M_utility 的最大影响
2. 相对差值 `child_M - parent_M` 本身就比绝对值更鲁棒——即使 steps head 预测偏了，同层节点的偏差方向一致，差值仍然有意义
3. `Q` 值的乘法门控提供了额外安全：训练初期 Q 值不极端，M_utility 影响自然较小

移除的参数：`--mlh_warmup`, `--mlh_warmup_loss`
移除的机制：`_update_mlh_slope()`, `X-MLH-Slope` HTTP header 同步

---

## 改动文件汇总

| # | 文件 | 改动类型 |
|---|------|----------|
| 1 | `src/cpp/MCTSNode.h` | 核心：新增 M 字段 + get_ucb 添加 M_utility |
| 2 | `src/cpp/MCTS.h` | 核心：backprop 累积 M + simulate 传 parent_M |
| 3 | `src/cpp/BatchedMCTS.h` | 核心：参数重命名 slope/cap |
| 4 | `src/cpp/bindings.cpp` | 绑定：参数重命名 |
| 5 | `src/environments/Connect4/Network.py` | 网络：移除 /42.0 归一化 |
| 6 | `src/MCTS.py` | Python MCTS：M 字段 + M_utility + update M |
| 7 | `src/MCTS_cpp.py` | 包装：参数重命名 |
| 8 | `src/player.py` | 透传：参数重命名 |
| 9 | `src/pipeline.py` | 移除 warmup + 参数重命名 |
| 10 | `server.py` | 入口：参数重命名 + 移除 warmup 参数 |
| 11 | `client.py` | 入口：移除 header 同步 + 参数重命名 |
| 12 | `play.py` | 入口：参数重命名 |
| 13 | `gui_play.py` | 入口：参数重命名 |

---

## 使用方法

### 编译

```bash
python setup.py build_ext --inplace
```

### 训练（启用 MLH）

Server:
```bash
python server.py --mlh_slope 0.03 --mlh_cap 0.15
```

Client:
```bash
python client.py --mlh_slope 0.03 --mlh_cap 0.15
```

### 人机对弈（启用 MLH）

```bash
python play.py -o --mlh_slope 0.03 --mlh_cap 0.15
```

### 禁用 MLH（默认行为）

不传 `--mlh_slope` 或传 `--mlh_slope 0` 即可，行为与未启用 MLH 完全一致。

---

## 与 LC0 实现的对比

| 方面 | LC0 | 本项目 |
|------|-----|--------|
| 调整位置 | UCB 选择时 | UCB 选择时 ✅（一致） |
| 每节点 M | backprop 累积 running avg | backprop 累积 running avg ✅（一致） |
| moves_left 归一化 | 复杂的自适应公式 | 原始步数（slope 隐含缩放） |
| 符号函数 | `FastSign(-q)` (q 为父视角) | `Q` (子视角，天然 soft gate) |
| Q 门控 | 硬阈值 threshold | Q 幅度自然门控（无需 threshold） |
| 参数数量 | 3 个 (slope, cap, threshold) | 2 个 (slope, cap) |
