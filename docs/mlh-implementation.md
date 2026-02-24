# Lc0 Moves Left Head (MLH) 集成文档

## 背景

### 策略追逐问题 (Policy Chasing)

Connect4 是已解决的博弈，先手有必胜策略。当 AlphaZero 训练到后期，输方的所有走法 Q 值都趋近于 -1，导致 PUCT 选择退化为纯策略先验驱动：

```
UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
```

当所有 `Q(s,a) ≈ -1` 时，`Q` 项无法区分走法，选择完全由 `P(s,a)` 主导。网络学习的策略又被这种搜索反馈强化，形成自我强化的循环——即"策略追逐"。

### 已有方案：lambda_s 混合

之前通过 `lambda_s` 参数在 `predict()` 中将 steps-to-end 预测混入 value：

```python
value = (1 - lambda_s) * value_base - lambda_s * advantage_sign * steps_adjusted
```

问题在于这种混合**始终生效**，无法区分正常局面和 Q 值退化的局面。

### Lc0 MLH 方案

Leela Chess Zero 的解决思路：将 moves_left 作为**独立信号**传给搜索引擎，仅当 `|Q| > threshold` 时才激活——直接解决"所有走法同样输"的问题，不干扰正常搜索。

核心公式：
```
if |value| > threshold:
    activation = (|value| - threshold) / (1 - threshold)
    bonus = mlh_factor * moves_left_norm * activation
    value += bonus   (if losing, 即 value < 0)
    value -= bonus   (if winning, 即 value > 0)
```

效果：
- **输方**：延长对局的走法获得更高价值（"输得慢"比"输得快"好）
- **赢方**：缩短对局的走法获得更高价值（"赢得快"比"赢得慢"好）
- **正常局面** (`|Q| < threshold`)：不受影响

---

## 新增参数

| 参数 | 类型 | 默认值 | 推荐值 (Connect4) | 说明 |
|------|------|--------|-------------------|------|
| `mlh_factor` | float | 0.0 | 0.3 | MLH 调整幅度，0=禁用 |
| `mlh_threshold` | float | 0.85 | 0.85 | 激活阈值，`\|Q\| > threshold` 时才生效 |

当 `mlh_factor=0.0` 时行为与改动前完全一致（向后兼容）。

建议使用 MLH 时设置 `lambda_s=0`，避免 Python 侧和 C++ 侧双重 value 调整。

---

## 架构与参数流

```
server.py / client.py / play.py / gui_play.py
    --mlh_factor  --mlh_threshold
         │
    config dict / argparse args
         │
    pipeline.py (setattr → self.mlh_factor, self.mlh_threshold)
         │
    ┌────┴──────────────────┐
    ▼                       ▼
AlphaZeroPlayer         BatchedAlphaZeroPlayer
(src/player.py)         (src/player.py)
    │                       │
    ▼                       ▼
MCTS_AZ (Python)        BatchedMCTS (src/MCTS_cpp.py)
(src/MCTS.py)               │
  .mlh_factor                ▼
  .mlh_threshold         C++ BatchedMCTS (src/cpp/BatchedMCTS.h)
                             │
                             ▼
                         C++ MCTS::backprop() (src/cpp/MCTS.h)
```

Network.predict() 返回值从 2 元组扩展为 3 元组：
```
(policy, value, moves_left_norm)
```
其中 `moves_left_norm = expected_steps / 42.0`，范围 [0, 1]。

---

## 改动文件详解

### 1. C++ 核心层

#### `src/cpp/MCTS.h`

**构造函数** — 新增 `mlh_factor` 和 `mlh_threshold` 成员：

```cpp
MCTS(float c_i, float c_b, float disc, float a,
     float noise_eps = 0.25f, float fpu_red = 0.4f, bool use_sym = true,
     float mlh_fac = 0.0f, float mlh_thres = 0.85f)
    : /* ... */ mlh_factor(mlh_fac), mlh_threshold(mlh_thres) { }
```

**backprop()** — 签名增加 `float moves_left_norm` 参数，在 backprop 循环前插入 MLH 调整：

```cpp
void backprop(std::span<const float> policy_logits, float value,
              float moves_left_norm, bool is_terminal)
{
    // MLH: 仅在非终局且 mlh_factor > 0 时生效
    if (mlh_factor > 0.0f && !is_terminal)
    {
        float abs_val = std::abs(value);
        if (abs_val > mlh_threshold)
        {
            float activation = (abs_val - mlh_threshold) / (1.0f - mlh_threshold);
            float bonus = mlh_factor * moves_left_norm * activation;
            value += (value < 0.0f ? bonus : -bonus);
        }
    }
    // ... 后续 backprop 逻辑不变
}
```

#### `src/cpp/BatchedMCTS.h`

**构造函数** — 接受 `mlh_factor`, `mlh_threshold` 并传给每个 MCTS 实例。

**set_mlh_params()** — 运行时动态更新 MLH 参数：
```cpp
void set_mlh_params(float factor, float threshold) {
    for (auto &m : trees) {
        m.mlh_factor = factor;
        m.mlh_threshold = threshold;
    }
}
```

**backprop_batch()** — 新增 `const float *moves_left` 参数：
```cpp
void backprop_batch(const float *policy_logits, const float *values,
                    const float *moves_left, const uint8_t *is_term)
```

#### `src/cpp/bindings.cpp`

- 构造函数绑定增加 `mlh_factor`, `mlh_threshold` 两个可选参数
- `backprop_batch` lambda 增加第 4 个 numpy 数组 `moves_left`，含尺寸校验
- 新增 `set_mlh_params(factor, threshold)` 绑定

### 2. Python Network 层

#### `src/environments/Connect4/Network.py`

**predict()** — 始终计算 `moves_left_norm` 并返回 3 元组：

```python
def predict(self, state):
    # ... forward pass ...
    steps_prob = log_steps.exp()
    idx = torch.arange(43, dtype=torch.float32, device=self.device)
    expected_steps = (steps_prob * idx).sum(dim=1)
    moves_left_norm = expected_steps / 42.0  # [0, 1]

    if self.lambda_s > 0:
        # 原有 Python 侧混合逻辑
        steps_adjusted = 2.0 * moves_left_norm - 1.0
        advantage_sign = torch.tanh(5.0 * value_base)
        value = (1 - self.lambda_s) * value_base - self.lambda_s * advantage_sign * steps_adjusted
    else:
        value = value_base

    return (log_prob.exp().cpu().numpy(),
            value.cpu().view(-1, 1).numpy(),
            moves_left_norm.cpu().view(-1, 1).numpy())
```

### 3. Python MCTS 层

#### `src/MCTS_cpp.py` (BatchedMCTS 包装器)

- 构造函数接受并传递 `mlh_factor`, `mlh_threshold`
- `batch_playout()` 解包 3 元组，管理 `moves_left` 数组：
  - 终局 `moves_left[term_mask] = 0.0`
  - 调用 `self.mcts.backprop_batch(probs, values, moves_left, is_term)`
- Cache 存储扩展为 `(prob, value, moves_left)` 三元组
- `refresh_cache()` 同步更新

#### `src/MCTS.py` (单游戏 Python MCTS)

- `MCTS` 和 `MCTS_AZ` 构造函数增加 `mlh_factor=0.0, mlh_threshold=0.85`
- `MCTS_AZ.playout()` 解包 3 元组并在 `node.update()` 前应用 MLH：

```python
probs, value, moves_left = self.policy.predict(state)
# ... 现有逻辑 ...
leaf_value = value.flatten()[0]
if self.mlh_factor > 0 and not env.done():
    ml = moves_left.flatten()[0]
    abs_v = abs(leaf_value)
    if abs_v > self.mlh_threshold:
        activation = (abs_v - self.mlh_threshold) / (1.0 - self.mlh_threshold)
        bonus = self.mlh_factor * ml * activation
        leaf_value += bonus if leaf_value < 0 else -bonus
```

### 4. Player 层

#### `src/player.py`

- `AlphaZeroPlayer.__init__()` 增加 `mlh_factor=0.0, mlh_threshold=0.85`，传给 `MCTS_AZ`
- `BatchedAlphaZeroPlayer.__init__()` 增加同样参数，传给 `BatchedMCTS`

### 5. 入口层

#### `server.py`

```python
parser.add_argument('--mlh_factor', type=float, default=0.0,
                    help='Moves Left Head factor (0=disabled, recommended 0.2-0.3)')
parser.add_argument('--mlh_threshold', type=float, default=0.85,
                    help='MLH activation threshold')
```
参数加入 config dict，通过 `pipeline.py` 的 `setattr` 机制传递。

#### `client.py`

同上 argparse 参数，传给 `BatchedAlphaZeroPlayer`。

#### `play.py`

同上 argparse 参数，传给 `AlphaZeroPlayer`。

#### `gui_play.py`

新增模块级常量：
```python
MLH_FACTOR = 0.0
MLH_THRESHOLD = 0.85
```
传给 `AlphaZeroPlayer`。

### 6. Pipeline 层

#### `src/pipeline.py`

- `AlphaZeroPlayer` 创建处使用 `getattr(self, 'mlh_factor', 0.0)`
- `_batched_eval_games()` 中评估用 `BatchedMCTS` 也传入 MLH 参数

### 7. 兼容性修复

以下文件的 `predict()` 调用从 2 元组解包改为 3 元组：

| 文件 | 改动 |
|------|------|
| `src/Cache.py` | `LRUCache.refresh()` 和 `LFUCache.refresh()` 解包并存储 3 元组 |
| `tools/diagnose_mcts.py` | `predict()` 解包修复；`RolloutAdapter.predict()` 返回 `moves_left=0.5` |
| `tools/mcts_worker.py` | `RolloutAdapter.predict()` 返回 `moves_left=0.5` |
| `GradCAM.py` | `pred, value, _ = model.predict(board)` |
| `src/environments/Connect4/utils.py` | `probs0, value0, _ = net.predict(state0)` |

---

## 改动文件汇总 (17 个文件)

| # | 文件 | 改动类型 |
|---|------|----------|
| 1 | `src/cpp/MCTS.h` | 核心：新增 MLH 字段 + backprop 调整逻辑 |
| 2 | `src/cpp/BatchedMCTS.h` | 核心：set_mlh_params + backprop_batch 传参 |
| 3 | `src/cpp/bindings.cpp` | 绑定：更新构造函数 + backprop_batch + set_mlh_params |
| 4 | `src/environments/Connect4/Network.py` | 网络：predict() 返回 3 元组 |
| 5 | `src/MCTS_cpp.py` | 包装：传 moves_left + cache 3 元组 |
| 6 | `src/MCTS.py` | Python MCTS：MLH 调整逻辑 |
| 7 | `src/player.py` | 透传：两个 Player 类参数透传 |
| 8 | `src/pipeline.py` | 透传：创建 player + 评估时传入 |
| 9 | `server.py` | 入口：argparse + config |
| 10 | `client.py` | 入口：argparse + Actor |
| 11 | `play.py` | 入口：argparse + 传参 |
| 12 | `gui_play.py` | 入口：常量 + 传参 |
| 13 | `src/Cache.py` | 兼容：refresh() 3 元组解包 |
| 14 | `tools/diagnose_mcts.py` | 兼容：predict 解包 + RolloutAdapter |
| 15 | `tools/mcts_worker.py` | 兼容：RolloutAdapter 返回 3 元组 |
| 16 | `GradCAM.py` | 兼容：predict 解包 |
| 17 | `src/environments/Connect4/utils.py` | 兼容：predict 解包 |

---

## 使用方法

### 编译

```bash
python setup.py build_ext --inplace
```

### 人机对弈 (启用 MLH)

```bash
python play.py -o --mlh_factor 0.3 --mlh_threshold 0.85 --lambda_s 0
```

### 分布式训练 (启用 MLH)

Server:
```bash
python server.py --mlh_factor 0.3 --mlh_threshold 0.85 --lambda_s 0
```

Client:
```bash
python client.py --mlh_factor 0.3 --mlh_threshold 0.85 --lambda_s 0
```

### 禁用 MLH (默认行为)

不传 `--mlh_factor` 或传 `--mlh_factor 0` 即可，行为与改动前完全一致。

---

## 推荐参数 (Connect4)

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `mlh_factor` | 0.3 | Connect4 最多42步，比国际象棋的 0.035 大 |
| `mlh_threshold` | 0.85 | 仅在 Q 值极端时激活 |
| `lambda_s` | 0.0 | 使用 MLH 时关闭 Python 侧混合，避免双重调整 |

---

## 与 Lc0 实现的差异

| 方面 | Lc0 | 本项目 |
|------|-----|--------|
| 调整位置 | UCB 选择时 (Q + M 项) | backprop 时调整 value |
| 每节点 M 值 | 通过 backprop 平均维护 | 仅在叶节点一次性调整 |
| moves_left 归一化 | `m / (a * max(1, sqrt(n)))` | `expected_steps / 42.0` |
| 棋盘规模适配 | 复杂的自适应公式 | 简单除以最大步数 |

选择在 backprop 而非 UCB 中调整的原因：避免每个节点额外存储 moves_left 均值，实现更简洁；对 Connect4 这种小型博弈，效果差异可忽略。
