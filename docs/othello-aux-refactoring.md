# Othello Auxiliary Head 机制重构文档

## 概述

本次改动将 Othello 的辅助头（auxiliary head）从复用 Connect4 的 MLH（Moves Left Head）机制，改为借鉴 KataGo 的 score utility 机制。核心变化：**每个游戏定义自己的 `compute_aux_utility()` 和 `terminal_aux()`**，C++ MCTS 通过模板编译期分派。

---

## 1. 问题分析

### 1.1 修改前的共享 MLH 逻辑（MCTS.h select_edge 内联）

```cpp
// ── 修改前：所有游戏共享同一套 MLH 逻辑 ──
float m_utility = 0.0f;
if (config_->mlh_slope > 0.0f &&
    e.child != -1 && pool_.node(e.child).n_visits > 0)
{
    float child_M = pool_.node(e.child).mean_M();
    if constexpr (Game::AUX_NEGATE_PER_PLY)
        child_M = -child_M;
    float m_diff = child_M - parent_M;
    m_utility = std::clamp(config_->mlh_slope * m_diff,
                           -config_->mlh_cap, config_->mlh_cap);

    // mlh_threshold 门控：|Q| 低于阈值时禁用
    if (std::abs(child_Q) < config_->mlh_threshold)
        m_utility = 0.0f;
    else
        m_utility *= child_Q;
}
```

### 1.2 三个语义错误

**错误 1：Q-gating 对棋差无意义**

MLH 里乘 `child_Q` 是为了判断方向——"赢着要快赢（减步数），输着要慢输（加步数）"，Q 的符号决定方向。但棋差本身有内在方向：正=对当前玩家有利，不需要外部信号判断加减。

更严重的是，`mlh_threshold` 让 Q 接近零时 utility 完全消失。但这恰恰是棋差 utility 最该发挥作用的场景——胜率分不出高下时，用棋差打破僵局：

```
Q = 0.01（接近平局）
  修改前: |Q| < threshold → m_utility = 0   ← 最需要棋差时反而禁用
  修改后: utility = factor × child_M          ← 正常工作
```

**错误 2：m_diff 对棋差无清晰语义**

`child_M - parent_M` 在 MLH 里表示"这步棋让剩余步数变化了多少"，有明确含义。但对棋差，"子节点期望棋差减父节点期望棋差"没有战略意义——棋差是终局属性，不是逐步递减的量。

**错误 3：丢弃不确定性信息**

NN 辅助头输出 129 类分布（对应棋差 [-64, +64]），但只取 `E[x]`，丢掉分布形状。高确信的 "+10" 和方差巨大的 "+10" 产生完全相同的 utility。

---

## 2. 修改方案

### 2.1 设计原则

- 每个游戏定义自己的 `compute_aux_utility()` 静态方法和 `terminal_aux()` 成员方法
- C++ MCTS 通过模板 `Game::compute_aux_utility()` 编译期分派，零运行时开销
- Othello 采用 KataGo 风格的 atan 映射 score utility
- Connect4 保留原有 MLH 机制不变
- NN 端计算 `E[atan(x/scale)]` 而非 `atan(E[x])`，自然引入不确定性衰减

### 2.2 Othello vs Connect4 对比

| | Connect4 (MLH) | Othello (Score Utility) |
|---|---|---|
| 辅助头预测内容 | 剩余步数 | 终局棋差 |
| 方向语义 | 无内在方向，需 Q 判断 | 正=有利，方向内在 |
| utility 公式 | `clamp(slope × Δm) × Q` | `factor × child_M` |
| 终局值 | `0.0`（游戏结束，剩余 0 步） | `(2/π) × atan(diff / scale)` |
| 逐层变换 | `+1`（步数每层加 1） | 取反（视角翻转） |
| NN 输出 | 原始 `E[x]` | `E[atan(x/scale)] × (2/π)` |
| 不确定性处理 | 无 | Jensen 不等式自动衰减 |

---

## 3. 代码变化详解

### 3.1 SearchConfig（MCTSNode.h）

```cpp
// ── 修改前 ──
float mlh_slope = 0.0f;
float mlh_cap = 0.2f;
float mlh_threshold = 0.5f;   // Q 门控阈值
float value_decay = 1.0f;

// ── 修改后 ──
float mlh_slope = 0.0f;           // Connect4 MLH 斜率（0=禁用）
float mlh_cap = 0.2f;             // Connect4 MLH 最大影响上限
float score_utility_factor = 0.0f; // Othello score utility 权重（0=禁用）
float score_scale = 8.0f;          // Othello atan 映射缩放分母
float value_decay = 1.0f;
```

移除 `mlh_threshold`，新增 `score_utility_factor` 和 `score_scale`。

### 3.2 Othello::compute_aux_utility（Othello.h）

```cpp
// ── 修改前：不存在，所有游戏共享 MCTS.h 里的内联逻辑 ──

// ── 修改后：Othello 定义自己的 compute_aux_utility ──
[[nodiscard]] static float compute_aux_utility(
    float child_M, float /*parent_M*/, float /*child_Q*/, const SearchConfig& cfg)
{
    if (cfg.score_utility_factor <= 0.0f) return 0.0f;
    return cfg.score_utility_factor * child_M;
}
```

- 不用 `parent_M`：不做父子差分，直接用子节点的绝对 utility
- 不用 `child_Q`：棋差方向内在于 M 的符号
- `child_M` 已是 atan 映射后的 utility（范围 [-1, 1]），乘以权重即可

### 3.3 Othello::terminal_aux（Othello.h）

```cpp
// ── 修改前 ──
[[nodiscard]] float terminal_aux() const
{
    return 0.0f;  // Othello 原本没有实现终局 aux
}

// ── 修改后 ──
[[nodiscard]] float terminal_aux(const SearchConfig& cfg) const
{
    int diff = std::popcount(bb[0]) - std::popcount(bb[1]);
    float raw = static_cast<float>(diff * turn);  // 当前玩家视角
    return std::atan(raw / cfg.score_scale) * (2.0f / 3.14159265f);
}
```

终局有确切棋差，用与 NN 端相同的 atan 映射，保证搜索树内语义一致。

### 3.4 Connect4::compute_aux_utility（Connect4.h）

```cpp
// ── 修改后：Connect4 保留原有 MLH 语义 ──
[[nodiscard]] static float compute_aux_utility(
    float child_M, float parent_M, float child_Q, const SearchConfig& cfg)
{
    if (cfg.mlh_slope <= 0.0f) return 0.0f;
    float m_diff = child_M - parent_M;
    float m_utility = std::clamp(cfg.mlh_slope * m_diff,
                                  -cfg.mlh_cap, cfg.mlh_cap);
    return m_utility * child_Q;  // Q 符号决定方向：赢快输慢
}
```

### 3.5 Connect4::terminal_aux（Connect4.h）

```cpp
// ── 修改后：签名统一，但 Connect4 终局 aux=0 ──
[[nodiscard]] float terminal_aux(const SearchConfig& /*cfg*/) const
{
    return 0.0f;  // 游戏结束，剩余 0 步
}
```

### 3.6 MCTS::select_edge（MCTS.h）

```cpp
// ── 修改前：14 行内联 MLH 逻辑（含 threshold 门控）──
float m_utility = 0.0f;
if (config_->mlh_slope > 0.0f &&
    e.child != -1 && pool_.node(e.child).n_visits > 0)
{
    float child_M = pool_.node(e.child).mean_M();
    if constexpr (Game::AUX_NEGATE_PER_PLY)
        child_M = -child_M;
    float m_diff = child_M - parent_M;
    m_utility = std::clamp(config_->mlh_slope * m_diff,
                           -config_->mlh_cap, config_->mlh_cap);
    if (std::abs(child_Q) < config_->mlh_threshold)
        m_utility = 0.0f;
    else
        m_utility *= child_Q;
}

// ── 修改后：5 行，委托给 Game::compute_aux_utility ──
float m_utility = 0.0f;
if (e.child != -1 && pool_.node(e.child).n_visits > 0)
{
    m_utility = Game::compute_aux_utility(
        child_M, parent_M, child_Q, *config_);
}
```

编译期通过模板 `Game` 分派到对应游戏的实现，零运行时开销。

### 3.7 MCTS::backprop / backprop_vl（MCTS.h）

```cpp
// ── 修改前 ──
propagate(wdl, is_terminal ? sim_env.terminal_aux() : moves_left);

// ── 修改后：terminal_aux 需要 config 访问 score_scale ──
propagate(wdl, is_terminal ? sim_env.terminal_aux(*config_) : moves_left);
```

### 3.8 Othello Network.predict()（Network.py）

```python
# ── 修改前：直接取期望棋差 ──
steps_prob = log_steps.exp()
idx = torch.arange(129, dtype=torch.float32, device=self.device)
expected_steps = (steps_prob * idx).sum(dim=1) - 64.0  # E[x]

return (log_prob.exp().cpu().numpy(),
        wdl.cpu().numpy(),
        expected_steps.cpu().view(-1, 1).numpy())

# ── 修改后：计算 E[atan(x/scale)] × (2/π) ──
steps_prob = log_steps.exp()
idx = torch.arange(log_steps.shape[-1], dtype=torch.float32, device=self.device)
disc_diff = idx - float(self.aux_target_offset)   # [-64, +64]
score_scale = getattr(self, 'score_scale', 8.0)
atan_vals = torch.atan(disc_diff / score_scale) * (2.0 / math.pi)
expected_utility = (steps_prob * atan_vals).sum(dim=1)  # E[atan(x/s)] × (2/π)

return (log_prob.exp().cpu().numpy(),
        wdl.cpu().numpy(),
        expected_utility.cpu().view(-1, 1).numpy())
```

---

## 4. E[atan(x/s)] vs atan(E[x])：不确定性衰减

这是本次改动最关键的设计选择。两种做法的数学差异：

### 4.1 Jensen 不等式

atan 是凹函数，由 Jensen 不等式：`E[f(X)] ≤ f(E[X])`，即 `E[atan(x)] ≤ atan(E[x])`。

等号成立当且仅当分布退化为点分布（方差=0）。分布越分散，差距越大。

### 4.2 数值示例

**情况 A：高确信**
```
分布: P(+10) = 1.0     → E[x] = +10
atan(E[x] / 8) × (2/π) = atan(1.25) × 0.637 = 0.57
E[atan(x/8)] × (2/π)   = atan(1.25) × 0.637 = 0.57   ← 相等
```

**情况 B：低确信（同均值 +10）**
```
分布: P(-20) = 0.5, P(+40) = 0.5     → E[x] = +10
atan(E[x] / 8) × (2/π) = 0.57                          ← 不变
E[atan(x/8)] × (2/π)   = 0.5 × atan(-2.5) × 0.637
                        + 0.5 × atan(5.0) × 0.637
                        = 0.5 × (-0.76) + 0.5 × 0.91
                        = 0.08                           ← 大幅衰减
```

同样均值 +10，不确信时 utility 从 0.57 降到 0.08。

### 4.3 为什么这是正确的行为

- 辅助头训练不充分时，输出分布为高熵（接近均匀），utility 自动趋近零，不会干扰搜索
- 局面模糊时（可能大赢也可能大输），保守地给出低 utility，让 WDL value 主导
- 局面明确时（分布集中），utility 接近 atan(E[x])，正常发挥作用
- 不需要任何阈值参数或 warmup 机制

---

## 5. atan 映射的直觉

`score_scale` 控制曲线形状：

```
score_scale = 8 时:

disc_diff    atan utility
  ±2         ±0.15       差距小，utility 温和增长
  ±4         ±0.30
  ±8         ±0.50       1 个 scale 的差距，utility = 0.5
  ±16        ±0.76       接近饱和
  ±32        ±0.90
  ±64        ±0.96       碾压局，几乎封顶
```

效果：赢 64 子和赢 32 子的 utility 差距很小（0.06），但赢 4 子和赢 8 子的差距明显（0.20）。大比分差异对搜索影响递减，避免极端棋差支配 UCB。

---

## 6. AUX_NEGATE_PER_PLY 的作用

Othello 设置 `AUX_NEGATE_PER_PLY = true`，Connect4 设置 `AUX_PLUS_ONE_PER_PLY = true`。

backprop 时 M 值沿路径传播的变换：

```cpp
// propagate() 中
if constexpr (Game::AUX_PLUS_ONE_PER_PLY)
    ml += 1.0f;    // Connect4: 父节点离终局多一步
if constexpr (Game::AUX_NEGATE_PER_PLY)
    ml = -ml;      // Othello: 翻转视角
```

**Connect4**：终局 M=0，每上一层 +1。父节点的 M 比子节点大 1。`m_diff = child_M - parent_M` 为负值（走一步后更接近终局），slope 为正时鼓励快赢。

**Othello**：每上一层取反。如果叶节点预测"当前玩家棋差 +10"，那么父节点（对手视角）存储 -10。`select_edge` 中再取反（`child_M = -child_M`），得到对 parent 的"我选这步棋，对手会得到棋差 +10，即我会得到 -10"——child_M 为 -10，utility 为负，不偏好这个分支。反之，如果子节点对手棋差为 -10（即我赢），child_M 取反后为 +10，utility 为正，偏好这个分支。

---

## 7. score_scale 同步机制

`score_scale` 必须在 C++ 终局回传和 Python NN 预测之间保持一致。

```
SearchConfig.score_scale ──(sync)──► CNN.score_scale
       │                                    │
       ▼                                    ▼
  terminal_aux():                      predict():
  atan(diff/scale)×(2/π)              E[atan(x/scale)]×(2/π)
       │                                    │
       └──────── 必须用相同 scale ──────────┘
```

同步点在 `MCTS_cpp.py` 的 `batch_playout()` 入口：

```python
# batch_playout 开头
if hasattr(pv_func, 'score_scale'):
    pv_func.score_scale = self.mcts.config.score_scale
```

Othello CNN 定义 `score_scale = 8.0` 类属性使 `hasattr` 为真。Connect4 CNN 无此属性，跳过同步。

运行时修改 `score_scale` 后，缓存中旧 scale 的 aux 值失效，自动清空：

```python
def set_score_utility_params(self, factor, scale):
    cfg = self.mcts.config
    old_scale = cfg.score_scale
    cfg.score_utility_factor = factor
    cfg.score_scale = scale
    if scale != old_scale and self.cache is not None and len(self.cache) > 0:
        self.cache._od.clear()
```

---

## 8. 参数说明

### Othello 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `score_utility_factor` | 0.0 | score utility 权重，0=禁用 |
| `score_scale` | 8.0 | atan 映射缩放分母 |

`score_utility_factor` 推荐范围 0.05 ~ 0.30，控制棋差 utility 相对于 Q + U 的影响力。`score_scale` 控制棋差到 utility 的映射曲线陡度，越小则小棋差的影响越大。

### Connect4 参数（不变）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mlh_slope` | 0.0 | MLH 斜率，0=禁用 |
| `mlh_cap` | 0.2 | MLH 最大影响上限 |

---

## 9. 改动文件汇总

| 文件 | 改动 |
|------|------|
| `src/cpp/MCTSNode.h` | SearchConfig: 移除 `mlh_threshold`，新增 `score_utility_factor`、`score_scale` |
| `src/cpp/Othello.h` | 新增 `compute_aux_utility()`、`terminal_aux(cfg)` |
| `src/cpp/Connect4.h` | 新增 `compute_aux_utility()`、`terminal_aux(cfg)` 签名统一 |
| `src/cpp/MCTS.h` | `select_edge` 委托给 `Game::compute_aux_utility()`；backprop 传 config |
| `src/cpp/mcts_bindings.cpp` | 绑定新参数，移除 `mlh_threshold` |
| `src/environments/Othello/Network.py` | `predict()` 输出 `E[atan(x/s)]×(2/π)`；新增 `score_scale` 类属性 |
| `src/MCTS_cpp.py` | 同步 `score_scale` 到网络；scale 变更时清空缓存 |
| `src/player.py` | 透传 `score_utility_factor`、`score_scale` |
| `src/pipeline.py` | 构造参数更新 |
| `server.py` | CLI 参数、config dict、热更新处理 |
| `client.py` | CLI 参数、服务器同步 |
| `play.py` | CLI 参数 |
| `gui_play.py` | 参数面板更新 |
| `gui_play_othello.py` | 标题 FINAL DIFF → SCORE UTIL，DiffBar 范围 [-1,1]，标注单位 |
| `static/dashboard.html` | 参数组名 MLH → Aux Utility |
