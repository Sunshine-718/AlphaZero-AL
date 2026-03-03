# AlphaZero-AL C++/Python 接口文档

本文档描述 C++ MCTS 引擎暴露给 Python 的完整接口，涵盖 `mcts_cpp` 和 `env_cpp` 两个 pybind11 模块，以及 Python 封装层 `MCTS_cpp.py`。

---

## 目录

- [模块总览](#模块总览)
- [mcts\_cpp 模块](#mcts_cpp-模块)
  - [BatchedMCTS\_\<Game\>](#batchedmctsgame)
  - [IEvaluator\_\<Game\>](#ievaluatorgame)
  - [RolloutEvaluator\_\<Game\>](#rolloutevaluatorgame)
- [env\_cpp 模块](#env_cpp-模块)
- [Python 封装层 — BatchedMCTS](#python-封装层--batchedmcts)
- [数据流](#数据流)
- [C++ 内部类型参考](#c-内部类型参考)
  - [WDLValue](#wdlvalue)
  - [MCTSNode](#mctsnode)
  - [SimResult](#simresult)
  - [MCTSGame Concept](#mctsgame-concept)

---

## 模块总览

| 模块 | 描述 | 注册的类 |
|------|------|----------|
| `mcts_cpp` | 批量 MCTS 搜索引擎 | `BatchedMCTS_Connect4`, `BatchedMCTS_Othello`, `IEvaluator_Connect4`, `IEvaluator_Othello`, `RolloutEvaluator_Connect4`, `RolloutEvaluator_Othello` |
| `env_cpp` | 游戏环境（棋盘逻辑） | `env_cpp.connect4.Env`, `env_cpp.othello.Env`, `env_cpp.gomoku.Env` |

每个游戏通过模板函数 `register_batched_mcts<Game>()` 一次性注册 3 个类。所有 C++ 计算阶段（`search_batch`、`backprop_batch`、`search`）执行前释放 GIL，允许 Python 线程并行运行。

---

## mcts_cpp 模块

### BatchedMCTS_\<Game\>

管理 N 个独立 MCTS 搜索树的并行容器，OpenMP 并行化。

#### 构造函数

```python
BatchedMCTS_Connect4(
    n_envs: int,                # 并行环境数量
    c_init: float,              # PUCT 初始常数
    c_base: float,              # PUCT 对数基数
    alpha: float,               # Dirichlet 噪声 alpha (<=0 禁用)
    noise_epsilon: float = 0.25,# 噪声混合权重 ε
    fpu_reduction: float = 0.4, # First Play Urgency 衰减系数
    use_symmetry: bool = True,  # 是否启用随机对称增广
    mlh_slope: float = 0.0,     # Moves Left Head 斜率
    mlh_cap: float = 0.2,      # MLH 影响上限
    mlh_threshold: float = 0.8, # MLH Q 阈值
    value_decay: float = 1.0    # 反向传播逐层衰减 (1.0=禁用)
)
```

#### 参数设置

| 方法 | 参数 | 说明 |
|------|------|------|
| `set_seed(seed: int)` | 随机种子 | 设置所有 OpenMP 线程的 RNG 种子 |
| `set_noise_epsilon(eps: float)` | ε ∈ [0,1] | Dirichlet 噪声混合权重 |
| `set_mlh_params(slope, cap, threshold)` | 三个 float | Moves Left Head 超参 |
| `set_c_init(val: float)` | c_init | PUCT 初始常数 |
| `set_c_base(val: float)` | c_base | PUCT 对数基数 |
| `set_alpha(val: float)` | alpha | Dirichlet alpha |
| `set_fpu_reduction(val: float)` | fpu_red | FPU 衰减系数 |
| `set_use_symmetry(val: bool)` | 开关 | 对称增广开关 |
| `set_value_decay(val: float)` | γ ∈ (0,1] | 反向传播衰减 (1.0=禁用) |

#### 树管理

| 方法 | 说明 |
|------|------|
| `reset_env(env_idx: int)` | 重置指定环境的搜索树 |
| `get_num_envs() -> int` | 返回并行环境数 |
| `prune_roots(actions: ndarray[int32])` | 批量树剪枝：将各环境选中动作的子树提升为新根 |

#### AlphaZero 搜索（Split API：NN 评估）

用于 AlphaZero 训练/推理，搜索流程拆分为 selection → Python NN → backprop 三步，避免 GIL 频繁切换。

##### search_batch

```python
search_batch(
    input_boards: ndarray[int8],   # shape (n_envs, *board_shape)
    turns: ndarray[int32]          # shape (n_envs,)，1 或 -1
) -> Tuple[
    leaf_boards: ndarray[int8],    # shape (n_envs, *board_shape)  叶节点棋盘
    term_d: ndarray[float32],      # shape (n_envs,)  终局和棋率
    term_p1w: ndarray[float32],    # shape (n_envs,)  终局 P1 胜率
    term_p2w: ndarray[float32],    # shape (n_envs,)  终局 P2 胜率
    is_terminal: ndarray[uint8],   # shape (n_envs,)  是否终局
    leaf_turns: ndarray[int32]     # shape (n_envs,)  叶节点落子方
]
```

Selection 阶段：从各环境根节点沿搜索树选择叶节点。对非终局叶节点可选应用随机对称变换（`use_symmetry=True` 时）。返回叶节点棋盘供 Python 端调用 NN。

##### backprop_batch

```python
backprop_batch(
    policy_logits: ndarray[float32],  # shape (n_envs, action_size)  策略 logits
    d_vals: ndarray[float32],         # shape (n_envs,)  和棋率（绝对视角）
    p1w_vals: ndarray[float32],       # shape (n_envs,)  P1 胜率（绝对视角）
    p2w_vals: ndarray[float32],       # shape (n_envs,)  P2 胜率（绝对视角）
    moves_left: ndarray[float32],     # shape (n_envs,)  预期剩余步数
    is_term: ndarray[uint8]           # shape (n_envs,)  是否终局
) -> None
```

Backpropagation 阶段：用 NN 评估结果展开叶节点并沿路径反向传播。如果 `use_symmetry` 启用，会在展开前自动对 policy 做对称逆变换。

#### 通用搜索入口（IEvaluator）

```python
search(
    evaluator: IEvaluator,         # C++ 评估器实例
    input_boards: ndarray[int8],   # shape (n_envs, *board_shape)
    turns: ndarray[int32],         # shape (n_envs,)
    n_playout: int                 # 模拟次数
) -> None
```

整个 playout 循环在 C++ 内完成：并行 selection → 批量评估（`evaluator.evaluate_batch()`）→ 并行 backprop → 重复 `n_playout` 次。对于纯 MCTS 基线，传入 `RolloutEvaluator` 即可。

#### 统计查询

##### get_all_counts

```python
get_all_counts() -> List[int]
# flat 向量，长度 n_envs × action_size
# counts[i * action_size + a] = 环境 i 动作 a 的访问次数
```

##### get_all_root_stats

```python
get_all_root_stats() -> ndarray[float32]
# shape (n_envs, 6 + action_size * 8)
```

每行布局：

| 偏移 | 字段 | 说明 |
|------|------|------|
| 0 | `root_N` | 根节点访问次数 |
| 1 | `root_Q` | 根节点 Q 值（当前落子方视角） |
| 2 | `root_M` | 根节点预期剩余步数 |
| 3 | `root_D` | 根节点和棋率（绝对） |
| 4 | `root_P1W` | 根节点 P1 胜率（绝对） |
| 5 | `root_P2W` | 根节点 P2 胜率（绝对） |
| 6 + a*8 + 0 | `a_N` | 动作 a 子节点访问次数 |
| 6 + a*8 + 1 | `a_Q` | 动作 a 子节点 Q 值 |
| 6 + a*8 + 2 | `a_prior` | 动作 a NN 先验概率 |
| 6 + a*8 + 3 | `a_noise` | 动作 a Dirichlet 噪声 |
| 6 + a*8 + 4 | `a_M` | 动作 a 预期剩余步数 |
| 6 + a*8 + 5 | `a_D` | 动作 a 和棋率（绝对） |
| 6 + a*8 + 6 | `a_P1W` | 动作 a P1 胜率（绝对） |
| 6 + a*8 + 7 | `a_P2W` | 动作 a P2 胜率（绝对） |

#### 类级常量属性

| 属性 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `action_size` | `int` | 动作空间大小 | Connect4=7, Othello=65 |
| `board_size` | `int` | 棋盘 flat 字节数 | Connect4=42, Othello=64 |
| `board_shape` | `tuple` | 棋盘形状 | Connect4=(6,7), Othello=(8,8) |

---

### IEvaluator_\<Game\>

叶节点评估器抽象基类。不可直接实例化，仅作为 `search()` 的参数类型。

子类需实现 `evaluate_batch()` 或 `evaluate_single()` 之一：
- **`evaluate_batch(states, turns, results)`** — 批量评估（默认实现：OpenMP 并行调用 `evaluate_single()`）
- **`evaluate_single(state, turn) -> Result`** — 单个评估（默认返回均匀 policy + 均匀 WDL）

评估结果 `Result` 包含：
- `policy`: `array<float, ACTION_SIZE>` — 策略概率（无需归一化，展开时自动归一化）
- `wdl`: `WDLValue` — 绝对视角 WDL (draw, p1_win, p2_win)
- `moves_left`: `float` — 预期剩余步数

---

### RolloutEvaluator_\<Game\>

继承 `IEvaluator`，随机 rollout 评估器。

```python
evaluator = mcts_cpp.RolloutEvaluator_Connect4()  # 默认构造，无参数
```

行为：对叶节点随机落子直到终局，返回均匀 policy + 终局 WDL（绝对视角）。

---

## env_cpp 模块

各游戏环境注册为子模块：

```python
import env_cpp

env = env_cpp.connect4.Env()  # Connect4 环境
env = env_cpp.othello.Env()   # Othello 环境
env = env_cpp.gomoku.Env()    # Gomoku 环境
```

> **注意**：Gomoku 有 env 绑定但尚未注册 `BatchedMCTS_Gomoku`。

---

## Python 封装层 — BatchedMCTS

`src/MCTS_cpp.py` 中的 `BatchedMCTS` 类封装了底层 C++ 接口，提供更友好的 Python API。

### 构造函数

```python
from src.MCTS_cpp import BatchedMCTS

mcts = BatchedMCTS(
    batch_size=100,             # 并行环境数
    c_init=1.25,                # PUCT c_init
    c_base=19652.0,             # PUCT c_base
    alpha=0.3,                  # Dirichlet alpha
    n_playout=800,              # 默认模拟次数
    game_name='Connect4',       # 游戏名（从 _BACKENDS 查找 C++ 类）
    board_converter=None,       # 自定义棋盘转换函数，默认 3-plane 相对视角
    cache_size=0,               # LRU 置换表容量（0=禁用）
    noise_epsilon=0.25,         # Dirichlet 噪声权重
    fpu_reduction=0.4,          # FPU 衰减
    use_symmetry=True,          # 对称增广
    mlh_slope=0.0,              # MLH 斜率
    mlh_cap=0.2,                # MLH 上限
    mlh_threshold=0.8,          # MLH Q 阈值
    value_decay=1.0             # 反向传播衰减
)
```

### 支持的游戏

```python
_BACKENDS = {
    'Connect4': mcts_cpp.BatchedMCTS_Connect4,
    'Othello':  mcts_cpp.BatchedMCTS_Othello,
}
```

### 核心方法

#### batch_playout — AlphaZero NN 搜索

```python
mcts.batch_playout(
    pv_func,                    # 需实现 .predict(states) 接口的 NN
    current_boards: ndarray,    # shape (batch_size, *board_shape), int8, X=1/O=-1
    turns: ndarray,             # shape (batch_size,), 1 或 -1
    n_playout: int = None       # 覆盖默认模拟次数
) -> BatchedMCTS  # 返回 self，支持链式调用
```

**`pv_func.predict()` 接口要求**：

```python
def predict(states: ndarray[float32]) -> Tuple[
    probs: ndarray[float32],    # shape (N, action_size)  策略概率
    wdl: ndarray[float32],      # shape (N, 3)  相对视角 WDL [draw, win, loss]
    moves_left: ndarray[float32] # shape (N, 1)  预期剩余步数
]:
```

输入 `states` 为 3-plane 相对视角棋盘，shape `(N, 3, *board_shape)`：
- Plane 0: 当前落子方棋子 = 1.0
- Plane 1: 对手棋子 = 1.0
- Plane 2: 回合指示器（全 1.0 或全 -1.0）

**置换表行为**：`cache_size > 0` 时启用 LRU 置换表。Cache hit 直接使用缓存结果，miss 则批量送 NN 后写入缓存。

#### rollout_playout — 纯 MCTS 基线

```python
mcts.rollout_playout(
    current_boards: ndarray,    # shape (batch_size, *board_shape)
    turns: ndarray              # shape (batch_size,)
) -> BatchedMCTS
```

内部创建 `RolloutEvaluator` 并调用 C++ `search()` 方法，整个搜索循环在 C++ 内完成。

#### refresh_cache — 刷新置换表

```python
mcts.refresh_cache(pv_func) -> BatchedMCTS
```

网络权重更新后调用，用新 NN 重新计算缓存中所有条目。

### 统计查询

| 方法 | 返回 | 说明 |
|------|------|------|
| `get_visits_count()` | `ndarray (B, A)` int | 各动作访问次数 |
| `get_mcts_probs()` | `ndarray (B, A)` float | 归一化访问概率 |
| `get_root_stats()` | `dict` | 完整根节点统计（见下表） |

#### get_root_stats 返回字段

| 键 | Shape | 说明 |
|----|-------|------|
| `root_N` | `(B,)` | 根节点总访问次数 |
| `root_Q` | `(B,)` | 根节点 Q 值 |
| `root_M` | `(B,)` | 根节点预期剩余步数 |
| `root_D` | `(B,)` | 根节点和棋率 |
| `root_P1W` | `(B,)` | 根节点 P1 胜率 |
| `root_P2W` | `(B,)` | 根节点 P2 胜率 |
| `N` | `(B, A)` | 各动作访问次数 |
| `Q` | `(B, A)` | 各动作 Q 值 |
| `prior` | `(B, A)` | 各动作 NN 先验概率 |
| `noise` | `(B, A)` | 各动作 Dirichlet 噪声 |
| `M` | `(B, A)` | 各动作预期剩余步数 |
| `D` | `(B, A)` | 各动作和棋率 |
| `P1W` | `(B, A)` | 各动作 P1 胜率 |
| `P2W` | `(B, A)` | 各动作 P2 胜率 |

### 树管理 & 参数设置

```python
mcts.reset_env(index)           # 重置指定环境
mcts.seed(seed)                 # 设置随机种子
mcts.prune_roots(actions)       # 批量树剪枝

mcts.set_noise_epsilon(eps)
mcts.set_mlh_params(slope, cap, threshold)
mcts.set_c_init(val)
mcts.set_c_base(val)
mcts.set_alpha(val)
mcts.set_fpu_reduction(val)
mcts.set_use_symmetry(val)
mcts.set_value_decay(val)
```

---

## 数据流

### AlphaZero NN 搜索

```
Python batch_playout()
  │
  ├── 循环 n_playout 次:
  │     │
  │     ├── C++ search_batch(boards, turns)     ← GIL 释放
  │     │     ├── 并行 simulate() (UCB selection)
  │     │     ├── 标记终局节点 (is_terminal 缓存)
  │     │     └── 对称变换叶节点 (use_symmetry)
  │     │     → 返回 (leaf_boards, d, p1w, p2w, is_term, turns)
  │     │
  │     ├── Python: 查置换表 / 调 NN
  │     │     ├── _convert_board() → 3-plane 相对视角
  │     │     ├── pv_func.predict()
  │     │     └── _relative_wdl_to_absolute() → 绝对视角
  │     │
  │     └── C++ backprop_batch(policy, d, p1w, p2w, ml, is_term) ← GIL 释放
  │           ├── 对称逆变换 policy
  │           ├── expand_leaf() (创建子节点)
  │           └── propagate() (沿路径更新 WDL/Q/M)
  │
  └── Python: get_mcts_probs() → 动作选择
```

### 纯 MCTS 基线

```
Python rollout_playout()
  │
  └── C++ search(RolloutEvaluator, boards, turns, n_playout) ← GIL 释放
        │
        └── 循环 n_playout 次:
              ├── 并行 simulate() (UCB selection)
              ├── 收集非终局叶节点
              ├── RolloutEvaluator.evaluate_batch()
              │     └── OpenMP 并行 evaluate_single()
              │           └── 随机落子到终局 → WDL
              └── 并行 backprop()
```

---

## C++ 内部类型参考

以下类型不直接暴露给 Python，但对理解接口语义至关重要。

### WDLValue

Win-Draw-Loss 三元组，绝对视角存储。

```cpp
struct WDLValue {
    float d = 0.0f;    // 和棋概率
    float p1w = 0.0f;  // P1 胜率
    float p2w = 0.0f;  // P2 胜率

    // 工厂方法
    static constexpr WDLValue draw();      // {1, 0, 0}
    static constexpr WDLValue p1_wins();   // {0, 1, 0}
    static constexpr WDLValue p2_wins();   // {0, 0, 1}
    static constexpr WDLValue uniform();   // {1/3, 1/3, 1/3}

    // 计算当前落子方视角的 Q 值
    float q(int turn) const;
    // turn=1: Q = p1w - p2w
    // turn=-1: Q = p2w - p1w

    // 增量均值更新（第 n 次采样）
    void update_mean(WDLValue sample, int n);

    // 向 uniform 衰减：γ * self + (1-γ) * uniform
    WDLValue decayed(float gamma) const;
};

// 辅助函数：check_winner() 返回值 → WDLValue
inline WDLValue winner_to_wdl(int winner);
// winner=1 → p1_wins(), winner=-1 → p2_wins()
```

### MCTSNode

搜索树节点，模板参数为 ACTION_SIZE。

```cpp
template <int ACTION_SIZE>
struct MCTSNode {
    int32_t parent;                          // 父节点索引 (-1=根)
    std::array<int32_t, ACTION_SIZE> children; // 子节点索引 (-1=未创建)

    int n_visits;     // 访问次数 N
    float Q;          // Q 值（当前落子方视角）
    WDLValue wdl;     // WDL 运行均值（绝对视角）
    float M;          // 预期剩余步数运行均值
    float prior;      // NN 策略先验 P(a)
    float noise;      // Dirichlet 噪声（仅根子节点）
    bool is_expanded;  // 是否已展开
    bool is_terminal;  // 是否确认终局（缓存优化）
    WDLValue terminal_wdl; // 终局值缓存

    // UCB 分数计算
    float get_ucb(c_init, c_base, parent_n, is_root,
                  noise_epsilon, fpu_value, parent_M,
                  mlh_slope, mlh_cap, mlh_threshold);
    // UCB = Q + U + M_utility
    // U = prior * sqrt(parent_n) / (1 + n_visits) * c(parent_n)
    // c(n) = log((1 + n + c_base) / c_base) + c_init
};
```

### SimResult

`simulate()` 的返回值。

```cpp
template <MCTSGame Game>
struct SimResult {
    Game board;            // 叶节点棋盘状态
    WDLValue terminal_wdl; // 终局 WDL（仅 is_terminal=true 时有效）
    bool is_terminal;      // 是否为终局状态
};
```

### MCTSGame Concept

所有游戏类型必须满足的 C++20 concept：

```cpp
template <typename G>
concept MCTSGame = requires {
    // 编译期常量
    G::Traits::ACTION_SIZE;      // int — 动作空间大小
    G::Traits::BOARD_SIZE;       // int — 棋盘 flat 字节数
    G::Traits::NUM_SYMMETRIES;   // int — 对称变换数量
    G::Traits::BOARD_SHAPE;      // array<int, N> — 棋盘维度

    // 核心游戏逻辑
    g.reset();                   // 重置到初始状态
    g.step(action);              // 执行动作
    g.check_winner() -> int;     // 0=未结束, 1=P1胜, -1=P2胜
    g.is_full() -> bool;         // 棋盘是否已满（平局）
    g.get_valid_moves();         // -> ValidMoves<ACTION_SIZE>

    // 对称变换
    g.apply_symmetry(sym_id);
    G::inverse_symmetry_policy(sym_id, policy_array);

    // 数据交换
    g.board_data() -> const int8_t*; // 棋盘 flat 数据指针
    g.get_turn() -> int;             // 当前落子方 (1/-1)
    g.set_turn(turn);
    g.import_board(src);             // 从 int8 数组导入棋盘
};
```

### 游戏参数速查

| 游戏 | ACTION_SIZE | BOARD_SIZE | BOARD_SHAPE | NUM_SYMMETRIES |
|------|-------------|------------|-------------|----------------|
| Connect4 | 7 | 42 | (6, 7) | 2 |
| Othello | 65 | 64 | (8, 8) | 8 |
| Gomoku | 225 | 225 | (15, 15) | 8 |

---

## 扩展：添加新评估器

实现 `IEvaluator<Game>` 接口即可插入 `search()` 流水线：

```cpp
template <MCTSGame Game>
class MyEvaluator : public IEvaluator<Game> {
public:
    using Result = typename IEvaluator<Game>::Result;

    // 方式 1：重写单个评估（自动 OpenMP 并行）
    Result evaluate_single(const Game &state, int turn) override {
        Result r;
        r.policy = compute_policy(state);   // 你的策略计算
        r.wdl = compute_wdl(state);         // 绝对视角 WDL
        r.moves_left = estimate_moves(state);
        return r;
    }

    // 方式 2：重写批量评估（自定义并行策略）
    void evaluate_batch(std::span<const Game> states,
                        std::span<const int> turns,
                        std::span<Result> results) override {
        // 自定义 batch 推理逻辑
    }
};
```

pybind11 注册：

```cpp
// 在 register_batched_mcts<Game>() 中或 PYBIND11_MODULE 中添加
py::class_<MyEvaluator<Game>, IEvaluator<Game>>(m, "MyEvaluator_Connect4")
    .def(py::init<>());
```

Python 使用：

```python
evaluator = mcts_cpp.MyEvaluator_Connect4()
mcts.search(evaluator, boards, turns, n_playout=800)
```
