# FPU (First-Play Urgency) 修改方案

## 目标
将 MCTS 中未访问节点的 UCB 值从 `∞`（强制全部访问）改为 FPU（基于父节点价值的悲观估计），提升搜索效率。

**FPU 公式**: `Q_unvisited = parent_value - fpu_reduction × √(seen_policy)`
- `parent_value`: 从当前玩家视角的父节点价值（`+parent.Q`，因为 Q 存的是该节点走棋玩家自己的视角，而 parent.Q ≈ avg(-children.Q)，恰好等于 UCB 中 -child.Q 的均值）
- `seen_policy`: 所有已访问子节点的 prior 之和
- `fpu_reduction`: 默认 0.4

同时将 CPUCT 从动态公式改为固定值（默认 4.0），与 FPU 配合使用。

---

## 修改 1: `src/cpp/MCTSNode.h`

### 修改前（line 39-54）:
```cpp
        // 计算 UCB 时需要传入父节点的访问次数，因为现在不通过指针找 parent
        [[nodiscard]] float get_ucb(float c_init, float c_base, float parent_n,
                                     bool is_root_node, float noise_epsilon) const {
            float effective_prior = prior;
            if (is_root_node) {
                effective_prior = (1.0f - noise_epsilon) * prior + noise_epsilon * noise;
            }

            if (n_visits == 0) {
                return std::numeric_limits<float>::infinity();
            }

            float c_puct = c_init + std::log((parent_n + c_base + 1.0f) / c_base);
            float u_score = c_puct * effective_prior * std::sqrt(parent_n) / (1.0f + n_visits);
            return -Q + u_score;
        }
```

### 修改后:
```cpp
        // FPU-based UCB: 未访问节点使用 fpu_value 而非 ∞
        [[nodiscard]] float get_ucb(float cpuct, float sqrt_parent_n,
                                     bool is_root_node, float noise_epsilon,
                                     float fpu_value) const {
            float effective_prior = prior;
            if (is_root_node) {
                effective_prior = (1.0f - noise_epsilon) * prior + noise_epsilon * noise;
            }

            float q_value = (n_visits == 0) ? fpu_value : -Q;
            float u_score = cpuct * effective_prior * sqrt_parent_n / (1.0f + n_visits);
            return q_value + u_score;
        }
```

### 变更说明:
- 参数 `c_init, c_base, parent_n` → `cpuct, sqrt_parent_n, fpu_value`
- `n_visits == 0` 时返回 `fpu_value + U` 而非 `∞`
- CPUCT 由调用者传入固定值，不再内部计算动态公式
- `sqrt_parent_n` 由调用者预计算，避免在循环内重复计算
- 可以移除 `#include <limits>`（不再需要 `infinity()`）

---

## 修改 2: `src/cpp/MCTS.h`

### 2a. 成员变量（line 34-35）

#### 修改前:
```cpp
        float c_init, c_base, discount, alpha;
        float noise_epsilon;
```

#### 修改后:
```cpp
        float cpuct, discount, alpha;
        float noise_epsilon;
        float fpu_reduction;
```

### 2b. 构造函数（line 37-43）

#### 修改前:
```cpp
        MCTS(float c_i, float c_b, float disc, float a, float noise_eps = 0.25f)
            : c_init(c_i), c_base(c_b), discount(disc), alpha(a), noise_epsilon(noise_eps)
        {
            // 预分配内存，减少 search 过程中的扩容
            node_pool.resize(2000);
            reset();
        }
```

#### 修改后:
```cpp
        MCTS(float cpuct_, float disc, float a, float noise_eps = 0.25f, float fpu_red = 0.4f)
            : cpuct(cpuct_), discount(disc), alpha(a), noise_epsilon(noise_eps), fpu_reduction(fpu_red)
        {
            // 预分配内存，减少 search 过程中的扩容
            node_pool.resize(2000);
            reset();
        }
```

#### 变更说明:
- 移除 `c_init`, `c_base`，改为 `cpuct`（固定 CPUCT 值）
- 新增 `fpu_reduction`（默认 0.4）
- 构造函数参数减少一个（去掉 `c_base`），新增 `fpu_red`

### 2c. simulate() 选择逻辑（line 81-108）

#### 修改前:
```cpp
            while (node_pool[curr_idx].is_expanded)
            {
                float best_score = -std::numeric_limits<float>::infinity();
                int best_action = -1;
                float p_n = static_cast<float>(node_pool[curr_idx].n_visits);

                auto valids = sim_env.get_valid_moves();
                if (valids.empty()) break;

                for (int action : valids)
                {
                    int32_t child_idx = node_pool[curr_idx].children[action];
                    if (child_idx != -1)
                    {
                        float score = node_pool[child_idx].get_ucb(c_init, c_base, p_n, curr_idx == root_idx, noise_epsilon);
                        if (score > best_score) {
                            best_score = score;
                            best_action = action;
                        }
                    }
                }
                if (best_action == -1) break;

                sim_env.step(best_action);
                curr_idx = node_pool[curr_idx].children[best_action];

                if (sim_env.check_winner() != 0 || sim_env.is_full()) break;
            }
```

#### 修改后:
```cpp
            while (node_pool[curr_idx].is_expanded)
            {
                float best_score = -std::numeric_limits<float>::infinity();
                int best_action = -1;
                float p_n = static_cast<float>(node_pool[curr_idx].n_visits);
                float sqrt_p_n = std::sqrt(p_n);
                bool is_root = (curr_idx == root_idx);

                auto valids = sim_env.get_valid_moves();
                if (valids.empty()) break;

                // FPU: 计算父节点价值和已访问子节点 prior 之和
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

                for (int action : valids)
                {
                    int32_t child_idx = node_pool[curr_idx].children[action];
                    if (child_idx != -1)
                    {
                        float score = node_pool[child_idx].get_ucb(cpuct, sqrt_p_n, is_root, noise_epsilon, fpu_value);
                        if (score > best_score) {
                            best_score = score;
                            best_action = action;
                        }
                    }
                }
                if (best_action == -1) break;

                sim_env.step(best_action);
                curr_idx = node_pool[curr_idx].children[best_action];

                if (sim_env.check_winner() != 0 || sim_env.is_full()) break;
            }
```

#### 变更说明:
- 在选择子节点之前，先遍历一次合法走法，累加已访问子节点的 prior 得到 `seen_policy`
- 计算 `fpu_value = +parent.Q - fpu_reduction * √(seen_policy)`
- `sqrt_p_n` 预计算一次，传入 `get_ucb`
- `get_ucb` 调用参数从 `(c_init, c_base, p_n, is_root, noise_epsilon)` 改为 `(cpuct, sqrt_p_n, is_root, noise_epsilon, fpu_value)`

---

## 修改 3: `src/cpp/BatchedMCTS.h`

### 构造函数（line 23-31）

#### 修改前:
```cpp
        BatchedMCTS(int num_envs, float c_init, float c_base, float discount, float alpha,
                    float noise_epsilon = 0.25f)
            : n_envs(num_envs)
        {
            mcts_envs.reserve(n_envs);
            for (int i = 0; i < n_envs; ++i)
            {
                mcts_envs.push_back(std::make_unique<MCTS<Game>>(c_init, c_base, discount, alpha, noise_epsilon));
            }
        }
```

#### 修改后:
```cpp
        BatchedMCTS(int num_envs, float cpuct, float discount, float alpha,
                    float noise_epsilon = 0.25f, float fpu_reduction = 0.4f)
            : n_envs(num_envs)
        {
            mcts_envs.reserve(n_envs);
            for (int i = 0; i < n_envs; ++i)
            {
                mcts_envs.push_back(std::make_unique<MCTS<Game>>(cpuct, discount, alpha, noise_epsilon, fpu_reduction));
            }
        }
```

#### 变更说明:
- `c_init, c_base` → `cpuct`（减少一个参数）
- 新增 `fpu_reduction`（默认 0.4）
- 传递给 `MCTS<Game>` 构造函数

---

## 修改 4: `src/cpp/bindings.cpp`

### 构造函数绑定（line 22-24）

#### 修改前:
```cpp
        .def(py::init<int, float, float, float, float, float>(),
             py::arg("n_envs"), py::arg("c_init"), py::arg("c_base"),
             py::arg("discount"), py::arg("alpha"), py::arg("noise_epsilon") = 0.25f)
```

#### 修改后:
```cpp
        .def(py::init<int, float, float, float, float, float>(),
             py::arg("n_envs"), py::arg("cpuct"), py::arg("discount"),
             py::arg("alpha"), py::arg("noise_epsilon") = 0.25f,
             py::arg("fpu_reduction") = 0.4f)
```

#### 变更说明:
- `c_init, c_base` → `cpuct`
- 新增 `fpu_reduction` 参数（默认 0.4）
- 参数总数不变（仍为 6 个），但含义改变

---

## 修改 5: `src/MCTS_cpp.py`

### `__init__` 方法（line 23-26）

#### 修改前:
```python
    def __init__(self, batch_size, c_init, c_base, discount, alpha, n_playout,
                 game_name='Connect4', board_converter=None, cache_size=0, noise_epsilon=0.25):
        backend_cls = _BACKENDS[game_name]
        self.mcts = backend_cls(batch_size, c_init, c_base, discount, alpha, noise_epsilon)
```

#### 修改后:
```python
    def __init__(self, batch_size, cpuct, discount, alpha, n_playout,
                 game_name='Connect4', board_converter=None, cache_size=0,
                 noise_epsilon=0.25, fpu_reduction=0.4):
        backend_cls = _BACKENDS[game_name]
        self.mcts = backend_cls(batch_size, cpuct, discount, alpha, noise_epsilon, fpu_reduction)
```

#### 变更说明:
- `c_init, c_base` → `cpuct`（减少一个参数）
- 新增 `fpu_reduction`（默认 0.4）
- 传递给 C++ 后端

---

## 修改 6: `src/player.py`

### `BatchedAlphaZeroPlayer.__init__`（line 145-150）

#### 修改前:
```python
    def __init__(self, policy_value_fn, n_envs, c_init=1.25, c_base=500, n_playout=100, discount=1, alpha=0.3,
                 game_name='Connect4', board_converter=None, cache_size=0):
        self.pv_func = policy_value_fn
        self.mcts = BatchedMCTS(n_envs, c_init, c_base, discount, alpha, n_playout,
                                game_name=game_name, board_converter=board_converter,
                                cache_size=cache_size)
```

#### 修改后:
```python
    def __init__(self, policy_value_fn, n_envs, cpuct=4.0, n_playout=100, discount=1, alpha=0.3,
                 game_name='Connect4', board_converter=None, cache_size=0,
                 noise_epsilon=0.25, fpu_reduction=0.4):
        self.pv_func = policy_value_fn
        self.mcts = BatchedMCTS(n_envs, cpuct, discount, alpha, n_playout,
                                game_name=game_name, board_converter=board_converter,
                                cache_size=cache_size, noise_epsilon=noise_epsilon,
                                fpu_reduction=fpu_reduction)
```

#### 变更说明:
- `c_init=1.25, c_base=500` → `cpuct=4.0`（使用固定 CPUCT，默认 4.0）
- 新增 `noise_epsilon` 和 `fpu_reduction` 参数透传
- 注意: 之前 `noise_epsilon` 没有从 `BatchedAlphaZeroPlayer` 传到 `BatchedMCTS`，现在补上

---

## 修改 7: `src/pipeline.py`

### `_batched_eval_games` 中创建 MCTS（line 172-179）

#### 修改前:
```python
        mcts_p1 = PyBatchedMCTS(
            n_envs, c_init=self.c_puct, c_base=500, discount=self.discount,
            alpha=self.dirichlet_alpha, n_playout=n_playout, game_name=self.env_name,
            noise_epsilon=eval_noise_eps)
        mcts_p2 = PyBatchedMCTS(
            n_envs, c_init=self.c_puct, c_base=500, discount=self.discount,
            alpha=self.dirichlet_alpha, n_playout=n_playout, game_name=self.env_name,
            noise_epsilon=eval_noise_eps)
```

#### 修改后:
```python
        mcts_p1 = PyBatchedMCTS(
            n_envs, cpuct=self.c_puct, discount=self.discount,
            alpha=self.dirichlet_alpha, n_playout=n_playout, game_name=self.env_name,
            noise_epsilon=eval_noise_eps)
        mcts_p2 = PyBatchedMCTS(
            n_envs, cpuct=self.c_puct, discount=self.discount,
            alpha=self.dirichlet_alpha, n_playout=n_playout, game_name=self.env_name,
            noise_epsilon=eval_noise_eps)
```

#### 变更说明:
- `c_init=..., c_base=500` → `cpuct=...`

---

## 修改 8: `client.py`

### 8a. argparse 参数（line 31-32）

#### 修改前:
```python
parser.add_argument('-c', '--c_init', type=float, default=1., help='C_puct init')
parser.add_argument('--c_base', type=float, default=500, help='C_puct base')
```

#### 修改后:
```python
parser.add_argument('-c', '--cpuct', type=float, default=4.0, help='CPUCT exploration constant')
parser.add_argument('--fpu_reduction', type=float, default=0.4, help='FPU reduction factor')
```

### 8b. 创建 BatchedAlphaZeroPlayer（line 68-75）

#### 修改前:
```python
        self.az_player = BatchedAlphaZeroPlayer(self.net,
                                                n_envs=self.batch_size,
                                                c_init=args.c_init,
                                                c_base=args.c_base,
                                                n_playout=args.n,
                                                discount=args.discount,
                                                alpha=args.alpha,
                                                cache_size=args.cache_size)
```

#### 修改后:
```python
        self.az_player = BatchedAlphaZeroPlayer(self.net,
                                                n_envs=self.batch_size,
                                                cpuct=args.cpuct,
                                                n_playout=args.n,
                                                discount=args.discount,
                                                alpha=args.alpha,
                                                cache_size=args.cache_size,
                                                fpu_reduction=args.fpu_reduction)
```

---

## 修改 9: `server.py`

### 9a. argparse 参数（line 29）

#### 修改前:
```python
parser.add_argument('-c', '--c_init', type=float, default=1., help='C_puct init')
```

#### 修改后:
```python
parser.add_argument('-c', '--cpuct', type=float, default=4.0, help='CPUCT exploration constant')
parser.add_argument('--fpu_reduction', type=float, default=0.4, help='FPU reduction factor')
```

### 9b. config 字典（line 50）

#### 修改前:
```python
          "c_puct": args.c_init,
```

#### 修改后:
```python
          "c_puct": args.cpuct,
```

### 9c. 关于 `pipeline.py` line 52-54 的 `AlphaZeroPlayer`

```python
        self.az_player = AlphaZeroPlayer(self.net, c_init=self.c_puct, n_playout=self.n_playout,
                                         discount=self.discount, alpha=self.dirichlet_alpha, is_selfplay=1,
                                         cache_size=self.cache_size)
```

这里 `AlphaZeroPlayer` 使用 Python MCTS（非 C++ BatchedMCTS），参数名是 `c_init`。
Python MCTS (`src/MCTS.py`) 的 FPU 改动比较复杂（需要在 TreeNode 层面传递父节点信息），且 `AlphaZeroPlayer` 只在 server 端的 Elo 评估中使用单局对弈。

**本次暂不修改 Python MCTS**，只修改 C++ BatchedMCTS（训练和批量评估的核心路径）。
这里 `c_init=self.c_puct` 传入的值会随着 server `--cpuct` 改变（因为 config 里的 key 不变，仍是 `c_puct`），不影响功能。

---

## 修改 10: `play.py`

### 10a. argparse 参数（line 22）

#### 修改前:
```python
parser.add_argument('-c', '--c_init', type=float, default=4, help='C_puct init')
```

#### 修改后:
```python
parser.add_argument('-c', '--cpuct', type=float, default=4, help='CPUCT exploration constant')
```

### 10b. 创建 AlphaZeroPlayer（line 46-47）

#### 修改前:
```python
            az_player = AlphaZeroPlayer(net, c_init=args.c_init,
                                        n_playout=args.n, discount=0.99, alpha=args.alpha, is_selfplay=0)
```

#### 修改后:
```python
            az_player = AlphaZeroPlayer(net, c_init=args.cpuct,
                                        n_playout=args.n, discount=0.99, alpha=args.alpha, is_selfplay=0)
```

注意: `AlphaZeroPlayer` 使用 Python MCTS，参数名仍为 `c_init`（暂不改 Python MCTS）。只是命令行参数名从 `--c_init` 改为 `--cpuct`。

---

## 不修改的文件（本次范围外）

| 文件 | 原因 |
|------|------|
| `src/MCTS.py` | Python MCTS，FPU 改动复杂，且非训练核心路径，暂不修改 |
| `gui_play.py` | 使用 Python MCTS 的 `AlphaZeroPlayer`，暂不修改 |
| `src/environments/Connect4/Network.py` | 网络容量问题，本次不改 |
| `src/environments/NetworkBase.py` | 训练超参数，本次不改 |
| `src/game.py` | 温度调度，本次不改 |

---

## 编译与验证

1. **编译**: `python setup.py build_ext --inplace`
2. **快速验证**: 启动一个 client 连接 server，确认 MCTS 正常运行
3. **搜索质量验证**: 在已知局面测试 100 次模拟的走法分布
