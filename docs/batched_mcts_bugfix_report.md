# C++ Batched MCTS Bug 修复报告

> **日期**：2026-02-11
> **Commit**：`c4f6da0` — 修复 C++ Batched MCTS 与 Python MCTS 训练数据不一致的问题
> **涉及文件**：`MCTS.h`, `BatchedMCTS.h`, `MCTS_cpp.py`, `player.py`, `game.py`

---

## 一、背景

项目中存在两套 MCTS 实现：

| 实现 | 入口 | 用途 |
|------|------|------|
| **Python `MCTS_AZ`**（`src/MCTS.py`） | `AlphaZeroPlayer` → `game.self_play()` | 单局自我对弈，已验证正确 |
| **C++ `BatchedMCTS`**（`src/cpp/*.h` + `src/MCTS_cpp.py`） | `BatchedAlphaZeroPlayer` → `game.batch_self_play()` | 批量并行自我对弈，训练数据与 Python 版不一致 |

经过逐行对比，发现 C++ 批量版存在 **4 个 Bug**，其中 2 个为严重级别，直接导致训练数据偏差。

---

## 二、Bug 详细分析

### Bug 1（严重）：终局节点被错误展开

#### 问题描述

在 Python 参考实现中，终局节点（游戏已结束的状态）**永远不会被展开**，始终保持为叶子节点。而 C++ 实现中，`backprop()` 对所有叶子节点一视同仁地执行展开逻辑，包括终局节点。

#### Python 正确行为（`MCTS.py:172-182`）

```python
def playout(self, env):
    noise = None
    node = self.select_leaf_node(env)      # 选择到叶子节点
    # ... 获取 NN 策略和价值 ...
    if not env.done():                      # 关键：只在非终局时展开
        if self.alpha is not None and not self.deterministic:
            noise = np.random.dirichlet([self.alpha for _ in action_probs])
        node.expand(action_probs, noise)    # 展开：创建子节点
    else:
        # 终局：使用实际胜负结果，不展开
        winner = env.winPlayer()
        if winner == 0:
            leaf_value = 0
        else:
            leaf_value = (1 if winner == env.turn else -1)
    node.update(leaf_value)                 # 回传价值
```

**关键点**：`if not env.done()` 守卫确保终局节点的 `children` 字典始终为空，`is_leaf` 属性永远为 `True`。当 MCTS 在后续 playout 中再次访问同一终局节点时：
1. `select_leaf_node()` 中的 `while not node.is_leaf` 在该节点停下（因为无子节点）
2. 再次检测到 `env.done()` 为 True
3. 再次使用实际胜负值回传
4. 该终局节点可被多次访问，每次都回传正确的真实值

#### C++ 错误行为（修复前的 `MCTS.h:137-197`）

```cpp
void backprop(const std::vector<float> &policy_logits, float value)
{
    if (current_leaf_idx == -1) return;

    // BUG: 无条件展开，没有检查是否为终局状态
    std::vector<float> final_policy = policy_logits;
    if (current_flipped) {
        std::reverse(final_policy.begin(), final_policy.end());
        sim_env.flip();
    }
    std::vector<int> valids = sim_env.get_valid_moves();
    // ... 噪声生成 ...
    // ... 为每个合法动作创建子节点 ...
    node_pool[current_leaf_idx].is_expanded = true;  // BUG: 终局节点也被标记为已展开

    // 回传价值
    int32_t update_idx = current_leaf_idx;
    float val = value;
    while (update_idx != -1) { /* ... */ }
}
```

#### 错误后果链

```
第 1 次访问终局节点:
  simulate() → 检测到终局 → 返回 is_terminal=true, terminal_val=-1
  backprop()  → 【错误】为终局节点创建子节点 → is_expanded=true

第 2 次访问同一终局节点:
  simulate() → while(is_expanded) 进入选择循环
           → 遍历终局节点的子节点（幽灵子节点）
           → 在已结束的游戏上继续执行 step()
           → 产生非法棋盘状态
           → 搜索树被污染，所有后续结果不可信
```

#### 修复方案（`MCTS.h:137-201`）

```cpp
void backprop(const std::vector<float> &policy_logits, float value, bool is_terminal)
{
    if (current_leaf_idx == -1) return;

    // 终局状态不展开，保持为叶子节点（与 Python MCTS_AZ 行为一致）
    if (!is_terminal)
    {
        std::vector<float> final_policy = policy_logits;
        if (current_flipped)
        {
            std::reverse(final_policy.begin(), final_policy.end());
            sim_env.flip();
        }
        std::vector<int> valids = sim_env.get_valid_moves();
        // ... 噪声生成（仅根节点）...
        // ... 为每个合法动作创建子节点 ...
        node_pool[current_leaf_idx].is_expanded = true;
    }
    // else: 终局状态，跳过展开，仅回传价值

    // 迭代式价值回传（终局和非终局共用）
    int32_t update_idx = current_leaf_idx;
    float val = value;
    while (update_idx != -1) {
        node_pool[update_idx].n_visits++;
        node_pool[update_idx].Q += (val - node_pool[update_idx].Q) / node_pool[update_idx].n_visits;
        val = -val * discount;
        update_idx = node_pool[update_idx].parent;
    }
}
```

同时修改 `BatchedMCTS.h:116`，将 `is_terminal` 标志传递给各 MCTS 实例：

```cpp
// 修复前:
mcts_envs[i]->backprop(policy, val);

// 修复后:
mcts_envs[i]->backprop(policy, val, is_term[i] != 0);
```

---

### Bug 2（严重）：终局状态的实际胜负值被丢弃

#### 问题描述

`search_batch()` 正确计算了终局状态的实际胜负值（胜：-1，平：0），并通过返回值传回 Python。但 Python 包装层将该值丢弃（赋给 `_`），对所有状态（包括终局）统一使用神经网络的预测值。

#### 错误代码（修复前的 `MCTS_cpp.py:24-32`）

```python
for _ in range(self.n_playout):
    leaf_boards, _, is_term, leaf_turns = self.mcts.search_batch(current_boards, turns)
    #            ↑
    #        终局实际值被丢弃！

    probs, values = pv_func.predict(self._convert_board(leaf_boards, leaf_turns))
    #                ↑
    #        所有状态都用 NN 预测值，包括终局

    self.mcts.backprop_batch(
        np.ascontiguousarray(probs, dtype=np.float32),
        np.ascontiguousarray(values.flatten(), dtype=np.float32),
        is_term
    )
```

#### 对比 Python 正确行为（`MCTS.py:176-181`）

```python
if not env.done():
    # 非终局：使用 NN 预测值
    node.expand(action_probs, noise)
else:
    # 终局：使用实际游戏结果
    winner = env.winPlayer()
    if winner == 0:
        leaf_value = 0           # 平局
    else:
        leaf_value = (1 if winner == env.turn else -1)  # 胜/负
```

#### 错误后果

- 训练初期 NN 不准确时，一个明显的终局胜利可能被 NN 预测为平局或失败
- 错误的价值信号污染搜索树，导致 MCTS 低估或高估某些走法
- 价值网络的训练目标被扭曲，形成恶性循环

#### 修复方案（`MCTS_cpp.py:24-38`）

```python
for _ in range(self.n_playout):
    leaf_boards, term_vals, is_term, leaf_turns = self.mcts.search_batch(current_boards, turns)
    #            ↑
    #        现在正确接收终局值

    probs, values = pv_func.predict(self._convert_board(leaf_boards, leaf_turns))
    values = values.flatten()

    # 终局状态使用实际胜负值，而非 NN 预测
    term_mask = is_term.astype(bool)
    values[term_mask] = term_vals[term_mask]

    self.mcts.backprop_batch(
        np.ascontiguousarray(probs, dtype=np.float32),
        np.ascontiguousarray(values, dtype=np.float32),
        is_term
    )
```

---

### Bug 3（中等）：训练策略目标的计算方式不一致

#### 问题描述

自我对弈生成的训练数据中，策略目标（`mcts_probs`）由 MCTS 的访问次数分布计算得来。Python 和 C++ 对同一组访问次数使用了不同的转换公式。

#### Python 正确行为（`player.py:122-137` — `AlphaZeroPlayer.get_action()`）

```python
def get_action(self, env, temp=0):
    action_probs = np.zeros((self.n_actions,), dtype=np.float32)
    actions, visits = self.mcts.get_action_visits(env)

    # 策略目标：对 log(visits) 进行温度缩放后 softmax
    visit_dist = softmax(np.log(visits) / max(temp, 1e-8))
    action_probs[list(actions)] = visit_dist

    # 采样逻辑
    if temp == 0:
        probs = np.zeros((len(visits),), dtype=np.float32)
        probs[np.where(np.array(visits) == max(visits))] = 1 / list(visits).count(max(visits))
    else:
        probs = visit_dist
    action = np.random.choice(actions, p=probs)

    return action, action_probs  # action_probs 是训练策略目标
```

**公式**：$\pi_a = \text{softmax}\left(\frac{\log N_a}{T}\right)$

其中 $N_a$ 是动作 $a$ 的访问次数，$T$ 是温度参数。

#### C++ 错误行为（修复前的 `player.py:167-186` — `BatchedAlphaZeroPlayer.get_action()`）

```python
for i in range(self.n_envs):
    visit = visits[i]
    temp = temps[i]
    visit_sum = visit.sum()
    if visit_sum == 0:
        probs = np.ones_like(visit) / len(visit)
    else:
        probs = visit / visit_sum  # BUG: 原始访问比例，未经温度缩放

    if temp == 0:
        action = np.argmax(visit)
    else:
        visit_temp = np.power(visit, 1.0 / temp)
        prob_sample = visit_temp / visit_temp.sum()
        action = np.random.choice(len(visit), p=prob_sample)

    batch_probs.append(probs)  # 训练策略目标使用了错误的 probs
```

**公式**：$\pi_a = \frac{N_a}{\sum_b N_b}$（无温度缩放）

#### 数值差异示例

假设访问次数 `visits = [50, 30, 20]`，温度 `temp = 0.5`：

| 方法 | 动作 0 | 动作 1 | 动作 2 |
|------|--------|--------|--------|
| Python `softmax(log(N)/T)` | 0.847 | 0.118 | 0.035 |
| C++ `N / sum(N)` | 0.500 | 0.300 | 0.200 |

差异巨大。温度越低（游戏后期），差异越明显。

#### 修复方案（`player.py:167-192`）

```python
for i in range(self.n_envs):
    visit = visits[i]
    temp = temps[i]

    # 训练策略目标：与 Python AlphaZeroPlayer 一致，使用 softmax(log(visits)/temp)
    action_probs = np.zeros(self.n_actions, dtype=np.float32)
    valid_mask = visit > 0
    if valid_mask.any():
        log_visits = np.log(np.maximum(visit[valid_mask], 1e-8))
        visit_dist = softmax(log_visits / max(temp, 1e-8))
        action_probs[valid_mask] = visit_dist
    else:
        action_probs = np.ones(self.n_actions, dtype=np.float32) / self.n_actions

    # 动作采样
    if temp == 0:
        action = np.argmax(visit)
    else:
        valid_actions = np.where(valid_mask)[0]
        action = np.random.choice(valid_actions, p=visit_dist)

    batch_actions.append(action)
    batch_probs.append(action_probs)
```

---

### Bug 4（轻微）：`batch_self_play` 中已结束的游戏未重置 MCTS 树

#### 问题描述

`batch_self_play()` 中，当某个游戏结束后，对应的 MCTS 搜索树没有被重置。由于 `current_boards` 和 `turns` 是对全部 `n_games` 构建的（而非仅活跃游戏），已结束游戏的陈旧棋盘仍然被送入 `search_batch()` 和 `backprop_batch()`，白白浪费计算资源。

#### 错误代码（修复前的 `game.py:111-112`）

```python
# 游戏结束，重置该环境的 MCTS 树（可选，取决于是否复用）
# player.reset_env(i)    ← 被注释掉了
```

#### 修复方案（`game.py:111-112`）

```python
# 游戏结束，重置该环境的 MCTS 树
player.mcts.reset_env(i)
```

---

## 三、修改文件汇总

### 1. `src/cpp/MCTS.h`

| 行号 | 修改内容 |
|------|----------|
| 137 | `backprop` 方法签名增加 `bool is_terminal` 参数 |
| 141-191 | 整个展开逻辑包裹在 `if (!is_terminal)` 中，终局时跳过 |
| 193-201 | 价值回传逻辑保持不变，终局和非终局共用 |

**修改前**：
```cpp
void backprop(const std::vector<float> &policy_logits, float value)
{
    // 无条件展开...
    node_pool[current_leaf_idx].is_expanded = true;
    // 价值回传...
}
```

**修改后**：
```cpp
void backprop(const std::vector<float> &policy_logits, float value, bool is_terminal)
{
    if (!is_terminal)
    {
        // 仅非终局时展开...
        node_pool[current_leaf_idx].is_expanded = true;
    }
    // 价值回传（共用）...
}
```

### 2. `src/cpp/BatchedMCTS.h`

| 行号 | 修改内容 |
|------|----------|
| 116 | 调用 `backprop` 时传入 `is_term[i] != 0` |

**修改前**：`mcts_envs[i]->backprop(policy, val);`
**修改后**：`mcts_envs[i]->backprop(policy, val, is_term[i] != 0);`

### 3. `src/MCTS_cpp.py`

| 行号 | 修改内容 |
|------|----------|
| 25 | `_` 改为 `term_vals`，接收终局值 |
| 28 | `values = values.flatten()` 单独一行 |
| 30-32 | 新增终局值替换逻辑 |

### 4. `src/player.py`

| 行号 | 修改内容 |
|------|----------|
| 171-179 | 策略目标改为 `softmax(log(visits)/temp)` |
| 181-186 | 采样逻辑相应调整 |

### 5. `src/game.py`

| 行号 | 修改内容 |
|------|----------|
| 111-112 | 取消注释 `reset_env(i)`，改为 `player.mcts.reset_env(i)` |

---

## 四、数据流对比（修复前 vs 修复后）

### 修复前的数据流

```
search_batch()
  ├─ 返回 leaf_boards, term_vals, is_term, leaf_turns
  │                     ↓
  │                 赋给 _ (丢弃!)
  │
  ├─ NN.predict(leaf_boards)  →  probs, values  (所有状态都用 NN 值)
  │
  └─ backprop_batch(probs, values, is_term)
       └─ backprop(policy, val)  ← 没有 is_terminal 参数
            └─ 无条件展开 + 回传
```

### 修复后的数据流

```
search_batch()
  ├─ 返回 leaf_boards, term_vals, is_term, leaf_turns
  │                     ↓
  │                 保留为 term_vals
  │
  ├─ NN.predict(leaf_boards)  →  probs, values
  │
  ├─ values[is_term] = term_vals[is_term]   ← 终局值替换
  │
  └─ backprop_batch(probs, values, is_term)
       └─ backprop(policy, val, is_terminal)  ← 传入终局标志
            ├─ if (!is_terminal): 展开 + 回传
            └─ else:              仅回传 (保持叶子)
```

---

## 五、验证方式

1. **编译测试**：`python setup.py build_ext --inplace` 编译通过
2. **终局节点测试**：构造接近终局的棋盘，运行 C++ MCTS，验证终局节点 `is_expanded == false` 且无子节点
3. **对比测试**：固定随机种子，同一网络和初始状态下分别运行 Python `MCTS_AZ` 和 C++ `BatchedMCTS`，对比：
   - 根节点访问次数分布
   - 选择的动作
   - 训练策略目标（`mcts_probs`）
4. **训练数据对比**：运行 `self_play()` 和 `batch_self_play()`，比较生成的 `(state, action, mcts_probs, value)` 元组
