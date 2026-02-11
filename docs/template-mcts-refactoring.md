# C++ MCTS 模板化重构：解耦游戏与算法

## 概述

将 `MCTS`、`BatchedMCTS`、`MCTSNode` 从与 Connect4 硬编码耦合，重构为基于 C++20 模板 + concept 的通用架构。重构后添加新游戏只需 3 步，MCTS 算法层不包含任何游戏特定代码。

## 重构前的问题

| 耦合点 | 具体表现 |
|--------|---------|
| `Constants.h` | `ROWS=6, COLS=7, ACTION_SIZE=7` 全局硬编码 |
| `MCTSNode` | `children` 数组固定 `std::array<int32_t, 7>` |
| `MCTS` | 直接 `#include "Connect4.h"`，成员 `Connect4 sim_env` |
| `BatchedMCTS` | `memcpy` 偏移量硬编码 `Config::ROWS * Config::COLS` |
| `bindings.cpp` | numpy 输出形状硬编码 `{batch_size, 6, 7}` |
| `MCTSNode::get_ucb()` | 读取全局 `Config::NOISE_EPSILON` |

## 新架构

```
GameContext.h          MCTSGame concept + ValidMoves<N>
    ↑                      ↑
MCTSNode.h<N>         MCTS.h<Game>  →  BatchedMCTS.h<Game>
                           ↑
                      bindings.cpp ──→ register_batched_mcts<Game>()
                           ↑
                      Connect4.h (实现 MCTSGame concept)
```

MCTS 层仅依赖 `GameContext.h` 中定义的 concept，不 include 任何具体游戏头文件。游戏头文件仅在 `bindings.cpp` 中被 include，用于模板实例化。

## 核心设计

### 1. MCTSGame Concept (`GameContext.h`)

每个游戏类型必须满足：

```cpp
template <typename G>
concept MCTSGame = requires(G g, const G cg, int action, const int8_t *src) {
    // 编译期常量
    { G::Traits::ACTION_SIZE } -> std::convertible_to<int>;
    { G::Traits::BOARD_SIZE }  -> std::convertible_to<int>;

    // 游戏逻辑
    { g.reset() }         -> std::same_as<void>;
    { g.step(action) }    -> std::same_as<void>;
    { cg.check_winner() } -> std::same_as<int>;
    { cg.is_full() }      -> std::same_as<bool>;
    { g.flip() }          -> std::same_as<void>;
    { cg.get_valid_moves() };

    // Python ↔ C++ I/O
    { cg.board_data() }      -> std::same_as<const int8_t *>;
    { cg.get_turn() }        -> std::same_as<int>;
    { g.set_turn(action) }   -> std::same_as<void>;
    { g.import_board(src) }  -> std::same_as<void>;

    requires std::is_copy_constructible_v<G>;
};
```

### 2. 游戏类的 Traits 结构体

每个游戏类内嵌一个 `Traits`，提供编译期维度信息：

```cpp
class Connect4 {
public:
    struct Traits {
        static constexpr int ROWS        = 6;
        static constexpr int COLS        = 7;
        static constexpr int ACTION_SIZE = 7;
        static constexpr int BOARD_SIZE  = ROWS * COLS;  // 42
        static constexpr std::array<int, 2> BOARD_SHAPE = {ROWS, COLS};
    };
    // ...
};
```

MCTS 层通过 `Game::Traits::ACTION_SIZE` 等在编译期获取维度，零运行时开销。

### 3. I/O 接口

游戏类提供 4 个方法，封装了 Python ↔ C++ 之间的棋盘数据传输：

| 方法 | 用途 |
|------|------|
| `board_data()` | 返回 `const int8_t*`，指向可 memcpy 的棋盘数据 |
| `import_board(const int8_t* src)` | 从原始字节导入棋盘状态（含内部重建，如 bitboard 同步） |
| `get_turn()` | 获取当前轮次 |
| `set_turn(int t)` | 设置当前轮次 |

`BatchedMCTS` 通过这些接口与游戏交互，不再直接操作 `board` 数组或调用 `sync_from_board()`。

### 4. ValidMoves 模板

从 Connect4 中抽出，参数化为最大动作数：

```cpp
template <int MAX_ACTIONS>
struct ValidMoves {
    int moves[MAX_ACTIONS];
    int count = 0;
    // begin(), end(), empty(), size()
};
```

### 5. bindings.cpp 注册模式

一个模板函数处理所有 pybind11 绑定，numpy 形状从 `Traits` 动态构建：

```cpp
template <MCTSGame Game>
void register_batched_mcts(py::module_ &m, const char *name) {
    using BM = BatchedMCTS<Game>;
    // ... 所有绑定，维度从 Game::Traits 获取
}

PYBIND11_MODULE(mcts_cpp, m) {
    register_batched_mcts<Connect4>(m, "BatchedMCTS_Connect4");
    // register_batched_mcts<NewGame>(m, "BatchedMCTS_NewGame");
}
```

### 6. Python 层分发

`MCTS_cpp.py` 通过字典将游戏名映射到 C++ 后端类：

```python
_BACKENDS = {
    'Connect4': mcts_cpp.BatchedMCTS_Connect4,
}

class BatchedMCTS:
    def __init__(self, ..., game_name='Connect4', board_converter=None):
        backend_cls = _BACKENDS[game_name]
        self.mcts = backend_cls(...)
        self.action_size = backend_cls.action_size   # 从 C++ Traits 获取
        self.board_shape = backend_cls.board_shape
        self._convert_board = board_converter or _default_convert_board
```

`board_converter` 允许每个游戏自定义棋盘 → 神经网络输入张量的转换逻辑。

## 添加新游戏的步骤

### 1. 编写游戏类 (`src/cpp/NewGame.h`)

实现 `MCTSGame` concept 的所有要求：

```cpp
class NewGame {
public:
    struct Traits {
        static constexpr int ACTION_SIZE = 9;      // 例如 TicTacToe
        static constexpr int BOARD_SIZE  = 9;
        static constexpr std::array<int, 2> BOARD_SHAPE = {3, 3};
    };

    int8_t board[3][3];
    int turn = 1;

    void reset();
    void step(int action);
    int check_winner() const;
    bool is_full() const;
    void flip();
    ValidMoves<Traits::ACTION_SIZE> get_valid_moves() const;

    const int8_t* board_data() const { return &board[0][0]; }
    int get_turn() const { return turn; }
    void set_turn(int t) { turn = t; }
    void import_board(const int8_t* src) { std::memcpy(board, src, sizeof(board)); }
};
```

### 2. 注册到 bindings.cpp

```cpp
#include "NewGame.h"
// ...
register_batched_mcts<NewGame>(m, "BatchedMCTS_NewGame");
```

### 3. 注册到 Python

在 `src/MCTS_cpp.py` 的 `_BACKENDS` 中添加：

```python
_BACKENDS = {
    'Connect4': mcts_cpp.BatchedMCTS_Connect4,
    'NewGame':  mcts_cpp.BatchedMCTS_NewGame,
}
```

可选：传入自定义 `board_converter`（如果默认的 3 通道转换不适用）。

## 变更文件清单

| 文件 | 操作 |
|------|------|
| `src/cpp/Constants.h` | **删除** |
| `src/cpp/GameContext.h` | 重写：`ValidMoves<N>` + `MCTSGame` concept |
| `src/cpp/Connect4.h` | 添加 `Traits`、I/O 方法，`Config::` → `Traits::` |
| `src/cpp/MCTSNode.h` | 模板化 `MCTSNode<int ACTION_SIZE>` |
| `src/cpp/MCTS.h` | 模板化 `MCTS<MCTSGame Game>` |
| `src/cpp/BatchedMCTS.h` | 模板化 `BatchedMCTS<MCTSGame Game>` |
| `src/cpp/bindings.cpp` | `register_batched_mcts<Game>()` 模板注册 |
| `src/MCTS_cpp.py` | `_BACKENDS` 分发 + `board_converter` |
| `src/player.py` | `BatchedAlphaZeroPlayer` 增加 `game_name` 参数 |

## 设计决策

**为什么用模板而不用虚函数？** MCTS 的 `simulate()` 和 `backprop()` 是热循环，每秒调用数百万次 `step()`、`check_winner()`、`get_valid_moves()`。模板在编译期展开，零虚函数调用开销。代价是每个游戏实例化一份 MCTS 代码，但游戏数量有限（2-5 个），二进制膨胀可忽略。

**为什么 `noise_epsilon` 改为运行时参数？** 原来作为 `Config::NOISE_EPSILON` 全局常量，删除 `Constants.h` 后自然成为 `MCTS` 的构造参数，传递给 `MCTSNode::get_ucb()`。这也允许不同游戏使用不同的噪声强度。

**为什么需要 `import_board()` 而不是直接 memcpy？** Connect4 使用 bitboard 优化，从 Python 导入 `int8_t` 数组后需要调用 `sync_from_board()` 重建 bitboard 状态。`import_board()` 将这个内部细节封装在游戏类内部，MCTS 层不需要知道。

**`flip()` 对不支持对称性的游戏怎么办？** 实现为空操作（no-op）即可。MCTS 以 50% 概率调用 `flip()`，空操作不影响正确性。
