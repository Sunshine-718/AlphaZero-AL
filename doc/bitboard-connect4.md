# Connect4 Bitboard 优化

## 概述

将 Connect4 游戏环境从 `int8_t board[6][7]` 数组实现重构为 Bitboard 实现，同时优化 `get_valid_moves()` 的堆分配和 `is_full()` 的检测方式。

## Bitboard 编码

使用两个 `uint64_t` 分别存储两个玩家的棋子位置。每列占 7 位（6 行 + 1 哨兵位），共 49 位：

```
col 0   col 1   col 2   col 3   col 4   col 5   col 6
  6      13      20      27      34      41      48    <- 哨兵位
  5      12      19      26      33      40      47    <- row 0 (顶行)
  4      11      18      25      32      39      46
  3      10      17      24      31      38      45
  2       9      16      23      30      37      44
  1       8      15      22      29      36      43
  0       7      14      21      28      35      42    <- row 5 (底行)
```

`board[row][col]` 对应 `bit = col * 7 + (ROWS - 1 - row)`。

核心状态：

| 字段 | 类型 | 说明 |
|------|------|------|
| `bb[2]` | `uint64_t[2]` | 两个玩家的棋子位图 |
| `height[7]` | `int[7]` | 每列下一个可用位的 bit 索引 |
| `n_pieces` | `int` | 棋盘上总棋子数 |
| `last_player_idx` | `int` | 上一步落子的玩家索引 |

## 优化细节

### 1. check_winner(): O(n) -> O(1)

原实现需要从 `last_r, last_c` 出发，沿 4 个方向各遍历最多 6 步。Bitboard 只需 4 次 shift + AND：

```cpp
uint64_t b = bb[last_player_idx];
uint64_t t;
t = b & (b >> 1);  if (t & (t >> 2))  return winner;  // 垂直
t = b & (b >> 7);  if (t & (t >> 14)) return winner;  // 水平
t = b & (b >> 6);  if (t & (t >> 12)) return winner;  // 对角线 (\)
t = b & (b >> 8);  if (t & (t >> 16)) return winner;  // 对角线 (/)
```

原理：`b & (b >> k)` 找到所有"在方向 k 上连续两个"的位置，再 `& (>> 2k)` 找到连续四个。

### 2. step(): O(ROWS) -> O(1)

原实现从底行向上遍历找空位。Bitboard 用 `height[col]` 直接定位：

```cpp
bb[player_idx] |= (1ULL << height[col]);
height[col]++;
```

### 3. get_valid_moves(): 去堆分配

返回类型从 `std::vector<int>` 改为栈上固定大小的 `ValidMoves` 结构体：

```cpp
struct ValidMoves {
    int moves[7];
    int count = 0;
    // 提供 begin()/end() 支持 range-for
};
```

在 MCTS selection 循环中，每次 playout 的每一层都要调用 `get_valid_moves()`，消除堆分配避免了大量 `new`/`delete` 开销。

### 4. is_full(): O(COLS) -> O(1)

维护 `n_pieces` 计数器，`is_full()` 变为 `return n_pieces == 42`。

## Python I/O 兼容

`BatchedMCTS` 通过 `memcpy` 在 Python numpy 数组和 C++ 之间传递 `int8_t board[6][7]`。为保持兼容：

- `board` 数组保留，`step()` 中实时同步
- `sync_from_board()`: `memcpy` 导入后重建 bitboard 状态
- `sync_to_board()`: 将 bitboard 写回 board 数组（`flip()` 中使用）

调用链：

```
Python numpy -> memcpy -> board[] -> sync_from_board() -> bitboard
                                                             |
                          board[] <- step() 实时同步 <--------+
                            |
                          memcpy -> Python numpy
```

## 涉及的文件

| 文件 | 改动 |
|------|------|
| `src/cpp/Connect4.h` | Bitboard 重写 + ValidMoves + n_pieces |
| `src/cpp/MCTS.h` | 适配 ValidMoves；Dirichlet 噪声改为栈上数组 |
| `src/cpp/BatchedMCTS.h` | memcpy 后调用 sync_from_board() |
| `src/cpp/GameContext.h` | get_valid_moves() concept 约束放宽 |
