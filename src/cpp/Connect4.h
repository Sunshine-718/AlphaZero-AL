#ifndef B5E525A8_D630_47C2_BA60_1A8D4D970BF6
#define B5E525A8_D630_47C2_BA60_1A8D4D970BF6
#pragma once
#include "GameContext.h"
#include <array>
#include <cstdint>
#include <cstring>

namespace AlphaZero
{
    /*
     * Bitboard 布局 (每列 7 位 = 6 行 + 1 哨兵位):
     *
     *   col 0   col 1   col 2   col 3   col 4   col 5   col 6
     *     6      13      20      27      34      41      48    ← 哨兵位
     *     5      12      19      26      33      40      47    ← 顶行 (row 0)
     *     4      11      18      25      32      39      46
     *     3      10      17      24      31      38      45
     *     2       9      16      23      30      37      44
     *     1       8      15      22      29      36      43
     *     0       7      14      21      28      35      42    ← 底行 (row 5)
     *
     * board[row][col] 对应 bit = col * 7 + (ROWS - 1 - row)
     *
     * bb[0] = 玩家 1 (turn=1) 的棋子
     * bb[1] = 玩家 -1 (turn=-1) 的棋子
     */
    class Connect4
    {
    public:
        struct Traits
        {
            static constexpr int ROWS = 6;
            static constexpr int COLS = 7;
            static constexpr int ACTION_SIZE = 7;
            static constexpr int BOARD_SIZE = ROWS * COLS; // 42
            static constexpr std::array<int, 2> BOARD_SHAPE = {ROWS, COLS};
        };

        // 为了与 BatchedMCTS memcpy I/O 兼容，保留 board 数组
        // 但仅在需要导出时才同步（lazy）
        int8_t board[Traits::ROWS][Traits::COLS];
        int turn;

        // Bitboard 核心状态
        uint64_t bb[2];              // 两个玩家的棋子位置
        int height[Traits::COLS];    // 每列下一个可用位的 bit 索引
        int n_pieces;                // 棋盘上的总棋子数
        int last_player_idx;         // 上一步落子的玩家索引 (0 或 1), -1 表示没有

        Connect4() { reset(); }

        void reset()
        {
            std::memset(board, 0, sizeof(board));
            turn = 1;
            bb[0] = 0;
            bb[1] = 0;
            n_pieces = 0;
            last_player_idx = -1;
            for (int c = 0; c < Traits::COLS; ++c)
            {
                height[c] = c * 7; // 每列底部的 bit 位置
            }
        }

        // ======== I/O 接口 (MCTSGame concept 要求) ========

        [[nodiscard]] const int8_t *board_data() const { return &board[0][0]; }
        [[nodiscard]] int get_turn() const { return turn; }
        void set_turn(int t) { turn = t; }

        // 从 Python memcpy 导入棋盘数据并重建 bitboard
        void import_board(const int8_t *src)
        {
            std::memcpy(board, src, sizeof(board));
            sync_from_board();
        }

        // ======== 内部同步方法 ========

        // 从 board 数组重建 bitboard 状态（用于从 Python memcpy 导入后）
        void sync_from_board()
        {
            bb[0] = 0;
            bb[1] = 0;
            n_pieces = 0;
            last_player_idx = -1;
            for (int c = 0; c < Traits::COLS; ++c)
            {
                height[c] = c * 7;
                for (int r = Traits::ROWS - 1; r >= 0; --r)
                {
                    if (board[r][c] != 0)
                    {
                        int player_idx = board[r][c] == 1 ? 0 : 1;
                        bb[player_idx] |= (1ULL << height[c]);
                        height[c]++;
                        n_pieces++;
                    }
                    else
                    {
                        break; // 碰到空格，这列的棋子到此为止
                    }
                }
            }
        }

        // 将 bitboard 状态写回 board 数组（用于导出到 Python）
        void sync_to_board()
        {
            std::memset(board, 0, sizeof(board));
            for (int c = 0; c < Traits::COLS; ++c)
            {
                int base = c * 7;
                for (int bit = base; bit < height[c]; ++bit)
                {
                    int row = Traits::ROWS - 1 - (bit - base);
                    if (bb[0] & (1ULL << bit))
                        board[row][c] = 1;
                    else
                        board[row][c] = -1;
                }
            }
        }

        // ======== 游戏逻辑 ========

        void step(int col)
        {
            int player_idx = turn == 1 ? 0 : 1;
            uint64_t move = 1ULL << height[col];
            bb[player_idx] |= move;

            // 同步到 board 数组
            int row = Traits::ROWS - 1 - (height[col] - col * 7);
            board[row][col] = turn;

            height[col]++;
            n_pieces++;
            last_player_idx = player_idx;
            turn = -turn;
        }

        [[nodiscard]] int check_winner() const
        {
            if (last_player_idx == -1)
                return 0;

            uint64_t b = bb[last_player_idx];

            // 垂直 (|): 间距 1
            uint64_t t = b & (b >> 1);
            if (t & (t >> 2)) return last_player_idx == 0 ? 1 : -1;

            // 水平 (—): 间距 7
            t = b & (b >> 7);
            if (t & (t >> 14)) return last_player_idx == 0 ? 1 : -1;

            // 对角线 (\): 间距 6
            t = b & (b >> 6);
            if (t & (t >> 12)) return last_player_idx == 0 ? 1 : -1;

            // 对角线 (/): 间距 8
            t = b & (b >> 8);
            if (t & (t >> 16)) return last_player_idx == 0 ? 1 : -1;

            return 0;
        }

        [[nodiscard]] ValidMoves<Traits::ACTION_SIZE> get_valid_moves() const
        {
            ValidMoves<Traits::ACTION_SIZE> moves;
            // 哨兵位掩码：如果 height[c] 到达了哨兵位 (c*7 + 6)，说明该列已满
            for (int c = 0; c < Traits::COLS; ++c)
            {
                if (height[c] < c * 7 + Traits::ROWS)
                {
                    moves.moves[moves.count++] = c;
                }
            }
            return moves;
        }

        [[nodiscard]] bool is_full() const
        {
            return n_pieces == Traits::ROWS * Traits::COLS;
        }

        void flip()
        {
            // 水平翻转 bitboard：交换列的位置
            // col c 的 bits [c*7, c*7+6] 交换到 col (6-c) 的 bits [(6-c)*7, (6-c)*7+6]
            uint64_t new_bb0 = 0, new_bb1 = 0;
            int new_height[Traits::COLS];

            for (int c = 0; c < Traits::COLS; ++c)
            {
                int src_base = c * 7;
                int dst_col = Traits::COLS - 1 - c;
                int dst_base = dst_col * 7;

                int col_height = height[c] - src_base; // 该列有多少棋子
                new_height[dst_col] = dst_base + col_height;

                for (int bit = 0; bit < col_height; ++bit)
                {
                    if (bb[0] & (1ULL << (src_base + bit)))
                        new_bb0 |= (1ULL << (dst_base + bit));
                    if (bb[1] & (1ULL << (src_base + bit)))
                        new_bb1 |= (1ULL << (dst_base + bit));
                }
            }

            bb[0] = new_bb0;
            bb[1] = new_bb1;
            std::memcpy(height, new_height, sizeof(height));

            // 同步 board 数组
            sync_to_board();
        }
    };
}

#endif /* B5E525A8_D630_47C2_BA60_1A8D4D970BF6 */
