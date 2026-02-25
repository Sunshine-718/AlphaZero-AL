#ifndef B5E525A8_D630_47C2_BA60_1A8D4D970BF6
#define B5E525A8_D630_47C2_BA60_1A8D4D970BF6
#pragma once
#include "GameContext.h"
#include <array>
#include <cstdint>
#include <cstring>

namespace AlphaZero
{
    /**
     * Connect4（四子棋）游戏逻辑，满足 MCTSGame concept。
     *
     * 使用 bitboard 实现，胜负检测 O(1)。
     *
     * Bitboard 布局（每列 7 位 = 6 行 + 1 哨兵位）：
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
     * board[row][col] 对应 bit = col × 7 + (ROWS - 1 - row)
     *
     * bb[0] = 玩家 1 (turn=1) 的棋子
     * bb[1] = 玩家 -1 (turn=-1) 的棋子
     */
    class Connect4
    {
    public:
        /// 游戏维度常量
        struct Traits
        {
            static constexpr int ROWS = 6;
            static constexpr int COLS = 7;
            static constexpr int BITS_PER_COL = ROWS + 1;      ///< 每列 bit 数（含哨兵位）
            static constexpr int ACTION_SIZE = 7;               ///< 动作空间 = 7 列
            static constexpr int BOARD_SIZE = ROWS * COLS;      ///< 棋盘元素数 = 42
            static constexpr std::array<int, 2> BOARD_SHAPE = {ROWS, COLS};
            static constexpr int NUM_SYMMETRIES = 2;            ///< 0=恒等, 1=水平翻转
        };

        /// 列掩码：col 列占用的 7 bit 位 (bit[col*7] ~ bit[col*7+6])
        static constexpr uint64_t col_mask(int col) { return 0x7FULL << (col * Traits::BITS_PER_COL); }

        int8_t board[Traits::ROWS][Traits::COLS]; ///< 显示用棋盘（-1/0/1 = P2/空/P1）
        int turn;                                  ///< 当前落子方（1 或 -1）

        uint64_t bb[2];                 ///< 两个玩家的 bitboard
        int height[Traits::COLS];       ///< 每列下一个可用 bit 的索引
        int n_pieces;                   ///< 棋盘上的总棋子数
        int last_player_idx;            ///< 上一步落子的玩家索引（0 或 1），-1 表示无

        Connect4() { reset(); }

        /// 重置为空棋盘，玩家 1 先手
        void reset()
        {
            std::memset(board, 0, sizeof(board));
            turn = 1;
            bb[0] = 0;
            bb[1] = 0;
            n_pieces = 0;
            last_player_idx = -1;
            for (int c = 0; c < Traits::COLS; ++c)
                height[c] = c * Traits::BITS_PER_COL;
        }

        // ======== I/O 接口（MCTSGame concept 要求）========

        /// 返回 board 数组的指针（供 memcpy 导出到 Python）
        [[nodiscard]] const int8_t *board_data() const { return &board[0][0]; }
        /// 获取当前落子方
        [[nodiscard]] int get_turn() const { return turn; }
        /// 设置当前落子方
        void set_turn(int t) { turn = t; }

        /**
         * 从 Python 端 memcpy 导入棋盘数据（int8 数组），并重建 bitboard 状态。
         * @param src 源数据指针，长度 BOARD_SIZE
         */
        void import_board(const int8_t *src)
        {
            std::memcpy(board, src, sizeof(board));
            sync_from_board();
        }

        // ======== 内部同步方法 ========

        /**
         * 从 board 数组重建 bitboard 状态。
         * 用于 import_board() 或 board setter 后同步内部状态。
         * 从底行向上扫描每列，遇到空格停止（Connect4 重力规则）。
         */
        void sync_from_board()
        {
            bb[0] = 0;
            bb[1] = 0;
            n_pieces = 0;
            last_player_idx = -1;
            for (int c = 0; c < Traits::COLS; ++c)
            {
                height[c] = c * Traits::BITS_PER_COL;
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
                        break;
                    }
                }
            }
            // 根据棋子数推断上一步落子方
            if (n_pieces > 0)
            {
                last_player_idx = (n_pieces % 2 == 1) ? 0 : 1;
            }
        }

        /**
         * 将 bitboard 状态写回 board 数组（用于导出到 Python 或 show()）。
         * 遍历每列的已用 bit，映射到 board[row][col]。
         */
        void sync_to_board()
        {
            std::memset(board, 0, sizeof(board));
            for (int c = 0; c < Traits::COLS; ++c)
            {
                int base = c * Traits::BITS_PER_COL;
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

        /**
         * 在指定列落子。更新 bitboard、board 数组、height、棋子计数，并切换落子方。
         * 调用方需确保该列未满（通过 get_valid_moves() 检查）。
         * @param col 落子列号 [0, 6]
         */
        void step(int col)
        {
            int player_idx = turn == 1 ? 0 : 1;
            uint64_t move = 1ULL << height[col];
            bb[player_idx] |= move;

            int row = Traits::ROWS - 1 - (height[col] - col * Traits::BITS_PER_COL);
            board[row][col] = turn;

            height[col]++;
            n_pieces++;
            last_player_idx = player_idx;
            turn = -turn;
        }

        /**
         * 检查是否有玩家获胜（bitboard O(1) 检测）。
         *
         * 对上一步落子方的 bitboard 做 4 方向（垂直、水平、两条对角线）的位移 AND 检测：
         * 如果存在连续 4 个 bit，则该方向有四子连线。
         *
         * @return 1（玩家 1 赢）、-1（玩家 -1 赢）、0（无人获胜）
         */
        [[nodiscard]] int check_winner() const
        {
            if (last_player_idx == -1)
                return 0;

            uint64_t b = bb[last_player_idx];
            int result = last_player_idx == 0 ? 1 : -1;

            // 四方向连续 4 子检测，每个方向的相邻 bit 间距不同：
            constexpr int H = Traits::BITS_PER_COL;     // 7: 水平（—）相邻列间距
            constexpr int V = 1;                         // 1: 垂直（|）相邻行间距
            constexpr int D1 = H - 1;                   // 6: 对角线（＼）
            constexpr int D2 = H + 1;                   // 8: 对角线（／）

            uint64_t t;
            t = b & (b >> V);  if (t & (t >> (2 * V)))  return result;  // 垂直
            t = b & (b >> H);  if (t & (t >> (2 * H)))  return result;  // 水平
            t = b & (b >> D1); if (t & (t >> (2 * D1))) return result;  // ＼ 对角线
            t = b & (b >> D2); if (t & (t >> (2 * D2))) return result;  // ／ 对角线

            return 0;
        }

        /**
         * 获取所有合法落子列（未满的列）。
         * @return ValidMoves 结构体，包含合法列号列表和数量
         */
        [[nodiscard]] ValidMoves<Traits::ACTION_SIZE> get_valid_moves() const
        {
            ValidMoves<Traits::ACTION_SIZE> moves;
            for (int c = 0; c < Traits::COLS; ++c)
            {
                if (height[c] < c * Traits::BITS_PER_COL + Traits::ROWS)
                    moves.moves[moves.count++] = c;
            }
            return moves;
        }

        /// 棋盘是否已满（42 颗棋子）
        [[nodiscard]] bool is_full() const
        {
            return n_pieces == Traits::ROWS * Traits::COLS;
        }

        /**
         * 对称变换：sym_id=0 恒等，sym_id=1 水平翻转。
         *
         * 水平翻转通过 bitboard 列掩码交换实现（col 0 ↔ col 6, 1 ↔ 5, 2 ↔ 4），
         * 中心列 3 不动。同步更新 height 数组和 board 数组。
         *
         * @param sym_id 对称变换 ID
         */
        void apply_symmetry(int sym_id)
        {
            if (sym_id == 0) return;

            constexpr int B = Traits::BITS_PER_COL;  // 7
            for (int p = 0; p < 2; ++p)
            {
                uint64_t src = bb[p];
                uint64_t dst = src & col_mask(3);     // 保留中心列 col 3

                // 交换对称列对：col c ↔ col (6-c)，位移差 = (6-2c) × BITS_PER_COL
                dst |= (src & col_mask(0)) << (6 * B); // col 0 → col 6
                dst |= (src & col_mask(6)) >> (6 * B); // col 6 → col 0
                dst |= (src & col_mask(1)) << (4 * B); // col 1 → col 5
                dst |= (src & col_mask(5)) >> (4 * B); // col 5 → col 1
                dst |= (src & col_mask(2)) << (2 * B); // col 2 → col 4
                dst |= (src & col_mask(4)) >> (2 * B); // col 4 → col 2

                bb[p] = dst;
            }

            constexpr int BPC = Traits::BITS_PER_COL;
            for (int c = 0; c < 3; ++c)
            {
                int mirror = Traits::COLS - 1 - c;
                int h_c = height[c] - c * BPC;
                int h_m = height[mirror] - mirror * BPC;
                height[c] = c * BPC + h_m;
                height[mirror] = mirror * BPC + h_c;
            }

            sync_to_board();
        }

        /**
         * 对 policy 数组应用对称逆变换。
         * 水平翻转是自逆的：交换 policy[i] 和 policy[COLS-1-i]。
         * @param sym_id 对称变换 ID
         * @param policy [in/out] 策略数组
         */
        static void inverse_symmetry_policy(int sym_id,
                                            std::array<float, Traits::ACTION_SIZE> &policy)
        {
            if (sym_id == 0) return;
            for (int i = 0; i < Traits::COLS / 2; ++i)
                std::swap(policy[i], policy[Traits::COLS - 1 - i]);
        }
    };
}

#endif /* B5E525A8_D630_47C2_BA60_1A8D4D970BF6 */
