#pragma once
#include "GameContext.h"
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>

namespace AlphaZero
{
    /**
     * Othello（黑白棋）游戏逻辑，满足 MCTSGame concept。
     *
     * 使用 bitboard 实现，合法步检测和翻子均为 O(1) 位运算。
     *
     * Bitboard 布局（flat mapping）：
     *   bit i = row i/8, col i%8
     *   bit 0  = (0,0),  bit 7  = (0,7)
     *   bit 56 = (7,0),  bit 63 = (7,7)
     *
     * bb[0] = 玩家 1 (Black, turn=+1) 的棋子
     * bb[1] = 玩家 -1 (White, turn=-1) 的棋子
     *
     * 动作空间：0-63 = 棋盘位置, 64 = pass
     */
    class Othello
    {
    public:
        struct Traits
        {
            static constexpr int ROWS = 8;
            static constexpr int COLS = 8;
            static constexpr int ACTION_SIZE = 65;                ///< 64 squares + 1 pass
            static constexpr int BOARD_SIZE = ROWS * COLS;        ///< 64
            static constexpr std::array<int, 2> BOARD_SHAPE = {ROWS, COLS};
            static constexpr int NUM_SYMMETRIES = 8;              ///< D4 group
        };

        static constexpr int PASS_ACTION = 64;

        // 边缘掩码：防止位移时列方向环绕
        static constexpr uint64_t NOT_A_FILE = 0xFEFEFEFEFEFEFEFEULL; ///< 排除 col 0
        static constexpr uint64_t NOT_H_FILE = 0x7F7F7F7F7F7F7F7FULL; ///< 排除 col 7

        int8_t board[Traits::ROWS][Traits::COLS]; ///< 显示用棋盘（-1/0/1 = White/空/Black）
        PlayerSign turn;                           ///< 当前落子方（+1 Black 或 -1 White）

        uint64_t bb[2];                 ///< 两个玩家的 bitboard
        int n_pieces;                   ///< 棋盘上的总棋子数
        int consecutive_passes;         ///< 连续 pass 计数（≥2 则终局）
        PlayerIndex last_player_idx;    ///< 上一步落子的玩家索引（0 或 1），-1 表示无

        Othello() { reset(); }

        /// 重置为标准 Othello 初始局面（中央 4 子），Black 先手
        void reset()
        {
            std::memset(board, 0, sizeof(board));
            turn = 1;
            // 标准初始局面：(3,3)=W, (3,4)=B, (4,3)=B, (4,4)=W
            board[3][3] = -1; board[3][4] = 1;
            board[4][3] = 1;  board[4][4] = -1;

            bb[0] = (1ULL << 28) | (1ULL << 35);  // Black: (3,4)=28, (4,3)=35
            bb[1] = (1ULL << 27) | (1ULL << 36);  // White: (3,3)=27, (4,4)=36
            n_pieces = 4;
            consecutive_passes = 0;
            last_player_idx = -1;
        }

        // ======== I/O 接口（MCTSGame concept 要求）========

        [[nodiscard]] const int8_t *board_data() const { return &board[0][0]; }
        [[nodiscard]] int get_turn() const { return turn; }
        void set_turn(int t) { turn = t; }

        void import_board(const int8_t *src)
        {
            std::memcpy(board, src, sizeof(board));
            sync_from_board();
        }

        // ======== 内部同步方法 ========

        /// 从 board 数组重建 bitboard 状态
        void sync_from_board()
        {
            bb[0] = 0;
            bb[1] = 0;
            n_pieces = 0;
            for (int r = 0; r < Traits::ROWS; ++r)
            {
                for (int c = 0; c < Traits::COLS; ++c)
                {
                    int bit = r * 8 + c;
                    if (board[r][c] == 1)
                        bb[0] |= (1ULL << bit);
                    else if (board[r][c] == -1)
                        bb[1] |= (1ULL << bit);
                }
            }
            n_pieces = std::popcount(bb[0]) + std::popcount(bb[1]);
            consecutive_passes = 0;
            last_player_idx = -1;
        }

        /// 从 bitboard 重建 board 数组
        void sync_to_board()
        {
            std::memset(board, 0, sizeof(board));
            for (int i = 0; i < 64; ++i)
            {
                if (bb[0] & (1ULL << i))
                    board[i / 8][i % 8] = 1;
                else if (bb[1] & (1ULL << i))
                    board[i / 8][i % 8] = -1;
            }
        }

        // ======== Bitboard 方向位移 ========

        /**
         * 将 bitboard 沿 8 个方向之一位移，自动应用边缘掩码防止列环绕。
         *
         * 方向编号：0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
         */
        static uint64_t shift_dir(uint64_t b, int dir)
        {
            switch (dir)
            {
                case 0: return b >> 8;                    // N
                case 1: return (b >> 7) & NOT_A_FILE;     // NE
                case 2: return (b << 1) & NOT_A_FILE;     // E
                case 3: return (b << 9) & NOT_A_FILE;     // SE
                case 4: return b << 8;                    // S
                case 5: return (b << 7) & NOT_H_FILE;     // SW
                case 6: return (b >> 1) & NOT_H_FILE;     // W
                case 7: return (b >> 9) & NOT_H_FILE;     // NW
                default: return 0;
            }
        }

        // ======== 核心 Othello 逻辑 ========

        /**
         * 计算当前落子方的所有合法放置位置（bitboard 表示）。
         * 对每个方向：从己方棋子出发，沿对手棋子链延伸，链尾的空位为合法位置。
         */
        [[nodiscard]] uint64_t compute_valid_positions() const
        {
            PlayerIndex pidx = turn == 1 ? 0 : 1;
            uint64_t own = bb[pidx];
            uint64_t opp = bb[1 - pidx];
            uint64_t empty = ~(own | opp);
            uint64_t valid = 0;

            for (int d = 0; d < 8; ++d)
            {
                uint64_t candidates = shift_dir(own, d) & opp;
                for (int i = 0; i < 5; ++i)
                    candidates |= shift_dir(candidates, d) & opp;
                valid |= shift_dir(candidates, d) & empty;
            }
            return valid;
        }

        /**
         * 计算在指定位置落子后需要翻转的对手棋子（bitboard 表示）。
         * 对每个方向：从落子位置出发，沿对手棋子链行走，直到碰到己方棋子则翻转整条链。
         */
        [[nodiscard]] uint64_t compute_flips(int pos) const
        {
            PlayerIndex pidx = turn == 1 ? 0 : 1;
            uint64_t own = bb[pidx];
            uint64_t opp = bb[1 - pidx];
            uint64_t placed = 1ULL << pos;
            uint64_t flipped = 0;

            for (int d = 0; d < 8; ++d)
            {
                uint64_t candidates = 0;
                uint64_t sq = shift_dir(placed, d);
                while (sq & opp)
                {
                    candidates |= sq;
                    sq = shift_dir(sq, d);
                }
                if (sq & own)
                    flipped |= candidates;
            }
            return flipped;
        }

        // ======== 游戏逻辑 ========

        /**
         * 执行动作：放置棋子并翻转，或 pass。
         * @param action 0-63 = 棋盘位置, 64 = pass
         */
        void step(int action)
        {
            if (action == PASS_ACTION)
            {
                consecutive_passes++;
                turn = -turn;
                return;
            }

            PlayerIndex pidx = turn == 1 ? 0 : 1;
            uint64_t placed = 1ULL << action;
            uint64_t flips = compute_flips(action);

            bb[pidx] |= placed | flips;
            bb[1 - pidx] &= ~flips;

            // 更新 board 数组
            int r = action / 8, c = action % 8;
            board[r][c] = static_cast<int8_t>(turn);
            for (uint64_t f = flips; f; f &= f - 1)
            {
                int bit = std::countr_zero(f);
                board[bit / 8][bit % 8] = static_cast<int8_t>(turn);
            }

            n_pieces++;
            consecutive_passes = 0;
            last_player_idx = pidx;
            turn = -turn;
        }

        /**
         * 判断是否终局。
         * 终局条件：棋盘满 或 连续两次 pass（双方均无合法步）。
         */
        [[nodiscard]] bool is_game_over() const
        {
            return n_pieces == 64 || consecutive_passes >= 2;
        }

        /**
         * 检查获胜方。
         * @return 1（Black 赢）、-1（White 赢）、0（未结束或平局）
         */
        [[nodiscard]] int check_winner() const
        {
            if (!is_game_over()) return 0;
            int p1 = std::popcount(bb[0]);
            int p2 = std::popcount(bb[1]);
            if (p1 > p2) return 1;
            if (p2 > p1) return -1;
            return 0;
        }

        /**
         * 获取合法动作列表。
         * - 终局 → 空列表
         * - 有合法放置 → 返回放置位置列表
         * - 无合法放置 → 返回 [64]（必须 pass）
         */
        [[nodiscard]] ValidMoves<Traits::ACTION_SIZE> get_valid_moves() const
        {
            ValidMoves<Traits::ACTION_SIZE> moves;
            if (is_game_over()) return moves;

            uint64_t valid = compute_valid_positions();
            if (valid == 0)
            {
                moves.moves[moves.count++] = PASS_ACTION;
                return moves;
            }
            for (uint64_t v = valid; v; v &= v - 1)
                moves.moves[moves.count++] = std::countr_zero(v);
            return moves;
        }

        /// 用于 MCTS 终局检测：终局即视为 "full"
        [[nodiscard]] bool is_full() const
        {
            return is_game_over();
        }

        // ======== 对称变换（D4 群：8 种对称）========

        /**
         * 坐标变换：对 (r,c) 应用 sym_id 对应的变换。
         *
         * 0=恒等, 1=顺时针90°, 2=旋转180°, 3=顺时针270°,
         * 4=水平翻转, 5=垂直翻转, 6=主对角线翻转, 7=副对角线翻转
         */
        static void transform_coord(int sym_id, int r, int c, int &nr, int &nc)
        {
            switch (sym_id)
            {
                case 0: nr = r;     nc = c;     break;
                case 1: nr = c;     nc = 7 - r; break;
                case 2: nr = 7 - r; nc = 7 - c; break;
                case 3: nr = 7 - c; nc = r;     break;
                case 4: nr = r;     nc = 7 - c; break;
                case 5: nr = 7 - r; nc = c;     break;
                case 6: nr = c;     nc = r;     break;
                case 7: nr = 7 - c; nc = 7 - r; break;
                default: nr = r;    nc = c;     break;
            }
        }

        /// 对 bitboard 应用对称变换
        static uint64_t transform_bb(uint64_t b, int sym_id)
        {
            if (sym_id == 0) return b;
            uint64_t result = 0;
            for (uint64_t bits = b; bits; bits &= bits - 1)
            {
                int i = std::countr_zero(bits);
                int nr, nc;
                transform_coord(sym_id, i / 8, i % 8, nr, nc);
                result |= (1ULL << (nr * 8 + nc));
            }
            return result;
        }

        /**
         * 对棋盘应用对称变换。
         * @param sym_id 对称变换 ID (0-7)
         */
        void apply_symmetry(int sym_id)
        {
            if (sym_id == 0) return;
            bb[0] = transform_bb(bb[0], sym_id);
            bb[1] = transform_bb(bb[1], sym_id);
            sync_to_board();
        }

        /// 对称变换的逆映射
        static constexpr int inverse_sym(int sym_id)
        {
            // 0→0, 1→3, 2→2, 3→1, 4→4, 5→5, 6→6, 7→7
            constexpr int inv[8] = {0, 3, 2, 1, 4, 5, 6, 7};
            return inv[sym_id];
        }

        /**
         * 对 policy 数组应用对称逆变换。
         * policy[0..63] = 棋盘位置概率, policy[64] = pass 概率（不变）。
         */
        static void inverse_symmetry_policy(int sym_id,
                                            std::array<float, Traits::ACTION_SIZE> &policy)
        {
            if (sym_id == 0) return;
            int inv = inverse_sym(sym_id);
            std::array<float, Traits::ACTION_SIZE> temp;
            for (int i = 0; i < 64; ++i)
            {
                int nr, nc;
                transform_coord(inv, i / 8, i % 8, nr, nc);
                temp[nr * 8 + nc] = policy[i];
            }
            temp[PASS_ACTION] = policy[PASS_ACTION];
            policy = temp;
        }
    };
}
