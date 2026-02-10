#ifndef B5E525A8_D630_47C2_BA60_1A8D4D970BF6
#define B5E525A8_D630_47C2_BA60_1A8D4D970BF6
#pragma once
#include "Constants.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace AlphaZero
{
    class Connect4
    {
    public:
        int8_t board[Config::ROWS][Config::COLS];
        int turn;

        int last_r = -1;
        int last_c = -1;

        Connect4() { reset(); }

        void reset()
        {
            std::memset(board, 0, sizeof(board));
            turn = 1;
            last_r = -1;
            last_c = -1;
        }

    private:
        [[nodiscard]] inline bool is_on_board(int r, int c) const
        {
            return r >= 0 && r < Config::ROWS && c >= 0 && c < Config::COLS;
        }

    public:
        [[nodiscard]]std::vector<int> get_valid_moves() const
        {
            std::vector<int> moves;
            moves.reserve(Config::COLS);
            for (int c = 0; c < Config::COLS; ++c)
            {
                if (board[0][c] == 0)
                    moves.push_back(c);
            }
            return moves;
        }

        void step(int col)
        {
            for (int r = Config::ROWS - 1; r >= 0; --r)
            {
                if (board[r][col] == 0)
                {
                    board[r][col] = turn;
                    last_r = r;
                    last_c = col;
                    turn = -turn;
                    return;
                }
            }
        }

        [[nodiscard]]int check_winner() const
        {
            if (last_r == -1)
                return 0;
            int player = board[last_r][last_c];

            static const int dr[] = {0, 1, 1, 1};
            static const int dc[] = {1, 0, 1, -1};

            for (int d = 0; d < 4; ++d)
            {
                int count = 1;
                for (int step = 1; step < 4; ++step)
                {
                    int r = last_r + dr[d] * step;
                    int c = last_c + dc[d] * step;
                    if (is_on_board(r, c) && board[r][c] == player)
                    {
                        count++;
                    }
                    else
                    {
                        break;
                    }
                }
                for (int step = 1; step < 4; ++step)
                {
                    int r = last_r - dr[d] * step;
                    int c = last_c - dc[d] * step;
                    if (is_on_board(r, c) && board[r][c] == player)
                    {
                        count++;
                    }
                    else
                    {
                        break;
                    }
                }
                if (count >= 4)
                    return player;
            }
            return 0;
        }

        [[nodiscard]]bool is_full() const
        {
            for (int c = 0; c < Config::COLS; ++c)
            {
                if (board[0][c] == 0)
                    return false;
            }
            return true;
        }

        void flip()
        {
            for (int r = 0; r < Config::ROWS; ++r)
            {
                std::reverse(std::begin(board[r]), std::end(board[r]));
            }
            if (last_c != -1)
            {
                last_c = Config::COLS - 1 - last_c;
            }
        }
    };
}

#endif /* B5E525A8_D630_47C2_BA60_1A8D4D970BF6 */
