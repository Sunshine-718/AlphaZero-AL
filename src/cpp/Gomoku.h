#pragma once
#include "GameContext.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace AlphaZero
{
class Gomoku
    {
    public:
        static constexpr int NUM_SYMMETRIES = 8;

        Gomoku(int board_size = 15, int n_in_row = 5)
        {
            set_params(board_size, n_in_row);
        }

        void set_params(int board_size, int n_in_row)
        {
            validate_config(board_size, n_in_row);
            board_size_ = board_size;
            n_in_row_ = n_in_row;
            board_.assign(action_size(), 0);
            reset();
        }

        void reset()
        {
            std::fill(board_.begin(), board_.end(), 0);
            turn_ = 1;
            n_pieces_ = 0;
            last_action_ = -1;
            last_player_ = 0;
            winner_ = 0;
            done_ = false;
        }

        [[nodiscard]] int board_size() const { return board_size_; }
        [[nodiscard]] int rows() const { return board_size_; }
        [[nodiscard]] int cols() const { return board_size_; }
        [[nodiscard]] int n_in_row() const { return n_in_row_; }
        [[nodiscard]] int action_size() const { return board_size_ * board_size_; }
        [[nodiscard]] int num_symmetries() const { return NUM_SYMMETRIES; }

        [[nodiscard]] const int8_t *board_data() const { return board_.data(); }
        [[nodiscard]] int get_turn() const { return turn_; }
        void set_turn(int t)
        {
            if (t != 1 && t != -1)
                throw std::runtime_error("turn must be 1 or -1");
            turn_ = t;
        }

        void import_board(const int8_t *src)
        {
            std::memcpy(board_.data(), src, static_cast<size_t>(action_size()) * sizeof(int8_t));
            sync_from_board();
        }

        void step(int action)
        {
            if (done_)
                throw std::runtime_error("game is already finished");
            if (action < 0 || action >= action_size())
                throw std::runtime_error("action out of range");
            if (board_[action] != 0)
                throw std::runtime_error("cell is already occupied");

            board_[action] = static_cast<int8_t>(turn_);
            n_pieces_++;
            last_action_ = action;
            last_player_ = turn_;

            // Incremental win check: only inspect lines passing through the latest move.
            if (has_line_from(action, last_player_))
            {
                winner_ = last_player_;
                done_ = true;
            }
            else if (n_pieces_ == action_size())
            {
                winner_ = 0;
                done_ = true;
            }

            turn_ = -turn_;
        }

        [[nodiscard]] int check_winner() const { return winner_; }
        [[nodiscard]] bool is_full() const { return n_pieces_ == action_size(); }
        [[nodiscard]] bool is_done() const { return done_; }
        [[nodiscard]] int get_last_action() const { return last_action_; }

        [[nodiscard]] std::vector<int> get_valid_moves() const
        {
            std::vector<int> moves;
            moves.reserve(static_cast<size_t>(action_size() - n_pieces_));
            for (int i = 0; i < action_size(); ++i)
                if (board_[i] == 0)
                    moves.push_back(i);
            return moves;
        }

        [[nodiscard]] int coord_to_action(int row, int col) const
        {
            if (!in_bounds(row, col))
                throw std::runtime_error("row/col out of range");
            return row * board_size_ + col;
        }

        [[nodiscard]] int inverse_symmetry_action(int sym_id, int action) const
        {
            if (action < 0 || action >= action_size())
                throw std::runtime_error("action out of range");
            if (sym_id < 0 || sym_id >= NUM_SYMMETRIES)
                throw std::runtime_error("invalid symmetry id");
            int row = action / board_size_;
            int col = action % board_size_;
            int nr = row;
            int nc = col;
            transform_coord(sym_id, row, col, nr, nc);
            return nr * board_size_ + nc;
        }

        void apply_symmetry(int sym_id)
        {
            if (sym_id < 0 || sym_id >= NUM_SYMMETRIES)
                throw std::runtime_error("invalid symmetry id");
            if (sym_id == 0)
                return;

            std::vector<int8_t> new_board(static_cast<size_t>(action_size()), 0);
            for (int r = 0; r < board_size_; ++r)
            {
                for (int c = 0; c < board_size_; ++c)
                {
                    int nr = r;
                    int nc = c;
                    transform_coord(sym_id, r, c, nr, nc);
                    new_board[nr * board_size_ + nc] = board_[r * board_size_ + c];
                }
            }
            board_.swap(new_board);

            if (last_action_ >= 0)
            {
                const int row = last_action_ / board_size_;
                const int col = last_action_ % board_size_;
                int nr = row;
                int nc = col;
                transform_coord(sym_id, row, col, nr, nc);
                last_action_ = nr * board_size_ + nc;
            }
        }

        void sync_from_board()
        {
            int p1 = 0;
            int p2 = 0;
            n_pieces_ = 0;
            last_action_ = -1;
            last_player_ = 0;
            winner_ = 0;
            done_ = false;

            for (int i = 0; i < action_size(); ++i)
            {
                const int8_t v = board_[i];
                if (v == 1)
                {
                    p1++;
                    n_pieces_++;
                    last_action_ = i;
                    last_player_ = 1;
                }
                else if (v == -1)
                {
                    p2++;
                    n_pieces_++;
                    last_action_ = i;
                    last_player_ = -1;
                }
                else if (v != 0)
                {
                    throw std::runtime_error("board values must be -1, 0, or 1");
                }
            }

            if (p1 == p2)
                turn_ = 1;
            else if (p1 == p2 + 1)
                turn_ = -1;
            else
                turn_ = (n_pieces_ % 2 == 0) ? 1 : -1;

            winner_ = find_winner_full_scan();
            done_ = (winner_ != 0) || (n_pieces_ == action_size());
        }

    private:
        int board_size_ = 15;
        int n_in_row_ = 5;

        std::vector<int8_t> board_;
        PlayerSign turn_ = 1;
        int n_pieces_ = 0;
        int last_action_ = -1;
        PlayerSign last_player_ = 0;
        int winner_ = 0;
        bool done_ = false;

        static void validate_config(int board_size, int n_in_row)
        {
            if (board_size <= 0)
                throw std::runtime_error("board_size must be positive");
            if (n_in_row <= 1)
                throw std::runtime_error("n_in_row must be >= 2");
            if (n_in_row > board_size)
                throw std::runtime_error("n_in_row must be <= board size");
        }

        [[nodiscard]] bool in_bounds(int row, int col) const
        {
            return row >= 0 && row < board_size_ && col >= 0 && col < board_size_;
        }

        [[nodiscard]] int at(int row, int col) const
        {
            return static_cast<int>(board_[row * board_size_ + col]);
        }

        [[nodiscard]] int count_direction(int row, int col, int dr, int dc, int player) const
        {
            int count = 0;
            int r = row + dr;
            int c = col + dc;
            while (in_bounds(r, c) && at(r, c) == player)
            {
                count++;
                r += dr;
                c += dc;
            }
            return count;
        }

        [[nodiscard]] bool has_line_from(int action, int player) const
        {
            const int row = action / board_size_;
            const int col = action % board_size_;

            static constexpr int DR[4] = {1, 0, 1, 1};
            static constexpr int DC[4] = {0, 1, 1, -1};

            for (int i = 0; i < 4; ++i)
            {
                const int forward = count_direction(row, col, DR[i], DC[i], player);
                const int backward = count_direction(row, col, -DR[i], -DC[i], player);
                if (1 + forward + backward >= n_in_row_)
                    return true;
            }
            return false;
        }

        [[nodiscard]] int find_winner_full_scan() const
        {
            for (int i = 0; i < action_size(); ++i)
            {
                int8_t v = board_[i];
                if (v != 0 && has_line_from(i, v))
                    return v;
            }
            return 0;
        }

        void transform_coord(int sym_id, int r, int c, int &nr, int &nc) const
        {
            // Square board: full D4 group (8 symmetries)
            const int n = board_size_;
            switch (sym_id)
            {
            case 0: nr = r;          nc = c;          break; // I
            case 1: nr = c;          nc = n - 1 - r;  break; // R90
            case 2: nr = n - 1 - r;  nc = n - 1 - c;  break; // R180
            case 3: nr = n - 1 - c;  nc = r;          break; // R270
            case 4: nr = r;          nc = n - 1 - c;  break; // horizontal mirror
            case 5: nr = n - 1 - r;  nc = c;          break; // vertical mirror
            case 6: nr = c;          nc = r;          break; // main diagonal
            case 7: nr = n - 1 - c;  nc = n - 1 - r;  break; // anti diagonal
            default:
                throw std::runtime_error("invalid symmetry id");
            }
        }
    };
}
