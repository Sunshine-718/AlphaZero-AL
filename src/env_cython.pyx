# cython: language_level=3
# distutils: language = c++
"""env_cython.pyx — 高性能 Connect-Four 环境 (增量检查优化版)
-------------------------------------------------
· 引入增量检查机制 (Incremental Check)，大幅提升 check_winner 速度。
· 优化了 valid_move 和 check_full 的检测逻辑。
· 兼容 Pickle，支持多进程。
"""

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Env:
    """6*7 Connect-Four board. 玩家棋子: 1 / -1, 空: 0"""

    # —— 私有成员 ——
    cdef np.ndarray _board          # 6×7 float32 ndarray
    cdef int        _turn           # 1 / -1 当前执子方
    cdef int        _rows
    cdef int        _cols
    
    # [新增] 记录最后落子的坐标，用于增量检查
    cdef int        _last_r
    cdef int        _last_c

    # ===== Pickle 协议 =====
    def __getstate__(self):
        """返回纯 Python 对象，包含新增的 last_r/c"""
        return (
            np.asarray(self._board, dtype=np.float32).copy(), 
            int(self._turn),
            int(self._last_r),
            int(self._last_c)
        )

    def __setstate__(self, state):
        board, turn, last_r, last_c = state
        self._board = np.asarray(board, dtype=np.float32).copy()
        self._turn  = int(turn)
        self._last_r = int(last_r)
        self._last_c = int(last_c)

    # ===== 只读属性 =====
    property board:
        def __get__(self):
            return np.asarray(self._board, dtype=np.float32).copy()

    # ===== 构造与初始化 =====
    def __cinit__(self, object board=None):
        if board is not None:
            self._board = np.asarray(board, dtype=np.float32).copy()
            self._rows = self._board.shape[0]
            self._cols = self._board.shape[1]
        else:
            self._rows = 6
            self._cols = 7
            self._board = np.zeros((self._rows, self._cols), dtype=np.float32)

        # 初始化最后落子位置为 -1 (表示开局无落子)
        self._last_r = -1
        self._last_c = -1

        # 重新计算当前执子方
        cdef int i, j, count = 0
        for i in range(self._rows):
            for j in range(self._cols):
                if self._board[i, j] != 0:
                    count += 1
        self._turn = 1 if count % 2 == 0 else -1

    property turn:
        def __get__(self):
            return self._turn
        def __set__(self, value):
            self._turn = value

    cpdef void reset(self):
        self._board[:, :] = 0.
        self._turn = 1
        self._last_r = -1
        self._last_c = -1

    cpdef object copy(self):
        cdef Env env_copy = Env()
        env_copy._board = np.copy(self._board)
        env_copy._turn  = self._turn
        env_copy._last_r = self._last_r
        env_copy._last_c = self._last_c
        return env_copy

    # ===== 游戏逻辑 (优化版) =====
    cpdef bint check_full(self):
        # 优化：只需要检查最上面一行是否有空位
        cdef float[:, :] board = self._board
        cdef int c
        for c in range(self._cols):
            if board[0, c] == 0:
                return False
        return True

    cpdef int check_winner(self):
        """增量检查：仅检查以 (_last_r, _last_c) 为中心的 4 条线"""
        # 如果还没落过子，直接返回 0
        if self._last_r == -1:
            return 0

        cdef float[:, :] board = self._board
        cdef int r = self._last_r
        cdef int c = self._last_c
        cdef int rows = self._rows
        cdef int cols = self._cols
        cdef float player = board[r, c]

        if player == 0: return 0

        cdef int count, i, nr, nc
        
        # 4个方向向量: (dr, dc) -> 水平, 垂直, 对角\, 对角/
        cdef int dr[4]
        cdef int dc[4]
        dr[:] = [0, 1, 1, 1]
        dc[:] = [1, 0, 1, -1]

        for i in range(4):
            count = 1  # 包含当前子
            
            # 向正方向延伸
            nr, nc = r + dr[i], c + dc[i]
            while 0 <= nr < rows and 0 <= nc < cols and board[nr, nc] == player:
                count += 1
                nr += dr[i]
                nc += dc[i]

            # 向负方向延伸
            nr, nc = r - dr[i], c - dc[i]
            while 0 <= nr < rows and 0 <= nc < cols and board[nr, nc] == player:
                count += 1
                nr -= dr[i]
                nc -= dc[i]
            
            if count >= 4:
                return <int>player

        return 0

    cpdef bint done(self):
        # 只要有人赢了或者棋盘满了，就结束
        # 优先检查 check_winner (速度快)，再检查 check_full
        return (self.check_winner() != 0) or self.check_full()

    cpdef list valid_move(self):
        # 优化：仅检查顶层是否为空
        return [i for i in range(self._cols) if self._board[0, i] == 0]

    cpdef list valid_mask(self):
        # 优化：仅检查顶层
        return [self._board[0, i] == 0 for i in range(self._cols)]

    cpdef void switch_turn(self):
        self._turn = -1 if self._turn == 1 else 1

    cpdef bint place(self, int action):
        """尝试在第 *action* 列落子。成功返回 True 并更新 _last_r/c。"""
        cdef float[:, :] board = self._board
        cdef int row
        
        # 从底部向上寻找空位
        for row in range(self._rows - 1, -1, -1):
            if board[row, action] == 0:
                board[row, action] = self._turn
                # [关键] 更新最后落子位置，供增量检查使用
                self._last_r = row
                self._last_c = action
                return True
        return False

    cpdef void step(self, int action):
        if self.place(action):
            self.switch_turn()

    cpdef np.ndarray current_state(self):
        """返回网络输入格式 (1, 3, 6, 7)"""
        cdef np.ndarray[np.float32_t, ndim=4] state = np.zeros((1, 3, 6, 7), dtype=np.float32)
        state[0, 0] = self._board == 1
        state[0, 1] = self._board == -1
        state[0, 2][:, :] = 1. if self._turn == 1 else -1.
        return state

    cpdef void show(self):
        cdef float[:, :] board = self._board
        cdef int r, c
        print('=' * 20)
        for r in range(6):
            row_str = []
            for c in range(7):
                val = <int>board[r, c]
                row_str.append('_' if val == 0 else ('X' if val == 1 else 'O'))
            print(' '.join(row_str))
        print('0 1 2 3 4 5 6')
        print('=' * 20)

    cpdef int winPlayer(self):
        return self.check_winner()

    # ===== 翻转相关 =====
    cpdef Env flip(self, bint inplace=False):
        """水平翻转棋盘。注意：必须同步更新 _last_c，否则增量检查会出错。"""
        cdef Env target = self if inplace else self.copy()
        target._board = target._board[:, ::-1].copy()
        
        # [关键] 如果最后一步存在，翻转后列号也会变化
        if target._last_c != -1:
            target._last_c = target._cols - 1 - target._last_c
            
        return target

    cpdef int flip_action(self, int col):
        """给定原列号，返回水平翻转后的列号（保留原有的对称性检测逻辑）。"""
        cdef float[:, :] board = self._board
        cdef int rows = board.shape[0]
        cdef int cols = board.shape[1]
        cdef int r, c
        cdef bint symmetric = True

        for r in range(rows):
            for c in range(cols // 2):
                if board[r, c] != board[r, cols - 1 - c]:
                    symmetric = False
                    break
            if not symmetric:
                break

        return col if symmetric else cols - 1 - col

    cpdef tuple random_flip(self, double p=0.5):
        """以概率 *p* 随机水平翻转，返回 (env_copy, flipped_flag)。"""
        cdef Env env_copy = self.copy()
        cdef bint flipped = False
        if np.random.rand() < p:
            env_copy = env_copy.flip(inplace=True)
            flipped = True
        return env_copy, flipped