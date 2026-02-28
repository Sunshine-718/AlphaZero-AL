from src.game import Game
from src.player import Human, AlphaZeroPlayer
from src.environments import load
import torch.nn.functional as F
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QLinearGradient, QRadialGradient, QPainterPath
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSpinBox, QComboBox, QSlider,
                             QPushButton, QCheckBox, QFrame,
                             QSizePolicy, QGroupBox, QScrollArea)
import time
import torch

# ── 配置 ──────────────────────────────────────────────────────────────────────
ENV_NAME            = 'Connect4'
NETWORK_DEFAULT     = 'CNN'
MODEL_NAME          = 'AZ'
MODEL_TYPE_DEFAULT  = 'current'
ANIMATION_MS        = 30

# ── MCTS 超参数 ───────────────────────────────────────────────────────────────
N_PLAYOUT_DEFAULT   = 500       # 每步模拟次数
C_INIT              = 1.4       # PUCT 探索常数 (UCB = Q + c_puct * P * sqrt(N_parent) / (1+N_child))
C_BASE              = 1000       # PUCT 对数基底 (c_puct = c_init + log((N_parent + c_base + 1) / c_base))
FPU_REDUCTION       = 0.2       # First Play Urgency: 未访问节点的 Q 惩罚系数
ALPHA               = 0       # Dirichlet 噪声 alpha (0=禁用噪声)
NOISE_EPSILON       = 0      # 噪声混合权重 ε: prior = (1-ε)*P + ε*noise (eval 模式下强制为 0)
USE_SYMMETRY        = True      # 叶节点随机对称变换增强
CACHE_SIZE          = 10000     # LRU 置换表大小 (0=禁用)
MLH_SLOPE           = 0.1       # Moves Left Head 斜率 (0=禁用, LC0 风格: 按 child_M - parent_M 缩放)
MLH_CAP             = 0.15       # MLH 最大效果上限
MLH_THRESHOLD       = 0.2       # MLH Q 阈值: |Q| < threshold 时抑制 M_utility (0=无阈值)

PARAMS_PATH_FMT     = './params/{model_name}_{env_name}_{network}_{model_type}.pt'

# ── 颜色主题（深色） ─────────────────────────────────────────────────────────
BG          = QColor(24, 26, 31)        # 窗口底色
BOARD_BG    = QColor(30, 33, 40)        # 棋盘背景
GRID_CLR    = QColor(55, 60, 72)        # 网格线
CELL_BG     = QColor(18, 20, 26)        # 空格
RED_DARK    = QColor(200, 50,  50)
RED_LIGHT   = QColor(240, 90,  90)
YEL_DARK    = QColor(200, 170, 20)
YEL_LIGHT   = QColor(250, 210, 50)
ACCENT      = QColor(82, 139, 255)      # 蓝色高亮（最后落子标记）
TEXT_MAIN   = "#e0e4ef"
TEXT_DIM    = "#6a7089"
PANEL_BG    = "#1a1d24"
CARD_BG     = "#22262f"
BTN_NORMAL  = "#3a4055"
BTN_HOVER   = "#4a5070"
BTN_ACTIVE  = "#5263a8"

STYLESHEET = f"""
QWidget {{
    background-color: {PANEL_BG};
    color: {TEXT_MAIN};
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {CARD_BG};
    border: 1px solid #2e3240;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
    color: {TEXT_MAIN};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    color: #8899cc;
}}
QLabel {{
    color: {TEXT_MAIN};
    background: transparent;
}}
QLabel#dim {{
    color: {TEXT_DIM};
    font-size: 11px;
}}
QComboBox, QSpinBox {{
    background-color: #2a2e3a;
    border: 1px solid #3a4055;
    border-radius: 5px;
    padding: 4px 8px;
    color: {TEXT_MAIN};
    min-height: 28px;
}}
QComboBox:hover, QSpinBox:hover {{
    border: 1px solid {ACCENT.name()};
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: #2a2e3a;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    height: 14px;
    margin: -5px 0;
    background: {ACCENT.name()};
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: #6a9aff;
}}
QSlider::sub-page:horizontal {{
    background: {BTN_ACTIVE};
    border-radius: 3px;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QCheckBox {{
    color: {TEXT_MAIN};
    background: transparent;
    spacing: 5px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid #3a4055;
    border-radius: 3px;
    background-color: #2a2e3a;
}}
QCheckBox::indicator:checked {{
    background-color: {BTN_ACTIVE};
    border-color: {BTN_ACTIVE};
}}
QPushButton {{
    background-color: {BTN_NORMAL};
    color: {TEXT_MAIN};
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
    min-height: 34px;
}}
QPushButton:hover {{
    background-color: {BTN_HOVER};
}}
QPushButton:pressed {{
    background-color: {BTN_ACTIVE};
}}
QPushButton#primary {{
    background-color: {BTN_ACTIVE};
}}
QPushButton#primary:hover {{
    background-color: #6273b8;
}}
QFrame#separator {{
    color: #2e3240;
}}
QWidget#statusBar {{
    background-color: {CARD_BG};
    border: 1px solid #2e3240;
    border-radius: 8px;
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
}}
QScrollBar::handle:vertical {{
    background: #3a4055;
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: #4a5070;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
"""


class BoardWidget(QWidget):
    """仅负责绘制棋盘，不处理业务逻辑。"""
    CELL   = 72
    MARGIN = 24

    def __init__(self, env, parent=None):
        super().__init__(parent)
        self.env = env
        cols, rows = 7, 6
        w = self.CELL * cols + self.MARGIN * 2
        h = self.CELL * rows + self.MARGIN * 2
        self.setFixedSize(w, h)
        self.last_move    = None
        self.anim_row     = -1
        self.anim_col     = -1
        self.anim_color   = None

    # ── 绘制 ─────────────────────────────────────────────────────────────────
    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        self._draw_background(qp)
        self._draw_grid(qp)
        self._draw_pieces(qp)
        self._draw_last_move(qp)
        self._draw_anim_piece(qp)

    def _draw_background(self, qp):
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 16, 16)
        qp.fillPath(path, BOARD_BG)

    def _draw_grid(self, qp):
        qp.setPen(QPen(GRID_CLR, 1))
        m, c = self.MARGIN, self.CELL
        for col in range(1, 7):
            x = m + col * c
            qp.drawLine(x, m, x, m + 6 * c)
        for row in range(1, 6):
            y = m + row * c
            qp.drawLine(m, y, m + 7 * c, y)

    def _board(self):
        state = self.env.current_state()
        return (state[0, 0] - state[0, 1]).astype(int)

    def _draw_pieces(self, qp):
        board = self._board()
        for r in range(6):
            for c in range(7):
                v = board[r][c]
                cx = self.MARGIN + c * self.CELL + self.CELL // 2
                cy = self.MARGIN + r * self.CELL + self.CELL // 2
                self._draw_circle(qp, cx, cy, v)

    def _draw_circle(self, qp, cx, cy, value, radius=None):
        r = radius or (self.CELL // 2 - 8)
        if value == 0:
            qp.setBrush(CELL_BG)
            qp.setPen(QPen(GRID_CLR, 1))
            qp.drawEllipse(QPointF(cx, cy), r, r)
            return
        dark  = RED_DARK  if value == 1 else YEL_DARK
        light = RED_LIGHT if value == 1 else YEL_LIGHT
        grad = QRadialGradient(cx - r * 0.3, cy - r * 0.3, r * 1.2)
        grad.setColorAt(0, light)
        grad.setColorAt(1, dark)
        qp.setBrush(grad)
        qp.setPen(Qt.NoPen)
        qp.drawEllipse(QPointF(cx, cy), r, r)

    def _draw_last_move(self, qp):
        if self.last_move is None:
            return
        r, c = self.last_move
        cx = self.MARGIN + c * self.CELL + self.CELL // 2
        cy = self.MARGIN + r * self.CELL + self.CELL // 2
        radius = self.CELL // 2 - 8
        qp.setBrush(Qt.NoBrush)
        pen = QPen(ACCENT, 3)
        qp.setPen(pen)
        qp.drawEllipse(QPointF(cx, cy), radius + 4, radius + 4)

    def _draw_anim_piece(self, qp):
        if self.anim_row < 0 or self.anim_col < 0:
            return
        cx = self.MARGIN + self.anim_col * self.CELL + self.CELL // 2
        cy = self.MARGIN + self.anim_row * self.CELL + self.CELL // 2
        v = 1 if self.anim_color == RED_DARK else -1
        self._draw_circle(qp, cx, cy, v)

    def col_at(self, x):
        """屏幕 x 坐标 → 列号，超出棋盘返回 -1。"""
        rel = x - self.MARGIN
        if rel < 0 or rel >= self.CELL * 7:
            return -1
        return rel // self.CELL

    def find_drop_row(self, col):
        board = self._board()
        for row in range(5, -1, -1):
            if board[row][col] == 0:
                return row
        return -1


class WinRateBar(QWidget):
    """横向三色胜率条。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(20)
        self.win = self.draw = self.lose = 0.0

    def set_rates(self, win, draw, lose):
        self.win, self.draw, self.lose = win, draw, lose
        self.update()

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, w, h), h // 2, h // 2)
        qp.setClipPath(path)
        x = 0
        for ratio, color in [(self.win, RED_DARK), (self.draw, QColor(80, 90, 110)), (self.lose, YEL_DARK)]:
            pw = int(w * ratio)
            qp.fillRect(x, 0, pw, h, color)
            x += pw


class StepsBar(QWidget):
    """横向进度条：显示预测剩余步数占最大步数的比例，右侧显示整数步数。"""
    MAX_STEPS = 42

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(20)
        self.steps = 0

    def set_steps(self, steps):
        self.steps = max(0, min(int(round(steps)), self.MAX_STEPS))
        self.update()

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        text = str(self.steps)
        fm = qp.fontMetrics()
        text_w = fm.horizontalAdvance(text) + 8
        bar_w = self.width() - text_w
        h = self.height()

        # 背景圆角
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, bar_w, h), h // 2, h // 2)
        qp.setClipPath(path)
        qp.fillRect(0, 0, bar_w, h, QColor(40, 44, 55))

        # 填充
        ratio = self.steps / self.MAX_STEPS if self.MAX_STEPS else 0
        fill_w = int(bar_w * ratio)
        qp.fillRect(0, 0, fill_w, h, QColor(82, 139, 255))

        # 右侧数字
        qp.setClipping(False)
        qp.setPen(QColor(TEXT_MAIN))
        qp.drawText(bar_w + 4, 0, text_w, h, Qt.AlignVCenter | Qt.AlignLeft, text)


class StatusBar(QWidget):
    """棋盘下方状态栏：对局结果、思考时间、胜率、剩余步数。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statusBar")

        root = QHBoxLayout(self)
        root.setContentsMargins(14, 8, 14, 8)
        root.setSpacing(16)

        # 左：对局结果 + 思考时间
        left = QVBoxLayout()
        left.setSpacing(2)
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumWidth(100)
        f = self.result_label.font()
        f.setPointSize(13)
        f.setBold(True)
        self.result_label.setFont(f)
        left.addWidget(self.result_label)
        self.thinking_label = QLabel("AI 思考时间: —")
        self.thinking_label.setObjectName("dim")
        self.thinking_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self.thinking_label)
        root.addLayout(left)

        # 中：胜率
        mid = QVBoxLayout()
        mid.setSpacing(4)
        rate_row = QHBoxLayout()
        self.win_lbl  = self._rate_lbl("Win",  RED_DARK.name())
        self.draw_lbl = self._rate_lbl("Draw", "#505a6e")
        self.lose_lbl = self._rate_lbl("Lose", YEL_DARK.name())
        rate_row.addWidget(self.win_lbl)
        rate_row.addWidget(self.draw_lbl)
        rate_row.addWidget(self.lose_lbl)
        mid.addLayout(rate_row)
        self.bar = WinRateBar()
        mid.addWidget(self.bar)
        root.addLayout(mid, stretch=1)

        # 右：剩余步数
        right = QVBoxLayout()
        right.setSpacing(4)
        steps_title = QLabel("<font color='#8090b0'><b>预测剩余</b></font>")
        steps_title.setAlignment(Qt.AlignCenter)
        steps_title.setTextFormat(Qt.RichText)
        right.addWidget(steps_title)
        self.steps_bar = StepsBar()
        self.steps_bar.setMinimumWidth(110)
        right.addWidget(self.steps_bar)
        root.addLayout(right)

    @staticmethod
    def _rate_lbl(prefix, color):
        lbl = QLabel(f"<font color='{color}'><b>{prefix}</b></font><br>—%")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setTextFormat(Qt.RichText)
        return lbl

    def set_rates(self, win, draw, lose):
        self.win_lbl.setText( f"<font color='{RED_DARK.name()}'><b>Win</b></font><br>{win:.1f}%")
        self.draw_lbl.setText(f"<font color='#505a6e'><b>Draw</b></font><br>{draw:.1f}%")
        self.lose_lbl.setText(f"<font color='{YEL_DARK.name()}'><b>Lose</b></font><br>{lose:.1f}%")
        self.bar.set_rates(win / 100, draw / 100, lose / 100)

    def set_thinking(self, seconds):
        if seconds < 0:
            self.thinking_label.setText("AI 思考时间: —")
        else:
            self.thinking_label.setText(f"AI 思考时间: {seconds:.2f} s")

    def set_result(self, text, color="#e0e4ef"):
        self.result_label.setText(f"<font color='{color}'>{text}</font>")
        self.result_label.setTextFormat(Qt.RichText)

    def set_steps(self, steps):
        self.steps_bar.set_steps(steps)


class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── 游戏设置 ─────────────────────────────────────────────────────────
        game_box = QGroupBox("游戏设置")
        gl = QVBoxLayout(game_box)
        gl.setSpacing(6)

        gl.addWidget(QLabel("网络结构:"))
        self.network_cb = QComboBox()
        self.network_cb.addItems(["CNN", "ViT"])
        gl.addWidget(self.network_cb)

        gl.addWidget(QLabel("模型权重:"))
        self.model_type_cb = QComboBox()
        self.model_type_cb.addItems(["current", "best"])
        gl.addWidget(self.model_type_cb)

        gl.addWidget(QLabel("先手:"))
        self.player_cb = QComboBox()
        self.player_cb.addItems(["我先手 (X)", "AI 先手 (X)"])
        gl.addWidget(self.player_cb)

        root.addWidget(game_box)

        # ── MCTS 搜索参数 ────────────────────────────────────────────────────
        mcts_box = QGroupBox("MCTS 搜索")
        ml = QVBoxLayout(mcts_box)
        ml.setSpacing(6)

        # 模拟次数
        row = QHBoxLayout()
        row.addWidget(QLabel("模拟次数"))
        self.n_playout_spin = QSpinBox()
        self.n_playout_spin.setRange(1, 10000)
        self.n_playout_spin.setValue(N_PLAYOUT_DEFAULT)
        row.addWidget(self.n_playout_spin)
        ml.addLayout(row)

        # 滑条参数
        self.c_init_slider   = self._add_slider(ml, "c_init",    0.0, 10.0, 0.1, C_INIT)
        self.c_base_slider   = self._add_slider(ml, "c_base",    1, 100000, 100, C_BASE, decimals=0)
        self.alpha_slider    = self._add_slider(ml, "alpha",     0.0, 5.0,  0.01, ALPHA)
        self.noise_eps_slider = self._add_slider(ml, "epsilon",  0.0, 1.0,  0.05, NOISE_EPSILON)
        self.fpu_slider      = self._add_slider(ml, "fpu",       0.0, 2.0,  0.1, FPU_REDUCTION)

        # 对称增强 checkbox
        self.symmetry_check = QCheckBox("对称增强")
        self.symmetry_check.setChecked(USE_SYMMETRY)
        ml.addWidget(self.symmetry_check)

        self.cache_slider      = self._add_slider(ml, "cache",     0, 1000000, 1000, CACHE_SIZE, decimals=0)
        self.mlh_slope_slider  = self._add_slider(ml, "mlh slope", 0.0, 1.0, 0.01, MLH_SLOPE)
        self.mlh_cap_slider    = self._add_slider(ml, "mlh cap",   0.0, 1.0, 0.05, MLH_CAP)
        self.mlh_threshold_slider = self._add_slider(ml, "mlh thr", 0.0, 1.0, 0.05, MLH_THRESHOLD)

        root.addWidget(mcts_box)

    # ── 工具方法 ──────────────────────────────────────────────────────────
    def _add_slider(self, layout, label, lo, hi, step, default, decimals=2):
        """添加一行: [label] [slider] [value_label]，返回 (slider, value_label)。
        slider 内部用整数，通过 _scale 属性换算浮点值。"""
        row = QHBoxLayout()
        name_lbl = QLabel(label)
        name_lbl.setFixedWidth(60)
        row.addWidget(name_lbl)

        scale = 10 ** decimals
        slider = QSlider(Qt.Horizontal)
        slider._scale = scale
        slider.setRange(int(lo * scale), int(hi * scale))
        slider.setSingleStep(max(1, int(step * scale)))
        slider.setValue(int(default * scale))

        val_lbl = QLabel(f"{default:.{decimals}f}" if decimals else str(int(default)))
        val_lbl.setFixedWidth(52)
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        val_lbl._decimals = decimals

        def _update(v):
            if decimals:
                val_lbl.setText(f"{v / scale:.{decimals}f}")
            else:
                val_lbl.setText(str(v))
        slider.valueChanged.connect(_update)

        row.addWidget(slider, stretch=1)
        row.addWidget(val_lbl)
        layout.addLayout(row)
        return slider


class Connect4GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaZero Connect4")
        self.setStyleSheet(STYLESHEET)
        self.setWindowFlags(Qt.Window)

        # 环境
        self.env_module = load(ENV_NAME)
        self.env = self.env_module.Env()

        # 子控件
        self.board = BoardWidget(self.env)
        self.panel = ControlPanel()
        self.status_bar = StatusBar()

        # 右侧面板放入滚动区域
        scroll = QScrollArea()
        scroll.setWidget(self.panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(260)

        # 操作按钮（右下角）
        self.undo_btn = QPushButton("悔棋")
        self.undo_btn.setEnabled(False)
        self.restart_btn = QPushButton("重新开始")
        self.restart_btn.setObjectName("primary")
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.undo_btn)
        btn_row.addWidget(self.restart_btn)

        # 左列：棋盘 + 状态栏（与棋盘等宽）
        left_col = QVBoxLayout()
        left_col.setSpacing(10)
        left_col.addWidget(self.board)
        left_col.addWidget(self.status_bar)

        # 右列：设置面板(滚动) + 按钮
        right_col = QVBoxLayout()
        right_col.setSpacing(10)
        right_col.addWidget(scroll, stretch=1)
        right_col.addLayout(btn_row)

        # 顶层布局
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(16, 16, 16, 16)
        hbox.setSpacing(16)
        hbox.addLayout(left_col)
        hbox.addLayout(right_col)
        self.adjustSize()
        self.setFixedSize(self.sizeHint())

        # 动画定时器
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._step_animation)
        self.anim_target_row = -1
        self.anim_callback   = None
        self.animating       = False

        # 悔棋历史栈：每次人类落子前保存 (env_copy, last_move)
        self._history: list = []

        # 延迟重载（避免控件变更时频繁重载）
        self.reload_timer = QTimer()
        self.reload_timer.setSingleShot(True)
        self.reload_timer.timeout.connect(self._reload_and_restart)

        # 仅重载模型权重/参数，不重置棋局（不清空悔棋栈）
        self._settings_timer = QTimer()
        self._settings_timer.setSingleShot(True)
        self._settings_timer.timeout.connect(self._reload_model)

        # 玩家对象
        self.human     = Human()
        self.az_player = AlphaZeroPlayer(None, c_init=None, c_base=C_BASE,
                                         n_playout=None,
                                         alpha=ALPHA, noise_epsilon=NOISE_EPSILON,
                                         is_selfplay=0, cache_size=CACHE_SIZE,
                                         fpu_reduction=FPU_REDUCTION,
                                         use_symmetry=USE_SYMMETRY,
                                         mlh_slope=MLH_SLOPE, mlh_cap=MLH_CAP,
                                         mlh_threshold=MLH_THRESHOLD)
        self._reload_model()

        # 连接信号 — 切换网络/权重/MCTS参数 → 延迟重载模型
        _mcts_reload = lambda _: self._settings_timer.start(400)
        self.panel.network_cb.currentIndexChanged.connect(_mcts_reload)
        self.panel.model_type_cb.currentIndexChanged.connect(_mcts_reload)
        self.panel.n_playout_spin.valueChanged.connect(_mcts_reload)
        # 高级 MCTS 参数（滑条 + checkbox）
        self.panel.c_init_slider.valueChanged.connect(_mcts_reload)
        self.panel.c_base_slider.valueChanged.connect(_mcts_reload)
        self.panel.alpha_slider.valueChanged.connect(_mcts_reload)
        self.panel.noise_eps_slider.valueChanged.connect(_mcts_reload)
        self.panel.fpu_slider.valueChanged.connect(_mcts_reload)
        self.panel.symmetry_check.stateChanged.connect(_mcts_reload)
        self.panel.cache_slider.valueChanged.connect(_mcts_reload)
        self.panel.mlh_slope_slider.valueChanged.connect(_mcts_reload)
        self.panel.mlh_cap_slider.valueChanged.connect(_mcts_reload)
        self.panel.mlh_threshold_slider.valueChanged.connect(_mcts_reload)
        # 切换先手 → 重新开始
        self.panel.player_cb.currentIndexChanged.connect(lambda _: self.reload_timer.start(100))
        self.restart_btn.clicked.connect(self._reload_and_restart)
        self.undo_btn.clicked.connect(self._undo)

        self._start_game()

    # ── 模型管理 ──────────────────────────────────────────────────────────────
    def _reload_model(self):
        network    = self.panel.network_cb.currentText()
        model_type = self.panel.model_type_cb.currentText()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = getattr(self.env_module, network)(lr=0, device=device)
        net.eval()
        self.net = net
        path = PARAMS_PATH_FMT.format(model_name=MODEL_NAME, env_name=ENV_NAME,
                                      network=network, model_type=model_type)
        self.net.load(path)

        # 从 UI 读取所有 MCTS 超参数，写入 player 内部状态
        _sv = lambda s: s.value() / s._scale   # slider → float
        p = self.az_player
        p._c_base         = int(_sv(self.panel.c_base_slider))
        p._fpu_reduction  = _sv(self.panel.fpu_slider)
        p._noise_eps      = _sv(self.panel.noise_eps_slider)
        p.noise_eps_init  = _sv(self.panel.noise_eps_slider)
        p._use_symmetry   = self.panel.symmetry_check.isChecked()
        p._cache_size     = int(_sv(self.panel.cache_slider))
        p._mlh_slope      = _sv(self.panel.mlh_slope_slider)
        p._mlh_cap        = _sv(self.panel.mlh_cap_slider)
        p._mlh_threshold  = _sv(self.panel.mlh_threshold_slider)

        p.reload(self.net,
                 c_puct=_sv(self.panel.c_init_slider),
                 n_playout=self.panel.n_playout_spin.value(),
                 alpha=_sv(self.panel.alpha_slider),
                 is_self_play=0)
        p.eval()
        self.player_color = 1 if "我先手" in self.panel.player_cb.currentText() else -1

    def _reload_and_restart(self):
        self._reload_model()
        self._start_game()

    # ── 游戏流程 ──────────────────────────────────────────────────────────────
    def _start_game(self):
        self.env.reset()
        self.board.last_move  = None
        self.board.anim_row   = -1
        self.board.anim_col   = -1
        self.board.anim_color = None
        self.board.update()
        self.status_bar.set_result("")
        self.status_bar.set_thinking(-1)
        self._history.clear()
        self.undo_btn.setEnabled(False)
        self._update_winrate()
        if self.env.turn != self.player_color:
            QTimer.singleShot(100, self._ai_move)

    def _update_winrate(self):
        with torch.no_grad():
            state = self.env.current_state()
            t = torch.from_numpy(state).float().to(self.net.device).unsqueeze(0)
            if t.dim() == 5:
                t = t.squeeze(1)
            _, vl, sl = self.net(t)
            vp = F.softmax(vl, dim=-1)[0].cpu().tolist()
            sp = F.softmax(sl, dim=-1)[0].cpu()
            expected_steps = (sp * torch.arange(len(sp), dtype=torch.float32)).sum().item()
        draw = vp[0] * 100
        if self.player_color == 1:
            win, lose = vp[1] * 100, vp[2] * 100
        else:
            win, lose = vp[2] * 100, vp[1] * 100
        self.status_bar.set_rates(win, draw, lose)
        self.status_bar.set_steps(expected_steps)

    def _ai_move(self):
        if self.animating or self.env.done():
            return
        t0 = time.time()
        action, _ = self.az_player.get_action(self.env)
        self.status_bar.set_thinking(time.time() - t0)
        row = self.board.find_drop_row(action)
        if row != -1:
            self.board.last_move = (row, action)
            self._start_animation(row, action, RED_DARK if -self.player_color == 1 else YEL_DARK,
                                  lambda: self._after_move(action))

    def _after_move(self, col):
        self.env.step(col)
        self._update_winrate()
        self.board.update()
        if self.env.done():
            self._show_result()
            return
        if self.env.turn != self.player_color:
            QTimer.singleShot(60, self._ai_move)

    def _show_result(self):
        winner = self.env.winPlayer()
        if winner == self.player_color:
            self.status_bar.set_result("你赢了！", "#5cb85c")
        elif winner == -self.player_color:
            self.status_bar.set_result("你输了！", "#d9534f")
        else:
            self.status_bar.set_result("平局！", "#f0ad4e")

    # ── 动画 ──────────────────────────────────────────────────────────────────
    def _start_animation(self, target_row, col, color, callback):
        self.animating        = True
        self.anim_target_row  = target_row
        self.anim_callback    = callback
        self.board.anim_row   = -1
        self.board.anim_col   = col
        self.board.anim_color = color
        self.anim_timer.start(ANIMATION_MS)

    def _step_animation(self):
        if self.board.anim_row < self.anim_target_row:
            self.board.anim_row += 1
            self.board.update()
        else:
            self.anim_timer.stop()
            self.board.anim_row = -1
            self.board.update()
            self.animating = False
            self.anim_callback()

    def _undo(self):
        """悔棋：回退到人类上一次落子前的状态（撤销人类 + AI 各一步）。"""
        if not self._history or self.animating:
            return
        saved_env, saved_last_move = self._history.pop()
        # Cython env 的 board 属性不可直接赋值，改为替换 env 对象本身
        self.env = saved_env
        self.board.env = saved_env
        self.board.last_move = saved_last_move
        self.board.anim_row  = -1
        self.board.anim_col  = -1
        self.board.update()
        self.status_bar.set_result("")
        self.undo_btn.setEnabled(bool(self._history))
        self._update_winrate()

    # ── 鼠标交互 ─────────────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if self.animating or self.env.done():
            return
        if self.env.turn != self.player_color:
            return
        # 将点击坐标转换到 board 子控件的本地坐标
        local = self.board.mapFromParent(event.pos())
        col = self.board.col_at(local.x())
        if col < 0 or col not in self.env.valid_move():
            return
        row = self.board.find_drop_row(col)
        if row == -1:
            return
        # 落子前保存快照（env + last_move），用于悔棋
        self._history.append((self.env.copy(), self.board.last_move))
        self.undo_btn.setEnabled(True)

        self.board.last_move = (row, col)
        color = RED_DARK if self.player_color == 1 else YEL_DARK
        self._start_animation(row, col, color, lambda: self._after_move(col))


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    app = QApplication([])
    app.setFont(QFont("Segoe UI", 11))
    gui = Connect4GUI()
    gui.show()
    app.exec_()
