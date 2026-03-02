"""AlphaZero Othello — Frosted Glass GUI"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

from src.player import Human, AlphaZeroPlayer
from src.environments import load
import torch
import time
import math
import threading
import numpy as np

from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF, pyqtSignal, QThread
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QPen,
    QRadialGradient, QPainterPath, QLinearGradient, QConicalGradient)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QComboBox, QSlider,
    QPushButton, QCheckBox, QFrame, QTabWidget,
    QSizePolicy, QScrollArea, QTextEdit,
    QGraphicsBlurEffect)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

ENV_NAME = 'Othello'
MODEL_NAME = 'AZ'
PARAMS_PATH = './params/{name}_{env}_{net}_{type}.pt'
N_ACTIONS = 65      # 64 squares + 1 pass
BOARD_SIZE = 8
MAX_STEPS = 60      # max possible moves in Othello

# Action ↔ board coordinate helpers
def action_to_rc(action):
    """Convert action index (0-63) to (row, col). Action 64 = pass."""
    if action == 64:
        return None
    return action // 8, action % 8

def rc_to_action(row, col):
    return row * 8 + col

def action_label(action):
    """Human-readable label for an action: e.g. 'd3' or 'pass'."""
    if action == 64:
        return "pass"
    r, c = action_to_rc(action)
    return f"{chr(ord('a') + c)}{r + 1}"


class Def:
    network = 'CNN'
    model_type = 'current'
    n_playout = 500
    c_init = 1.4
    c_base = 1000
    fpu = 0.2
    alpha = 0.0
    noise_eps = 0.0
    symmetry = True
    cache = 10000
    mlh_slope = 0.1
    mlh_cap = 0.15
    mlh_thr = 0.


# ═══════════════════════════════════════════════════════════════════════════════
# Theme — Frosted Glass / Glassmorphism
# ═══════════════════════════════════════════════════════════════════════════════

class C:
    # Backgrounds
    BG       = "#0b0b1e"
    SURFACE  = "rgba(255, 255, 255, 10)"
    SURFACE2 = "rgba(255, 255, 255, 15)"
    BORDER   = "rgba(255, 255, 255, 25)"
    BORDER2  = "rgba(255, 255, 255, 40)"

    # Text
    TEXT     = "#e0e0f5"
    DIM      = "#9090b0"
    MUTED    = "#505070"
    GREEN_T  = "#4ade80"

    # Accents
    ACCENT   = "#a78bfa"
    ACCENT2  = "#c4b5fd"
    CYAN     = "#38bdf8"
    CYAN2    = "#0ea5e9"
    MAGENTA  = "#c084fc"
    GREEN    = "#4ade80"
    RED_HEX  = "#fb7185"
    YEL_HEX  = "#fbbf24"

    # QColor objects — accent
    ACCENT_CLR   = QColor(167, 139, 250)
    ACCENT_DIM   = QColor(167, 139, 250, 40)
    ACCENT_GLOW  = QColor(167, 139, 250, 80)
    MAGENTA_CLR  = QColor(192, 132, 252)

    # Piece colors  — Black & White for Othello
    BLACK        = QColor(30, 30, 50)
    BLACK_LT     = QColor(80, 80, 110)
    BLACK_GLOW   = QColor(100, 100, 160, 50)
    WHITE        = QColor(230, 230, 245)
    WHITE_LT     = QColor(255, 255, 255)
    WHITE_GLOW   = QColor(230, 230, 245, 50)

    # Board
    BOARD_BG     = QColor(12, 12, 35, 200)
    CELL_BG      = QColor(8, 8, 25, 180)
    GRID_CORE    = QColor(255, 255, 255, 18)
    GRID_GLOW    = QColor(167, 139, 250, 12)
    HOVER        = QColor(167, 139, 250, 25)
    WIN_GLOW     = QColor(167, 139, 250, 150)

    # Glass
    GLASS_FILL   = QColor(255, 255, 255, 10)
    GLASS_BORDER = QColor(255, 255, 255, 25)
    GLASS_HL     = QColor(255, 255, 255, 15)

    # Board surface — dark green for Othello
    BOARD_GREEN  = QColor(0, 80, 50, 180)
    BOARD_GREEN_LT = QColor(10, 100, 65, 200)


STYLESHEET = """
/* ── Base ── */
QWidget {
    background: transparent;
    color: #e0e0f5;
    font-family: "Consolas", "Cascadia Code", monospace;
    font-size: 13px;
}
/* ── Tabs ── */
QTabWidget::pane {
    border: 1px solid rgba(255, 255, 255, 20);
    border-radius: 8px;
    background: rgba(255, 255, 255, 6);
    padding: 8px;
    top: -1px;
}
QTabBar::tab {
    background: transparent;
    color: #7070a0;
    padding: 7px 18px;
    border: none;
    border-bottom: 2px solid transparent;
    font-family: "Consolas", monospace;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QTabBar::tab:selected { color: #a78bfa; border-bottom-color: #a78bfa; }
QTabBar::tab:hover:!selected { color: #c4b5fd; }
/* ── Combo / Spin ── */
QComboBox, QSpinBox {
    background: rgba(255, 255, 255, 8);
    border: 1px solid rgba(255, 255, 255, 18);
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 26px;
    color: #e0e0f5;
    font-family: "Consolas", monospace;
    selection-background-color: rgba(167, 139, 250, 30);
}
QComboBox:hover, QSpinBox:hover {
    border-color: rgba(255, 255, 255, 35);
    background: rgba(255, 255, 255, 12);
}
QComboBox:focus, QSpinBox:focus { border-color: rgba(167, 139, 250, 60); }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #16163a;
    border: 1px solid rgba(255, 255, 255, 20);
    selection-background-color: rgba(167, 139, 250, 30);
    color: #e0e0f5;
    outline: none;
}
/* ── Slider ── */
QSlider { min-height: 22px; }
QSlider::groove:horizontal { height: 3px; background: rgba(255, 255, 255, 15); border-radius: 1px; }
QSlider::handle:horizontal {
    width: 12px; height: 12px; margin: -5px 0;
    background: #a78bfa; border-radius: 6px;
}
QSlider::handle:horizontal:hover { background: #c4b5fd; }
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(167, 139, 250, 80), stop:1 #a78bfa);
    border-radius: 1px;
}
/* ── Buttons ── */
QPushButton {
    background: rgba(255, 255, 255, 6);
    color: #c4b5fd;
    border: 1px solid rgba(255, 255, 255, 18);
    border-radius: 6px;
    padding: 6px 16px;
    font-family: "Consolas", monospace;
    font-weight: bold;
    font-size: 12px;
    min-height: 30px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
QPushButton:hover {
    background: rgba(167, 139, 250, 15);
    border-color: rgba(167, 139, 250, 50);
    color: #e0d4ff;
}
QPushButton:pressed { background: rgba(167, 139, 250, 25); }
QPushButton#primary {
    background: rgba(167, 139, 250, 18);
    border-color: rgba(167, 139, 250, 50);
}
QPushButton#primary:hover {
    background: rgba(167, 139, 250, 30);
    border-color: rgba(167, 139, 250, 70);
}
QPushButton#danger { border-color: rgba(251, 113, 133, 60); color: #fb7185; }
QPushButton#danger:hover { background: rgba(251, 113, 133, 12); border-color: rgba(251, 113, 133, 80); }
/* ── Checkbox ── */
QCheckBox { spacing: 6px; background: transparent; color: #9090b0; font-family: "Consolas"; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid rgba(255, 255, 255, 20); border-radius: 3px;
    background: rgba(255, 255, 255, 6);
}
QCheckBox::indicator:checked { background: #a78bfa; border-color: #a78bfa; }
QCheckBox:hover { color: #c4b5fd; }
/* ── Scrollbar ── */
QScrollBar:vertical { width: 5px; background: transparent; }
QScrollBar::handle:vertical { background: rgba(255, 255, 255, 20); border-radius: 2px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: rgba(167, 139, 250, 60); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
/* ── TextEdit ── */
QTextEdit {
    background: rgba(255, 255, 255, 4);
    border: 1px solid rgba(255, 255, 255, 12);
    border-radius: 6px;
    padding: 6px;
    color: #9090b0;
    font-family: "Consolas", monospace;
    font-size: 11px;
    selection-background-color: rgba(167, 139, 250, 30);
}
/* ── Labels ── */
QLabel { background: transparent; }
/* ── Separators ── */
QFrame#sep { background: rgba(255, 255, 255, 12); max-height: 1px; }
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sv(slider):
    return slider.value() / slider._scale


def _make_slider(layout, label, lo, hi, step, default, decimals=2, tooltip=""):
    row = QHBoxLayout()
    name = QLabel(label)
    name.setFixedWidth(76)
    name.setStyleSheet(f"color: {C.DIM}; font-size: 11px; font-family: Consolas;")
    if tooltip:
        name.setToolTip(tooltip)
    row.addWidget(name)

    scale = 10 ** decimals
    s = QSlider(Qt.Horizontal)
    s._scale = scale
    s._decimals = decimals
    s.setRange(int(lo * scale), int(hi * scale))
    s.setSingleStep(max(1, int(step * scale)))
    s.setValue(int(default * scale))
    row.addWidget(s, stretch=1)

    fmt = f"{{:.{decimals}f}}" if decimals else "{:.0f}"
    vl = QLabel(fmt.format(default))
    vl.setFixedWidth(55)
    vl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    vl.setStyleSheet(f"color: {C.GREEN_T}; font-family: Consolas; font-size: 11px;")
    s.valueChanged.connect(lambda v: vl.setText(fmt.format(v / scale)))
    row.addWidget(vl)
    layout.addLayout(row)
    return s


def _sep():
    f = QFrame()
    f.setObjectName("sep")
    f.setFrameShape(QFrame.HLine)
    f.setFixedHeight(1)
    return f


def _draw_glass(qp, rect, radius=10, fill_alpha=10, border_alpha=25):
    r = QRectF(rect)
    path = QPainterPath()
    path.addRoundedRect(r, radius, radius)
    qp.fillPath(path, QColor(255, 255, 255, fill_alpha))
    qp.save()
    qp.setClipPath(path)
    hl_h = min(50, r.height() * 0.3)
    highlight = QLinearGradient(r.x(), r.y(), r.x(), r.y() + hl_h)
    highlight.setColorAt(0, QColor(255, 255, 255, 18))
    highlight.setColorAt(1, QColor(255, 255, 255, 0))
    qp.fillRect(QRectF(r.x(), r.y(), r.width(), hl_h), highlight)
    qp.restore()
    qp.setPen(QPen(QColor(255, 255, 255, border_alpha), 1))
    qp.setBrush(Qt.NoBrush)
    qp.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)


def _draw_soft_glow(qp, x1, y1, x2, y2, color, core_w=1):
    for w, alpha in [(core_w + 4, 10), (core_w + 2, 30), (core_w, 100)]:
        c = QColor(color)
        c.setAlpha(alpha)
        qp.setPen(QPen(c, w))
        qp.drawLine(int(x1), int(y1), int(x2), int(y2))


# ═══════════════════════════════════════════════════════════════════════════════
# Board Widget — Othello 8×8
# ═══════════════════════════════════════════════════════════════════════════════

class BoardWidget(QWidget):
    CELL = 56
    MARGIN = 24

    def __init__(self, env, parent=None):
        super().__init__(parent)
        self.env = env
        w = self.CELL * 8 + self.MARGIN * 2
        h = self.CELL * 8 + self.MARGIN + 8
        self.setFixedSize(w, h)
        self.setMouseTracking(True)

        self.last_move = None          # (row, col) or 'pass'
        self.hover_cell = None         # (row, col)
        self.interactive = True
        self.ghost_color = None        # QColor for player piece preview

        # Scan-line animation
        self.scan_y = -1
        self.scanning = False

        # MCTS overlay
        self.overlay_data = None       # dict with arrays indexed by action
        self.overlay_best = -1

        # Valid moves cache for ghost/hover
        self._valid_set = set()

    def _board(self):
        """Return 8×8 int board: 1 = black, -1 = white, 0 = empty."""
        state = self.env.current_state()
        turn = 1 if state[0, 2, 0, 0] >= 0 else -1
        return ((state[0, 0] - state[0, 1]) * turn).astype(int)

    def update_valid(self):
        self._valid_set = set(self.env.valid_move())

    # ── Paint ───────────────────────────────────────────────────────────────
    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        self._draw_bg(qp)
        self._draw_hover_cell(qp)
        self._draw_grid(qp)
        self._draw_pieces(qp)
        self._draw_last_move(qp)
        self._draw_overlay(qp)
        self._draw_ghost(qp)
        self._draw_coords(qp)
        self._draw_scan_line(qp)

    def _draw_bg(self, qp):
        _draw_glass(qp, self.rect(), radius=12, fill_alpha=8, border_alpha=20)
        # Dark green board surface
        m = self.MARGIN
        board_rect = QRectF(m, m, self.CELL * 8, self.CELL * 8)
        qp.fillRect(board_rect, C.BOARD_GREEN)
        # Subtle inner glow
        bw, bh = self.CELL * 8, self.CELL * 8
        cx, cy = m + bw / 2, m + bh / 2
        inner = QRadialGradient(cx, cy, max(bw, bh) * 0.6)
        inner.setColorAt(0, QColor(167, 139, 250, 6))
        inner.setColorAt(1, QColor(167, 139, 250, 0))
        qp.fillRect(board_rect, inner)

    def _draw_hover_cell(self, qp):
        if self.hover_cell is None or not self.interactive:
            return
        r, c = self.hover_cell
        action = rc_to_action(r, c)
        if action not in self._valid_set:
            return
        x = self.MARGIN + c * self.CELL
        y = self.MARGIN + r * self.CELL
        qp.fillRect(int(x), int(y), self.CELL, self.CELL,
                     QColor(167, 139, 250, 25))

    def _draw_grid(self, qp):
        m, c = self.MARGIN, self.CELL
        # Glow pass
        qp.setPen(QPen(C.GRID_GLOW, 2))
        for col in range(1, 8):
            x = m + col * c
            qp.drawLine(x, m, x, m + 8 * c)
        for row in range(1, 8):
            y = m + row * c
            qp.drawLine(m, y, m + 8 * c, y)
        # Core pass
        qp.setPen(QPen(C.GRID_CORE, 1))
        for col in range(1, 8):
            x = m + col * c
            qp.drawLine(x, m, x, m + 8 * c)
        for row in range(1, 8):
            y = m + row * c
            qp.drawLine(m, y, m + 8 * c, y)
        # Outer border
        border_rect = QRectF(m - 1, m - 1, 8 * c + 2, 8 * c + 2)
        for w, alpha in [(3, 6), (1, 25)]:
            qp.setPen(QPen(QColor(255, 255, 255, alpha), w))
            qp.setBrush(Qt.NoBrush)
            qp.drawRoundedRect(border_rect, 3, 3)
        # Star points (Othello standard: 4 dots)
        qp.setBrush(QColor(255, 255, 255, 40))
        qp.setPen(Qt.NoPen)
        for sr, sc in [(2, 2), (2, 6), (6, 2), (6, 6)]:
            sx = m + sc * c
            sy = m + sr * c
            qp.drawEllipse(QPointF(sx, sy), 3, 3)

    def _draw_pieces(self, qp):
        board = self._board()
        for r in range(8):
            for cc in range(8):
                v = board[r][cc]
                if v == 0:
                    continue
                cx = self.MARGIN + cc * self.CELL + self.CELL // 2
                cy = self.MARGIN + r * self.CELL + self.CELL // 2
                self._draw_piece(qp, cx, cy, v)

    def _draw_piece(self, qp, cx, cy, value, alpha=255):
        rad = self.CELL // 2 - 6
        if value == 1:
            # Black piece
            dk = QColor(C.BLACK); lt = QColor(C.BLACK_LT); glow_c = QColor(C.BLACK_GLOW)
        else:
            # White piece
            dk = QColor(C.WHITE); lt = QColor(C.WHITE_LT); glow_c = QColor(C.WHITE_GLOW)
        dk.setAlpha(alpha); lt.setAlpha(alpha)

        # Outer glow halo
        for i in range(3):
            gc = QColor(glow_c)
            gc.setAlpha(max(0, glow_c.alpha() - i * 15))
            qp.setBrush(Qt.NoBrush)
            qp.setPen(QPen(gc, 2))
            qp.drawEllipse(QPointF(cx, cy), rad + 3 + i * 3, rad + 3 + i * 3)

        # Core piece with glass-like gradient
        grad = QRadialGradient(cx - rad * 0.25, cy - rad * 0.25, rad * 1.1)
        grad.setColorAt(0, lt)
        grad.setColorAt(0.7, dk)
        grad.setColorAt(1, QColor(dk.red() // 2, dk.green() // 2, dk.blue() // 2, alpha))
        qp.setBrush(grad)
        qp.setPen(Qt.NoPen)
        qp.drawEllipse(QPointF(cx, cy), rad, rad)

        # Glass specular highlight
        spec = QRadialGradient(cx - rad * 0.2, cy - rad * 0.35, rad * 0.45)
        spec.setColorAt(0, QColor(255, 255, 255, 90 if value == -1 else 40))
        spec.setColorAt(0.5, QColor(255, 255, 255, 30 if value == -1 else 15))
        spec.setColorAt(1, QColor(255, 255, 255, 0))
        qp.setBrush(spec)
        qp.drawEllipse(QPointF(cx - rad * 0.2, cy - rad * 0.35), rad * 0.4, rad * 0.3)

    def _draw_last_move(self, qp):
        if self.last_move is None or self.last_move == 'pass':
            return
        r, cc = self.last_move
        cx = self.MARGIN + cc * self.CELL + self.CELL // 2
        cy = self.MARGIN + r * self.CELL + self.CELL // 2
        rad = self.CELL // 2 - 3
        for w, a in [(5, 15), (3, 40), (2, 100)]:
            qp.setBrush(Qt.NoBrush)
            qp.setPen(QPen(QColor(167, 139, 250, a), w))
            qp.drawEllipse(QPointF(cx, cy), rad, rad)

    def _draw_overlay(self, qp):
        """Draw MCTS stats overlay on valid move cells."""
        if self.overlay_data is None or not self.interactive:
            return
        od = self.overlay_data
        n_arr, q_arr, w_arr = od['N'], od['Q'], od['W']
        rad = self.CELL // 2 - 6

        for action in range(64):
            if n_arr[action] <= 0:
                continue
            r, c = action // 8, action % 8
            cx = self.MARGIN + c * self.CELL + self.CELL // 2
            cy = self.MARGIN + r * self.CELL + self.CELL // 2
            is_best = (action == self.overlay_best)

            # Semi-transparent glass circle
            fill_a = 35 if is_best else 15
            border_a = 60 if is_best else 25
            qp.setBrush(QColor(167, 139, 250, fill_a))
            qp.setPen(QPen(QColor(167, 139, 250, border_a), 1))
            qp.drawEllipse(QPointF(cx, cy), rad, rad)

            main_clr = QColor(C.ACCENT) if is_best else QColor(C.DIM)
            sub_clr = QColor(167, 139, 250, 160) if is_best else QColor(C.MUTED)

            # N%
            qp.setPen(main_clr)
            qp.setFont(QFont("Consolas", 8, QFont.Bold))
            n_text = f"{n_arr[action]:.0f}%" if n_arr[action] >= 10 else f"{n_arr[action]:.1f}%"
            qp.drawText(QRectF(cx - rad, cy - rad, rad * 2, rad * 0.9),
                         Qt.AlignCenter | Qt.AlignBottom, n_text)

            # Q value
            qp.setPen(sub_clr)
            qp.setFont(QFont("Consolas", 7))
            q_val = -q_arr[action]
            q_text = f"{q_val:+.2f}" if abs(q_val) < 10 else f"{q_val:+.0f}"
            qp.drawText(QRectF(cx - rad, cy - rad * 0.15, rad * 2, rad * 0.7),
                         Qt.AlignCenter, q_text)

            # W%
            qp.setPen(sub_clr)
            qp.setFont(QFont("Consolas", 7))
            w_text = f"W:{w_arr[action]:.0f}"
            qp.drawText(QRectF(cx - rad, cy + rad * 0.15, rad * 2, rad * 0.85),
                         Qt.AlignCenter | Qt.AlignTop, w_text)

    def _draw_ghost(self, qp):
        if self.hover_cell is None or not self.interactive or self.ghost_color is None:
            return
        r, c = self.hover_cell
        action = rc_to_action(r, c)
        if action not in self._valid_set:
            return
        cx = self.MARGIN + c * self.CELL + self.CELL // 2
        cy = self.MARGIN + r * self.CELL + self.CELL // 2
        rad = self.CELL // 2 - 6
        gc = QColor(self.ghost_color)
        gc.setAlpha(50)
        qp.setBrush(gc)
        qp.setPen(QPen(QColor(self.ghost_color.red(), self.ghost_color.green(),
                               self.ghost_color.blue(), 80), 1))
        qp.drawEllipse(QPointF(cx, cy), rad, rad)

    def _draw_coords(self, qp):
        """Draw row/column coordinate labels around the board."""
        m, c = self.MARGIN, self.CELL
        qp.setFont(QFont("Consolas", 9))
        qp.setPen(QColor(C.MUTED))
        for i in range(8):
            # Column labels (a-h) at top
            x = m + i * c + c // 2
            qp.drawText(QRectF(x - 10, 4, 20, m - 4), Qt.AlignCenter,
                         chr(ord('a') + i))
            # Row labels (1-8) at left
            y = m + i * c + c // 2
            qp.drawText(QRectF(2, y - 8, m - 4, 16), Qt.AlignCenter,
                         str(i + 1))

    def _draw_scan_line(self, qp):
        if not self.scanning or self.scan_y < 0:
            return
        m = self.MARGIN
        bw = self.CELL * 8
        y = m + self.scan_y
        grad = QLinearGradient(m, y, m + bw, y)
        grad.setColorAt(0, QColor(167, 139, 250, 0))
        grad.setColorAt(0.2, QColor(167, 139, 250, 40))
        grad.setColorAt(0.5, QColor(167, 139, 250, 70))
        grad.setColorAt(0.8, QColor(167, 139, 250, 40))
        grad.setColorAt(1, QColor(167, 139, 250, 0))
        qp.setPen(QPen(grad, 2))
        qp.drawLine(m, int(y), m + bw, int(y))
        for dy, a in [(-2, 12), (-1, 25), (1, 25), (2, 12)]:
            qp.setPen(QPen(QColor(167, 139, 250, a), 1))
            qp.drawLine(m, int(y + dy), m + bw, int(y + dy))

    # ── Mouse ───────────────────────────────────────────────────────────────
    def mouseMoveEvent(self, event):
        cell = self.cell_at(event.x(), event.y())
        if cell != self.hover_cell:
            self.hover_cell = cell
            self.update()

    def leaveEvent(self, _):
        if self.hover_cell is not None:
            self.hover_cell = None
            self.update()

    # ── Utilities ───────────────────────────────────────────────────────────
    def cell_at(self, x, y):
        rx = x - self.MARGIN
        ry = y - self.MARGIN
        if rx < 0 or rx >= self.CELL * 8 or ry < 0 or ry >= self.CELL * 8:
            return None
        return ry // self.CELL, rx // self.CELL


# ═══════════════════════════════════════════════════════════════════════════════
# Display Widgets
# ═══════════════════════════════════════════════════════════════════════════════

class WinRateBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(14)
        self.w_rate = self.d_rate = self.l_rate = 0.0

    def set_rates(self, win, draw, lose):
        self.w_rate, self.d_rate, self.l_rate = win, draw, lose
        self.update()

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, w, h), 4, 4)
        qp.setClipPath(path)
        qp.fillRect(0, 0, w, h, QColor(255, 255, 255, 8))
        x = 0
        # Black win, Draw (cyan), White win
        for ratio, clr in [
            (self.w_rate, C.BLACK), (self.d_rate, QColor(56, 189, 248)),
            (self.l_rate, C.WHITE)
        ]:
            pw = int(w * ratio)
            if pw > 0:
                grad = QLinearGradient(x, 0, x, h)
                grad.setColorAt(0, QColor(clr.red(), clr.green(), clr.blue(), 200))
                grad.setColorAt(1, QColor(clr.red() // 2, clr.green() // 2, clr.blue() // 2, 200))
                qp.fillRect(int(x), 0, pw, h, grad)
            x += pw
        qp.setClipping(False)
        qp.setPen(QPen(QColor(255, 255, 255, 25), 1))
        qp.drawLine(0, 0, w, 0)


class StepsBar(QWidget):
    MAX_STEPS = MAX_STEPS

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(14)
        self.steps = 0

    def set_steps(self, steps):
        self.steps = max(0, min(int(round(steps)), self.MAX_STEPS))
        self.update()

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        text = str(self.steps)
        fm = qp.fontMetrics()
        tw = fm.horizontalAdvance(text) + 8
        bw = self.width() - tw
        h = self.height()
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, bw, h), 4, 4)
        qp.setClipPath(path)
        qp.fillRect(0, 0, bw, h, QColor(255, 255, 255, 8))
        ratio = self.steps / self.MAX_STEPS if self.MAX_STEPS else 0
        fill_w = int(bw * ratio)
        if fill_w > 0:
            grad = QLinearGradient(0, 0, fill_w, 0)
            grad.setColorAt(0, QColor(80, 60, 140))
            grad.setColorAt(1, QColor(167, 139, 250))
            qp.fillRect(0, 0, fill_w, h, grad)
        qp.setClipping(False)
        qp.setPen(QColor(C.ACCENT))
        qp.setFont(QFont("Consolas", 10))
        qp.drawText(bw + 4, 0, tw, h, Qt.AlignVCenter | Qt.AlignLeft, text)


class RootStatsWidget(QWidget):
    """MCTS root node statistics — visit distribution, Q values, WDL.
    For Othello: shows top-N moves as horizontal bars instead of 7-column chart."""
    TOP_N = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(125)
        self.dimmed = False
        self.visits = None
        self.q_values = None
        self.prior = None
        self.child_m = None
        self.child_d = None
        self.child_p1w = None
        self.child_p2w = None
        self.root_n = 0
        self.root_q = 0.0
        self.root_m = 0.0
        self.wdl = None
        self.chosen = -1
        self.ai_turn = 1

    def set_data(self, stats, chosen=-1, ai_turn=1):
        self.visits = stats['N'].copy()
        self.q_values = stats['Q'].copy()
        self.prior = stats['prior'].copy()
        self.child_m = stats['M'].copy()
        self.child_d = stats['D'].copy()
        self.child_p1w = stats['P1W'].copy()
        self.child_p2w = stats['P2W'].copy()
        self.root_n = float(stats['root_N'])
        self.root_q = float(stats['root_Q'])
        self.root_m = float(stats['root_M'])
        self.wdl = np.array([float(stats['root_D']),
                             float(stats['root_P1W']),
                             float(stats['root_P2W'])])
        self.chosen = chosen
        self.ai_turn = ai_turn
        self.update()

    def clear_data(self):
        self.visits = None
        self.q_values = None
        self.prior = None
        self.child_m = None
        self.child_d = None
        self.child_p1w = None
        self.child_p2w = None
        self.root_n = 0
        self.root_q = 0.0
        self.root_m = 0.0
        self.wdl = None
        self.chosen = -1
        self.update()

    def snapshot(self):
        if self.visits is None:
            return None
        return dict(visits=self.visits.copy(), q_values=self.q_values.copy(),
                    prior=self.prior.copy(),
                    child_m=self.child_m.copy(), child_d=self.child_d.copy(),
                    child_p1w=self.child_p1w.copy(), child_p2w=self.child_p2w.copy(),
                    root_n=self.root_n,
                    root_q=self.root_q, root_m=self.root_m,
                    wdl=self.wdl.copy(), chosen=self.chosen,
                    ai_turn=self.ai_turn)

    def restore(self, snap):
        if snap is None:
            self.clear_data()
            return
        self.visits = snap['visits']
        self.q_values = snap['q_values']
        self.prior = snap['prior']
        self.child_m = snap['child_m']
        self.child_d = snap['child_d']
        self.child_p1w = snap['child_p1w']
        self.child_p2w = snap['child_p2w']
        self.root_n = snap['root_n']
        self.root_q = snap['root_q']
        self.root_m = snap['root_m']
        self.wdl = snap['wdl']
        self.chosen = snap['chosen']
        self.ai_turn = snap['ai_turn']
        self.update()

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        if self.visits is None or self.root_n <= 0:
            qp.setPen(QColor(C.MUTED))
            qp.setFont(QFont("Consolas", 9))
            qp.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "-- awaiting search --")
            return

        summary_h = 16
        self._draw_summary(qp, w, summary_h)

        # Show top-N moves as horizontal bars
        order = np.argsort(-self.visits)
        top = [i for i in order[:self.TOP_N] if self.visits[i] > 0]
        if not top:
            return

        total_v = self.visits.sum()
        max_v = self.visits[top[0]]
        bar_h = 11
        y_start = summary_h + 4
        bar_w_max = w - 80  # leave space for label + percentage

        for rank, idx in enumerate(top):
            y = y_start + rank * (bar_h + 2)
            v = self.visits[idx]
            is_ch = (idx == self.chosen)
            pct = v / total_v if total_v > 0 else 0
            bw = max(2, int(bar_w_max * v / max_v))

            # Action label
            qp.setPen(QColor(C.ACCENT) if is_ch else QColor(C.MUTED))
            qp.setFont(QFont("Consolas", 8, QFont.Bold if is_ch else QFont.Normal))
            lbl = action_label(idx)
            qp.drawText(QRectF(4, y, 36, bar_h), Qt.AlignVCenter | Qt.AlignRight, lbl)

            # Bar
            bx = 44
            ct = QColor(167, 139, 250) if is_ch else QColor(140, 100, 200)
            cb = QColor(70, 50, 120) if is_ch else QColor(50, 30, 70)
            grad = QLinearGradient(bx, y, bx + bw, y)
            grad.setColorAt(0, cb)
            grad.setColorAt(1, ct)
            path = QPainterPath()
            path.addRoundedRect(QRectF(bx, y, bw, bar_h), 3, 3)
            qp.fillPath(path, grad)

            if is_ch:
                for gw, ga in [(bw, 10), (bw, 16)]:
                    gp = QPainterPath()
                    gp.addRoundedRect(QRectF(bx, y - 1, gw, bar_h + 2), 4, 4)
                    qp.fillPath(gp, QColor(167, 139, 250, ga))

            # Percentage
            qp.setPen(QColor(C.GREEN_T) if is_ch else QColor(C.DIM))
            qp.setFont(QFont("Consolas", 7))
            pct_text = f"{pct:.0%}" if pct >= 0.1 else f"{pct:.1%}"
            qp.drawText(QRectF(bx + bw + 4, y, 40, bar_h),
                         Qt.AlignVCenter | Qt.AlignLeft, pct_text)

        if self.dimmed:
            qp.fillRect(QRectF(0, 0, w, h), QColor(11, 11, 30, 190))

    def _draw_summary(self, qp, w, sh):
        d_pct = self.wdl[0] * 100
        if self.ai_turn == 1:
            w_pct, l_pct = self.wdl[1] * 100, self.wdl[2] * 100
        else:
            w_pct, l_pct = self.wdl[2] * 100, self.wdl[1] * 100

        parts = [
            (f"N:{int(self.root_n)}", C.ACCENT),
            (f"Q:{self.root_q:+.2f}", C.GREEN_T),
            (f"M:{self.root_m:.1f}", C.YEL_HEX),
            (f"W:{w_pct:.0f}", C.GREEN),
            (f"D:{d_pct:.0f}", C.DIM),
            (f"L:{l_pct:.0f}", C.RED_HEX),
        ]
        qp.setFont(QFont("Consolas", 8))
        fm = qp.fontMetrics()
        x = 2
        for txt, color in parts:
            qp.setPen(QColor(color))
            qp.drawText(int(x), int(sh - 3), txt)
            x += fm.horizontalAdvance(txt) + 6


class ChildStatsTable(QWidget):
    """Glass table showing per-action child node statistics."""
    ROW_H = 16
    HDR_H = 18
    MAX_ROWS = 8

    COLS = ['Pos', 'N', 'N%', 'Q', 'W%', 'D%', 'L%', 'M', 'P', 'N/P']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dimmed = False
        self._stats = None
        self._update_height()

    def _update_height(self):
        self.setFixedHeight(self.HDR_H + self.ROW_H * self.MAX_ROWS + 4)

    def set_source(self, root_stats_widget):
        self._stats = root_stats_widget

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        s = self._stats

        _draw_glass(qp, self.rect(), radius=8, fill_alpha=8, border_alpha=18)

        if s is None or s.visits is None or s.root_n <= 0:
            qp.setPen(QColor(C.MUTED))
            qp.setFont(QFont("Consolas", 9))
            qp.drawText(QRectF(0, 0, w, h), Qt.AlignCenter,
                         "-- awaiting search --")
            return

        total_v = s.visits.sum()

        ratios = [0.07, 0.12, 0.08, 0.10, 0.10, 0.09, 0.10, 0.10, 0.09, 0.10]
        pad = 6
        usable = w - pad * 2
        col_x = [pad]
        col_w = []
        for r in ratios:
            cw = int(usable * r)
            col_w.append(cw)
            col_x.append(col_x[-1] + cw)
        col_w[-1] = usable - sum(col_w[:-1])

        # Header
        qp.setFont(QFont("Consolas", 8))
        qp.setPen(QColor(C.DIM))
        for ci, name in enumerate(self.COLS):
            rect = QRectF(col_x[ci], 2, col_w[ci], self.HDR_H)
            qp.drawText(rect, Qt.AlignCenter, name)
        y_line = self.HDR_H
        qp.setPen(QPen(QColor(255, 255, 255, 15), 1))
        qp.drawLine(pad, int(y_line), int(w - pad), int(y_line))

        # Sort by visits, show top MAX_ROWS
        order = np.argsort(-s.visits)
        shown = [i for i in order if s.visits[i] > 0][:self.MAX_ROWS]

        ai = s.ai_turn
        for rank, idx in enumerate(shown):
            y = self.HDR_H + rank * self.ROW_H
            n = int(s.visits[idx])
            is_chosen = (idx == s.chosen)

            if is_chosen:
                qp.fillRect(QRectF(pad, y, usable, self.ROW_H),
                             QColor(167, 139, 250, 15))

            qp.setFont(QFont("Consolas", 8))

            n_pct = (n / total_v * 100) if total_v > 0 else 0.0
            q = -s.q_values[idx]
            d_pct = s.child_d[idx] * 100
            if ai == 1:
                w_pct = s.child_p1w[idx] * 100
                l_pct = s.child_p2w[idx] * 100
            else:
                w_pct = s.child_p2w[idx] * 100
                l_pct = s.child_p1w[idx] * 100
            m_val = s.child_m[idx]
            prior = s.prior[idx] * 100

            cells = [
                (action_label(idx),  C.ACCENT if is_chosen else C.MUTED),
                (str(n) if n > 0 else '-',
                    C.TEXT if n > 0 else C.MUTED),
                (f"{n_pct:.1f}" if n > 0 else '-',
                    C.GREEN_T if is_chosen and n > 0 else (C.DIM if n > 0 else C.MUTED)),
                (f"{q:+.3f}" if n > 0 else '-',
                    C.GREEN if (n > 0 and q > 0.05) else
                    (C.RED_HEX if (n > 0 and q < -0.05) else C.DIM)),
                (f"{w_pct:.1f}" if n > 0 else '-',
                    C.GREEN if n > 0 else C.MUTED),
                (f"{d_pct:.1f}" if n > 0 else '-',
                    C.DIM),
                (f"{l_pct:.1f}" if n > 0 else '-',
                    C.RED_HEX if n > 0 else C.MUTED),
                (f"{m_val:.1f}" if n > 0 else '-',
                    C.YEL_HEX if n > 0 else C.MUTED),
                (f"{prior:.1f}" if prior > 0.05 else '<.1',
                    C.MAGENTA if prior > 5 else C.DIM),
                (f"{n / (s.prior[idx] * 100):.0f}" if (n > 0 and s.prior[idx] > 1e-6) else '-',
                    C.TEXT if n > 0 else C.MUTED),
            ]

            for ci, (txt, color) in enumerate(cells):
                qp.setPen(QColor(color))
                rect = QRectF(col_x[ci], y, col_w[ci], self.ROW_H)
                qp.drawText(rect, Qt.AlignCenter, txt)

        if self.dimmed:
            qp.fillRect(QRectF(0, 0, w, h), QColor(11, 11, 30, 190))


# ═══════════════════════════════════════════════════════════════════════════════
# Status Panel — Glass Card
# ═══════════════════════════════════════════════════════════════════════════════

class StatusPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(16)

        # ── Left: turn + disc count + time + result ──
        left = QVBoxLayout()
        left.setSpacing(2)
        self.turn_lbl = QLabel("")
        self.turn_lbl.setAlignment(Qt.AlignCenter)
        self.turn_lbl.setTextFormat(Qt.RichText)
        f = QFont("Consolas", 12)
        f.setBold(True)
        self.turn_lbl.setFont(f)
        left.addWidget(self.turn_lbl)

        self.disc_lbl = QLabel("")
        self.disc_lbl.setAlignment(Qt.AlignCenter)
        self.disc_lbl.setTextFormat(Qt.RichText)
        left.addWidget(self.disc_lbl)

        self.time_lbl = QLabel("--")
        self.time_lbl.setAlignment(Qt.AlignCenter)
        self.time_lbl.setStyleSheet(f"color: {C.DIM}; font-size: 10px; font-family: Consolas;")
        left.addWidget(self.time_lbl)

        self.result_lbl = QLabel("")
        self.result_lbl.setAlignment(Qt.AlignCenter)
        self.result_lbl.setTextFormat(Qt.RichText)
        f2 = QFont("Consolas", 11)
        f2.setBold(True)
        self.result_lbl.setFont(f2)
        left.addWidget(self.result_lbl)
        root.addLayout(left)

        # ── Middle: WDL ──
        mid = QVBoxLayout()
        mid.setSpacing(2)
        rate_row = QHBoxLayout()
        self.win_lbl = self._rate_label("WIN", C.GREEN)
        self.draw_lbl = self._rate_label("DRAW", C.CYAN)
        self.lose_lbl = self._rate_label("LOSE", C.RED_HEX)
        rate_row.addWidget(self.win_lbl)
        rate_row.addWidget(self.draw_lbl)
        rate_row.addWidget(self.lose_lbl)
        mid.addLayout(rate_row)
        self.wdl_bar = WinRateBar()
        mid.addWidget(self.wdl_bar)
        self.nn_wdl_lbl = QLabel("")
        self.nn_wdl_lbl.setAlignment(Qt.AlignCenter)
        self.nn_wdl_lbl.setTextFormat(Qt.RichText)
        mid.addWidget(self.nn_wdl_lbl)
        root.addLayout(mid, stretch=1)

        # ── Right: steps ──
        right = QVBoxLayout()
        right.setSpacing(2)
        steps_title = QLabel(f"<font color='{C.DIM}' style='font-family:Consolas;font-size:10px;'>REMAINING</font>")
        steps_title.setAlignment(Qt.AlignCenter)
        steps_title.setTextFormat(Qt.RichText)
        right.addWidget(steps_title)
        self.steps_bar = StepsBar()
        self.steps_bar.setMinimumWidth(100)
        right.addWidget(self.steps_bar)
        self.nn_steps_lbl = QLabel("")
        self.nn_steps_lbl.setAlignment(Qt.AlignCenter)
        self.nn_steps_lbl.setTextFormat(Qt.RichText)
        right.addWidget(self.nn_steps_lbl)
        root.addLayout(right)

    def paintEvent(self, event):
        super().paintEvent(event)
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        _draw_glass(qp, self.rect(), radius=10, fill_alpha=10, border_alpha=22)

    @staticmethod
    def _rate_label(prefix, color):
        lbl = QLabel(f"<font color='{color}' style='font-family:Consolas;font-size:10px;'>"
                     f"{prefix}</font><br><font color='{color}'>--%</font>")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setTextFormat(Qt.RichText)
        return lbl

    def set_mcts_rates(self, win, draw, lose):
        self.win_lbl.setText(
            f"<font color='{C.GREEN}' style='font-family:Consolas;font-size:10px;'>WIN</font>"
            f"<br><font color='{C.GREEN}'>{win:.1f}%</font>")
        self.draw_lbl.setText(
            f"<font color='{C.CYAN}' style='font-family:Consolas;font-size:10px;'>DRAW</font>"
            f"<br><font color='{C.CYAN}'>{draw:.1f}%</font>")
        self.lose_lbl.setText(
            f"<font color='{C.RED_HEX}' style='font-family:Consolas;font-size:10px;'>LOSE</font>"
            f"<br><font color='{C.RED_HEX}'>{lose:.1f}%</font>")
        self.wdl_bar.set_rates(win / 100, draw / 100, lose / 100)

    def set_disc_count(self, black, white):
        self.disc_lbl.setText(
            f"<font color='{C.TEXT}' style='font-family:Consolas;font-size:11px;'>"
            f"\u25cf {black}</font>"
            f"<font color='{C.DIM}' style='font-family:Consolas;font-size:11px;'> - </font>"
            f"<font color='{C.DIM}' style='font-family:Consolas;font-size:11px;'>"
            f"\u25cb {white}</font>")

    def set_nn_rates(self, win, draw, lose):
        self.nn_wdl_lbl.setText(
            f"<font color='{C.MUTED}' style='font-family:Consolas;font-size:9px;'>"
            f"nn {win:.1f} / {draw:.1f} / {lose:.1f}</font>")

    def set_mcts_steps(self, m):
        self.steps_bar.set_steps(m)

    def set_nn_steps(self, s):
        self.nn_steps_lbl.setText(
            f"<font color='{C.MUTED}' style='font-family:Consolas;font-size:9px;'>"
            f"nn {s:.0f}</font>")

    def clear_mcts(self):
        for lbl, prefix, color in [(self.win_lbl, 'WIN', C.GREEN),
                                    (self.draw_lbl, 'DRAW', C.CYAN),
                                    (self.lose_lbl, 'LOSE', C.RED_HEX)]:
            lbl.setText(f"<font color='{color}' style='font-family:Consolas;"
                        f"font-size:10px;'>{prefix}</font><br>"
                        f"<font color='{color}'>--%</font>")
        self.wdl_bar.set_rates(0, 0, 0)
        self.steps_bar.set_steps(0)
        self.nn_wdl_lbl.setText("")
        self.nn_steps_lbl.setText("")

    def set_thinking(self, sec):
        self.time_lbl.setText(f"{sec:.2f}s" if sec >= 0 else "--")

    def set_turn(self, text, color):
        self.turn_lbl.setText(f"<font color='{color}'>{text}</font>")

    def set_result(self, text, color=None):
        if text:
            self.result_lbl.setText(f"<font color='{color or C.TEXT}'>{text}</font>")
        else:
            self.result_lbl.setText("")


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter Console — Glass Tabs
# ═══════════════════════════════════════════════════════════════════════════════

class ParameterConsole(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        title = QLabel(f"<font color='{C.ACCENT}' style='font-family:Consolas;"
                       f"font-size:11px;letter-spacing:3px;'>PARAMETERS</font>")
        title.setTextFormat(Qt.RichText)
        root.addWidget(title)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs, stretch=1)

        self._build_game_tab()
        self._build_mcts_tab()
        self._build_mlh_tab()

    def _build_game_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        for label_text, attr_name, items in [
            ("NETWORK", "network_cb", ["CNN"]),
            ("WEIGHTS", "model_type_cb", ["current", "best"]),
            ("PLAYER", "player_cb", ["Human Black (first)", "Human White (second)"]),
        ]:
            lay.addWidget(QLabel(
                f"<font color='{C.DIM}' style='font-family:Consolas;font-size:10px;"
                f"letter-spacing:1px;'>{label_text}</font>"))
            cb = QComboBox()
            cb.addItems(items)
            setattr(self, attr_name, cb)
            lay.addWidget(cb)

        lay.addStretch()
        self.tabs.addTab(w, "GAME")

    def _build_mcts_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 6)
        lay.setSpacing(10)

        row = QHBoxLayout()
        lbl = QLabel("sims")
        lbl.setFixedWidth(76)
        lbl.setStyleSheet(f"color: {C.DIM}; font-size: 11px; font-family: Consolas;")
        lbl.setToolTip("MCTS simulations per move")
        row.addWidget(lbl)
        self.n_playout_spin = QSpinBox()
        self.n_playout_spin.setRange(1, 10000)
        self.n_playout_spin.setValue(Def.n_playout)
        row.addWidget(self.n_playout_spin)
        lay.addLayout(row)

        self.c_init_sl = _make_slider(lay, "c_init", 0, 10, 0.1, Def.c_init,
                                      tooltip="PUCT exploration constant")
        self.c_base_sl = _make_slider(lay, "c_base", 1, 100000, 100, Def.c_base, decimals=0,
                                      tooltip="PUCT log base")
        self.fpu_sl = _make_slider(lay, "fpu", 0, 2, 0.1, Def.fpu,
                                   tooltip="First Play Urgency reduction")
        self.alpha_sl = _make_slider(lay, "alpha", 0, 5, 0.01, Def.alpha,
                                     tooltip="Dirichlet noise alpha (0=disabled)")
        self.eps_sl = _make_slider(lay, "epsilon", 0, 1, 0.05, Def.noise_eps,
                                   tooltip="Noise mix weight")
        self.cache_sl = _make_slider(lay, "cache", 0, 1000000, 1000, Def.cache, decimals=0,
                                     tooltip="LRU transposition table size (0=disabled)")

        self.sym_check = QCheckBox("SYMMETRY AUG")
        self.sym_check.setChecked(Def.symmetry)
        self.sym_check.setToolTip("Random symmetry transform on leaf nodes")
        lay.addWidget(self.sym_check)

        lay.addStretch()
        self.tabs.addTab(w, "MCTS")

    def _build_mlh_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        info = QLabel(f"<font color='{C.DIM}' style='font-family:Consolas;font-size:10px;'>"
                      f"Moves-Left Head (LC0)</font>")
        info.setTextFormat(Qt.RichText)
        lay.addWidget(info)

        self.mlh_slope_sl = _make_slider(lay, "slope", 0, 1, 0.01, Def.mlh_slope,
                                         tooltip="MLH strength (0=disabled)")
        self.mlh_cap_sl = _make_slider(lay, "cap", 0, 1, 0.05, Def.mlh_cap,
                                       tooltip="MLH max effect")
        self.mlh_thr_sl = _make_slider(lay, "threshold", 0, 1, 0.05, Def.mlh_thr,
                                       tooltip="MLH Q threshold")
        lay.addStretch()
        self.tabs.addTab(w, "MLH")

    def reset_defaults(self):
        self.network_cb.setCurrentText(Def.network)
        self.model_type_cb.setCurrentText(Def.model_type)
        self.player_cb.setCurrentIndex(0)
        self.n_playout_spin.setValue(Def.n_playout)
        self.c_init_sl.setValue(int(Def.c_init * self.c_init_sl._scale))
        self.c_base_sl.setValue(int(Def.c_base * self.c_base_sl._scale))
        self.fpu_sl.setValue(int(Def.fpu * self.fpu_sl._scale))
        self.alpha_sl.setValue(int(Def.alpha * self.alpha_sl._scale))
        self.eps_sl.setValue(int(Def.noise_eps * self.eps_sl._scale))
        self.cache_sl.setValue(int(Def.cache * self.cache_sl._scale))
        self.sym_check.setChecked(Def.symmetry)
        self.mlh_slope_sl.setValue(int(Def.mlh_slope * self.mlh_slope_sl._scale))
        self.mlh_cap_sl.setValue(int(Def.mlh_cap * self.mlh_cap_sl._scale))
        self.mlh_thr_sl.setValue(int(Def.mlh_thr * self.mlh_thr_sl._scale))


# ═══════════════════════════════════════════════════════════════════════════════
# Move Log
# ═══════════════════════════════════════════════════════════════════════════════

class MoveLog(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self._lines = []

    def add_move(self, num, player, action):
        color = C.TEXT if player == 1 else C.DIM
        symbol = "\u25cf" if player == 1 else "\u25cb"  # ● / ○
        self._lines.append(
            f'<span style="color:{C.MUTED}">#{num:2d}</span> '
            f'<span style="color:{color}"><b>{symbol}</b></span> '
            f'<span style="color:{C.DIM}">&rarr; {action_label(action)}</span>')
        self.setHtml('<br>'.join(self._lines))
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_log(self):
        self._lines.clear()
        self.clear()

    def snapshot(self):
        return list(self._lines)

    def restore(self, lines):
        self._lines = list(lines)
        self.setHtml('<br>'.join(self._lines))
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


# ═══════════════════════════════════════════════════════════════════════════════
# MCTS Search Worker (background thread)
# ═══════════════════════════════════════════════════════════════════════════════

class ContinuousSearchWorker(QThread):
    CHUNK = 50

    progress = pyqtSignal(dict, object)
    ai_ready = pyqtSignal(dict, object, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bmcts = None
        self._pv_fn = None
        self._board = None
        self._turns = None
        self._is_ai_turn = False
        self._threshold = 500
        self._t0 = 0.0
        self._ai_acted = False
        self._paused = True
        self._stop_flag = False
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()

    def set_position(self, bmcts, pv_fn, board, turns, is_ai_turn, threshold):
        self._bmcts = bmcts
        self._pv_fn = pv_fn
        self._board = np.ascontiguousarray(board, dtype=np.int8)
        self._turns = np.ascontiguousarray(turns, dtype=np.int32)
        self._is_ai_turn = is_ai_turn
        self._threshold = threshold
        self._t0 = time.time()
        self._ai_acted = False

    def resume(self):
        self._paused = False
        self._idle.clear()
        self._wake.set()

    def pause_and_wait(self):
        self._paused = True
        self._wake.set()
        self._idle.wait()

    def stop(self):
        self._stop_flag = True
        self._paused = True
        self._wake.set()

    def run(self):
        while not self._stop_flag:
            if self._paused:
                self._idle.set()
                self._wake.wait()
                self._wake.clear()
                continue
            if self._stop_flag:
                break

            bm = self._bmcts
            pv = self._pv_fn
            if bm is None or pv is None:
                self._paused = True
                continue

            bm.batch_playout(pv, self._board, self._turns,
                             n_playout=self.CHUNK)

            if self._paused or self._stop_flag:
                continue

            raw = bm.get_root_stats()
            stats_0 = {}
            for k, v in raw.items():
                val = v[0]
                stats_0[k] = val.copy() if hasattr(val, 'copy') else float(val)
            visits = bm.get_visits_count()[0].copy()

            self.progress.emit(stats_0, visits)

            if self._is_ai_turn and not self._ai_acted:
                if stats_0['root_N'] >= self._threshold:
                    self._ai_acted = True
                    self._paused = True
                    elapsed = time.time() - self._t0
                    self.ai_ready.emit(stats_0, visits, elapsed)

        self._idle.set()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════════════════════

class OthelloGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaZero Othello — Glass")
        self.setStyleSheet(STYLESHEET)
        self.setWindowFlags(Qt.Window)

        # ── Core objects ────────────────────────────────────────────────────
        self.env_module = load(ENV_NAME)
        self.env = self.env_module.Env()
        self.human = Human()
        self.az_player = AlphaZeroPlayer(
            None, c_init=None, c_base=Def.c_base, n_playout=None,
            alpha=Def.alpha, noise_epsilon=Def.noise_eps,
            is_selfplay=0, cache_size=Def.cache,
            fpu_reduction=Def.fpu, use_symmetry=Def.symmetry,
            game_name='Othello',
            mlh_slope=Def.mlh_slope, mlh_cap=Def.mlh_cap,
            mlh_threshold=Def.mlh_thr)
        self.player_color = 1    # 1 = Black (first player)
        self.move_count = 0

        # ── Widgets ─────────────────────────────────────────────────────────
        self.board = BoardWidget(self.env)
        self.status = StatusPanel()
        self.console = ParameterConsole()

        self.ai_root_stats = RootStatsWidget()
        self.ai_child_table = ChildStatsTable()
        self.ai_child_table.set_source(self.ai_root_stats)
        self.hint_root_stats = RootStatsWidget()
        self.hint_child_table = ChildStatsTable()
        self.hint_child_table.set_source(self.hint_root_stats)

        self.move_log = MoveLog()

        # ── Buttons ─────────────────────────────────────────────────────────
        self.undo_btn = QPushButton("UNDO")
        self.undo_btn.setEnabled(False)
        self.pass_btn = QPushButton("PASS")
        self.pass_btn.setToolTip("Pass your turn (only when no legal moves)")
        self.restart_btn = QPushButton("NEW GAME")
        self.restart_btn.setObjectName("primary")
        self.reset_btn = QPushButton("DEFAULTS")
        self.reset_btn.setToolTip("Reset all parameters to defaults")
        self.log_btn = QPushButton("LOG")
        self.log_btn.setToolTip("Toggle move log")
        self.pause_btn = QPushButton("PAUSE")
        self.pause_btn.setToolTip("Pause / resume background MCTS search")
        self.pause_btn.setCheckable(True)
        self._search_paused = False
        self.hint_btn = QPushButton("HINT")
        self.hint_btn.setToolTip("Toggle YOUR ANALYSIS visibility")
        self.hint_btn.setCheckable(True)
        self.hint_btn.setChecked(True)
        self._hint_visible = True

        # ── Layout ──────────────────────────────────────────────────────────
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(8)

        # Top row: board + console+buttons (right side height = board height)
        top_row = QHBoxLayout()
        top_row.setSpacing(16)
        top_row.addWidget(self.board)

        right_container = QWidget()
        right_container.setFixedHeight(self.board.height())
        right_col = QVBoxLayout(right_container)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(4)
        right_col.addWidget(self.console, stretch=1)
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(6)
        btn_row1.addWidget(self.undo_btn)
        btn_row1.addWidget(self.pass_btn)
        btn_row1.addWidget(self.restart_btn)
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(6)
        btn_row2.addWidget(self.pause_btn)
        btn_row2.addWidget(self.hint_btn)
        btn_row2.addWidget(self.log_btn)
        btn_row3 = QHBoxLayout()
        btn_row3.setSpacing(6)
        btn_row3.addWidget(self.reset_btn)
        right_col.addLayout(btn_row1)
        right_col.addLayout(btn_row2)
        right_col.addLayout(btn_row3)
        top_row.addWidget(right_container)
        main_layout.addLayout(top_row)

        # Status (full width)
        main_layout.addWidget(self.status)

        # Bottom row: AI panel | Hint panel
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)

        # AI last move panel
        ai_panel = QVBoxLayout()
        ai_panel.setSpacing(3)
        ai_title = QLabel(
            f"<font color='{C.ACCENT}' style='font-family:Consolas;"
            f"font-size:11px;letter-spacing:2px;'>AI LAST MOVE</font>")
        ai_title.setTextFormat(Qt.RichText)
        ai_panel.addWidget(ai_title)
        ai_panel.addWidget(self.ai_root_stats)
        ai_panel.addWidget(self.ai_child_table)
        bottom_row.addLayout(ai_panel)

        # Vertical separator
        vsep = QFrame()
        vsep.setFrameShape(QFrame.VLine)
        vsep.setStyleSheet("color: rgba(255, 255, 255, 15);")
        vsep.setFixedWidth(1)
        bottom_row.addWidget(vsep)

        # Hint panel
        self.hint_container = QWidget()
        hint_panel = QVBoxLayout(self.hint_container)
        hint_panel.setContentsMargins(0, 0, 0, 0)
        hint_panel.setSpacing(3)
        hint_title = QLabel(
            f"<font color='{C.GREEN_T}' style='font-family:Consolas;"
            f"font-size:11px;letter-spacing:2px;'>YOUR ANALYSIS</font>")
        hint_title.setTextFormat(Qt.RichText)
        hint_panel.addWidget(hint_title)
        hint_panel.addWidget(self.hint_root_stats)
        hint_panel.addWidget(self.hint_child_table)
        self.hint_blur = QGraphicsBlurEffect()
        self.hint_blur.setBlurRadius(0)
        self.hint_blur.setEnabled(False)
        self.hint_container.setGraphicsEffect(self.hint_blur)
        bottom_row.addWidget(self.hint_container)

        main_layout.addLayout(bottom_row)

        # Window size — calculate from content
        total_w = self.board.width() + 320 + 48
        # top_margin(12) + board(480) + spacing(8) + status(~80) + spacing(8)
        # + bottom_panels(~300) + bottom_margin(12) ≈ 900
        total_h = (12 + self.board.height() + 8 + 85 + 8
                   + 20 + 125 + 150 + 12)
        self.setFixedSize(total_w, total_h)

        # MoveLog overlay drawer
        self._build_log_drawer()

        # ── Timers ──────────────────────────────────────────────────────────
        self.reload_timer = QTimer()
        self.reload_timer.setSingleShot(True)
        self.reload_timer.timeout.connect(self._reload_and_restart)

        self.settings_timer = QTimer()
        self.settings_timer.setSingleShot(True)
        self.settings_timer.timeout.connect(self._reload_and_restart)

        self.param_timer = QTimer()
        self.param_timer.setSingleShot(True)
        self.param_timer.timeout.connect(self._update_search_params)

        self.scan_timer = QTimer()
        self.scan_timer.timeout.connect(self._step_scan)

        self.worker = ContinuousSearchWorker()
        self.worker.progress.connect(self._on_progress)
        self.worker.ai_ready.connect(self._on_ai_ready)
        self.worker.start()

        self._history = []

        self._reload_model()

        # ── Connect signals ─────────────────────────────────────────────────
        _model_delayed = lambda _=None: self.settings_timer.start(400)
        self.console.network_cb.currentIndexChanged.connect(_model_delayed)
        self.console.model_type_cb.currentIndexChanged.connect(_model_delayed)

        _param_delayed = lambda _=None: self.param_timer.start(150)
        self.console.n_playout_spin.valueChanged.connect(self._on_sims_changed)
        self.console.c_init_sl.valueChanged.connect(_param_delayed)
        self.console.c_base_sl.valueChanged.connect(_param_delayed)
        self.console.fpu_sl.valueChanged.connect(_param_delayed)
        self.console.alpha_sl.valueChanged.connect(_param_delayed)
        self.console.eps_sl.valueChanged.connect(_param_delayed)
        self.console.cache_sl.valueChanged.connect(_param_delayed)
        self.console.sym_check.stateChanged.connect(_param_delayed)
        self.console.mlh_slope_sl.valueChanged.connect(_param_delayed)
        self.console.mlh_cap_sl.valueChanged.connect(_param_delayed)
        self.console.mlh_thr_sl.valueChanged.connect(_param_delayed)

        self.console.player_cb.currentIndexChanged.connect(
            lambda _: self.reload_timer.start(100))
        self.restart_btn.clicked.connect(self._reload_and_restart)
        self.undo_btn.clicked.connect(self._undo)
        self.pass_btn.clicked.connect(self._human_pass)
        self.reset_btn.clicked.connect(self.console.reset_defaults)
        self.pause_btn.clicked.connect(self._toggle_search_pause)
        self.hint_btn.clicked.connect(self._toggle_hint)
        self.log_btn.clicked.connect(self._toggle_log)

        self._start_game()

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        super().closeEvent(event)

    # ── Window background — gradient + decorative orbs ───────────────────────
    def paintEvent(self, event):
        qp = QPainter(self)
        w, h = self.width(), self.height()

        bg = QLinearGradient(0, 0, w, h)
        bg.setColorAt(0.0, QColor(11, 11, 30))
        bg.setColorAt(0.4, QColor(18, 14, 45))
        bg.setColorAt(0.7, QColor(15, 12, 38))
        bg.setColorAt(1.0, QColor(11, 11, 30))
        qp.fillRect(self.rect(), bg)

        # Decorative orbs
        orb1 = QRadialGradient(w * 0.15, h * 0.2, w * 0.35)
        orb1.setColorAt(0, QColor(124, 91, 245, 28))
        orb1.setColorAt(0.5, QColor(124, 91, 245, 10))
        orb1.setColorAt(1, QColor(124, 91, 245, 0))
        qp.fillRect(self.rect(), orb1)

        orb2 = QRadialGradient(w * 0.85, h * 0.75, w * 0.3)
        orb2.setColorAt(0, QColor(56, 189, 248, 22))
        orb2.setColorAt(0.5, QColor(56, 189, 248, 8))
        orb2.setColorAt(1, QColor(56, 189, 248, 0))
        qp.fillRect(self.rect(), orb2)

        # Green orb (for Othello feel)
        orb3 = QRadialGradient(w * 0.5, h * 0.12, w * 0.25)
        orb3.setColorAt(0, QColor(0, 180, 100, 18))
        orb3.setColorAt(0.5, QColor(0, 180, 100, 6))
        orb3.setColorAt(1, QColor(0, 180, 100, 0))
        qp.fillRect(self.rect(), orb3)

        # Subtle dot pattern
        qp.setPen(Qt.NoPen)
        qp.setBrush(QColor(255, 255, 255, 3))
        for x in range(0, w, 50):
            for y in range(0, h, 50):
                qp.drawEllipse(QPointF(x, y), 0.5, 0.5)

    # ═══════════════════════════════════════════════════════════════════════
    # MoveLog Drawer (overlay)
    # ═══════════════════════════════════════════════════════════════════════

    def _build_log_drawer(self):
        d = QFrame(self)
        d.setStyleSheet(
            "QFrame { background: rgba(16, 14, 40, 230); "
            "border-left: 1px solid rgba(255, 255, 255, 15); }")
        d.setFixedWidth(260)
        lay = QVBoxLayout(d)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        hdr = QHBoxLayout()
        title = QLabel(
            f"<font color='{C.ACCENT}' style='font-family:Consolas;"
            f"font-size:11px;letter-spacing:2px;'>MOVE LOG</font>")
        title.setTextFormat(Qt.RichText)
        hdr.addWidget(title)
        hdr.addStretch()
        close_btn = QPushButton("\u2715")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet(
            f"QPushButton {{ border:none; color:{C.DIM}; font-size:14px; }}"
            f"QPushButton:hover {{ color:{C.ACCENT}; }}")
        close_btn.clicked.connect(self._toggle_log)
        hdr.addWidget(close_btn)
        lay.addLayout(hdr)
        lay.addWidget(_sep())
        lay.addWidget(self.move_log, stretch=1)

        d.hide()
        self.log_drawer = d

    def _toggle_log(self):
        self.log_drawer.setVisible(not self.log_drawer.isVisible())
        if self.log_drawer.isVisible():
            self.log_drawer.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'log_drawer'):
            self.log_drawer.setGeometry(
                self.width() - 260, 0, 260, self.height())

    # ═══════════════════════════════════════════════════════════════════════
    # Model Management
    # ═══════════════════════════════════════════════════════════════════════

    def _reload_model(self):
        network = self.console.network_cb.currentText()
        model_type = self.console.model_type_cb.currentText()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = getattr(self.env_module, network)(lr=0, device=device)
        net.eval()
        self.net = net
        path = PARAMS_PATH.format(name=MODEL_NAME, env=ENV_NAME,
                                  net=network, type=model_type)
        try:
            self.net.load(path)
        except Exception:
            self.status.set_result("MODEL ERROR", C.RED_HEX)

        p = self.az_player
        p._c_base = int(_sv(self.console.c_base_sl))
        p._fpu_reduction = _sv(self.console.fpu_sl)
        p._noise_eps = _sv(self.console.eps_sl)
        p.noise_eps_init = _sv(self.console.eps_sl)
        p._use_symmetry = self.console.sym_check.isChecked()
        p._cache_size = int(_sv(self.console.cache_sl))
        p._mlh_slope = _sv(self.console.mlh_slope_sl)
        p._mlh_cap = _sv(self.console.mlh_cap_sl)
        p._mlh_threshold = _sv(self.console.mlh_thr_sl)

        p.reload(self.net,
                 c_puct=_sv(self.console.c_init_sl),
                 n_playout=self.console.n_playout_spin.value(),
                 alpha=_sv(self.console.alpha_sl),
                 is_self_play=0)
        p.eval()
        self.player_color = 1 if self.console.player_cb.currentIndex() == 0 else -1

    def _reload_and_restart(self):
        self.worker.pause_and_wait()
        self._reload_model()
        self._start_game()

    def _update_search_params(self):
        self.worker.pause_and_wait()
        m = self.az_player.mcts
        m.set_c_init(_sv(self.console.c_init_sl))
        m.set_c_base(_sv(self.console.c_base_sl))
        m.set_alpha(_sv(self.console.alpha_sl))
        m.set_fpu_reduction(_sv(self.console.fpu_sl))
        m.set_noise_epsilon(_sv(self.console.eps_sl))
        m.set_use_symmetry(self.console.sym_check.isChecked())
        m.set_mlh_params(_sv(self.console.mlh_slope_sl),
                         _sv(self.console.mlh_cap_sl),
                         _sv(self.console.mlh_thr_sl))
        new_cache = int(_sv(self.console.cache_sl))
        old_cache = getattr(m, 'cache_size', 0) or 0
        if new_cache != old_cache:
            from src.Cache import LRUCache
            m.cache = LRUCache(new_cache) if new_cache > 0 else None
            m.cache_size = new_cache
        if not self._search_paused:
            self.worker.resume()

    # ═══════════════════════════════════════════════════════════════════════
    # Game Flow
    # ═══════════════════════════════════════════════════════════════════════

    def _start_game(self):
        self.worker.pause_and_wait()
        self._stop_scan()
        self.env.reset()
        self.az_player.mcts.reset_env(0)
        self.board.last_move = None
        self.board.interactive = True
        self.board.ghost_color = QColor(C.BLACK) if self.player_color == 1 else QColor(C.WHITE)
        self.board.scanning = False
        self.board.scan_y = -1
        self.board.overlay_data = None
        self.board.update_valid()
        self.board.update()

        self.status.set_result("")
        self.status.set_thinking(-1)
        self.status.clear_mcts()
        self.ai_root_stats.clear_data()
        self.ai_child_table.update()
        self.hint_root_stats.clear_data()
        self.hint_child_table.update()
        self.move_count = 0
        self.move_log.clear_log()
        self._history.clear()
        self.undo_btn.setEnabled(False)
        self._update_pass_btn()
        self._update_analysis()
        self._update_turn_label()

        self._search_paused = False
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("PAUSE")

        is_ai_first = (self.env.turn != self.player_color)
        if is_ai_first:
            self._start_scan()
        self._resume_search(is_ai_turn=is_ai_first)

    def _update_turn_label(self):
        if self.env.done():
            return
        if self.env.turn == self.player_color:
            color = C.TEXT if self.player_color == 1 else C.DIM
            self.status.set_turn("YOUR TURN", color)
            self.board.interactive = True
        else:
            self.status.set_turn("AI ACTIVE", C.ACCENT)
            self.board.interactive = False

    def _update_pass_btn(self):
        """Enable PASS only when it's human's turn and pass is the only valid move."""
        if self.env.done():
            self.pass_btn.setEnabled(False)
            return
        if self.env.turn != self.player_color:
            self.pass_btn.setEnabled(False)
            return
        valid = self.env.valid_move()
        # Pass (action 64) is valid only when no board moves exist
        self.pass_btn.setEnabled(64 in valid and len(valid) == 1)

    def _update_analysis(self):
        # Disc count
        board = self.board._board()
        self.status.set_disc_count(int(np.sum(board == 1)), int(np.sum(board == -1)))

        with torch.no_grad():
            state = self.env.current_state()
            t = torch.from_numpy(state).float().to(self.net.device).unsqueeze(0)
            if t.dim() == 5:
                t = t.squeeze(1)
            _, vl, sl = self.net(t)

            vp = vl[0].exp().cpu().tolist()
            draw_pct = vp[0] * 100
            if self.player_color == self.env.turn:
                win_pct, lose_pct = vp[1] * 100, vp[2] * 100
            else:
                win_pct, lose_pct = vp[2] * 100, vp[1] * 100
            self.status.set_nn_rates(win_pct, draw_pct, lose_pct)

            sp = sl[0].exp().cpu()
            expected = (sp * torch.arange(len(sp), dtype=torch.float32)).sum().item()
            self.status.set_nn_steps(expected)

    def _update_status_mcts(self, stats_0):
        d = float(stats_0['root_D'])
        p1w = float(stats_0['root_P1W'])
        p2w = float(stats_0['root_P2W'])
        m = float(stats_0['root_M'])
        if self.player_color == 1:
            win_pct, lose_pct = p1w * 100, p2w * 100
        else:
            win_pct, lose_pct = p2w * 100, p1w * 100
        self.status.set_mcts_rates(win_pct, d * 100, lose_pct)
        self.status.set_mcts_steps(m)

    # ── Scan-line animation ─────────────────────────────────────────────────
    def _start_scan(self):
        self.board.scanning = True
        self.board.scan_y = 0
        self.scan_timer.start(30)

    def _stop_scan(self):
        self.scan_timer.stop()
        self.board.scanning = False
        self.board.scan_y = -1
        self.board.update()

    def _step_scan(self):
        max_y = self.board.CELL * 8
        self.board.scan_y += 4
        if self.board.scan_y > max_y:
            self.board.scan_y = 0
        self.board.update()

    # ── Continuous search helpers ──────────────────────────────────────────
    def _resume_search(self, is_ai_turn):
        p = self.az_player
        board = self.env.board[np.newaxis, ...]
        turns = np.array([self.env.turn], dtype=np.int32)
        threshold = self.console.n_playout_spin.value()
        self.worker.set_position(p.mcts, p.pv_fn, board, turns,
                                 is_ai_turn, threshold)
        if not self._search_paused:
            self.worker.resume()

    def _toggle_search_pause(self):
        self._search_paused = self.pause_btn.isChecked()
        if self._search_paused:
            self.worker.pause_and_wait()
            self._stop_scan()
            self.pause_btn.setText("RESUME")
        else:
            self.pause_btn.setText("PAUSE")
            if not self.env.done():
                is_ai = (self.env.turn != self.player_color)
                if is_ai:
                    self._start_scan()
                self.worker.resume()

    def _toggle_hint(self):
        self._hint_visible = self.hint_btn.isChecked()
        self.hint_root_stats.dimmed = not self._hint_visible
        self.hint_child_table.dimmed = not self._hint_visible
        if self._hint_visible:
            self.hint_blur.setBlurRadius(0)
            self.hint_blur.setEnabled(False)
            s = self.hint_root_stats
            if s.visits is not None and s.root_n > 0:
                total = s.visits.sum()
                if total > 0:
                    n_pct = s.visits / total * 100
                    q_arr = s.q_values.copy()
                    p1w = s.child_p1w * 100
                    p2w = s.child_p2w * 100
                    w_arr = p1w if self.player_color == 1 else p2w
                    self.board.overlay_data = {'N': n_pct, 'Q': q_arr, 'W': w_arr}
                    self.board.overlay_best = int(np.argmax(s.visits))
        else:
            self.hint_blur.setBlurRadius(48)
            self.hint_blur.setEnabled(True)
            self.board.overlay_data = None
        self.board.update()
        self.hint_root_stats.update()
        self.hint_child_table.update()

    def _on_sims_changed(self, _=None):
        new_thr = self.console.n_playout_spin.value()
        self.worker.pause_and_wait()
        self.worker._threshold = new_thr
        if self.worker._is_ai_turn and not self.worker._ai_acted:
            raw = self.az_player.mcts.get_root_stats()
            root_n = float(raw['root_N'][0])
            if root_n >= new_thr:
                stats_0 = {}
                for k, v in raw.items():
                    val = v[0]
                    stats_0[k] = val.copy() if hasattr(val, 'copy') else float(val)
                visits = self.az_player.mcts.get_visits_count()[0].copy()
                elapsed = time.time() - self.worker._t0
                self._on_ai_ready(stats_0, visits, elapsed)
                return
        if not self._search_paused:
            self.worker.resume()

    # ── Continuous search callbacks ────────────────────────────────────────
    def _on_progress(self, stats_0, visits):
        self._update_status_mcts(stats_0)
        if self.env.turn != self.player_color:
            self.board.overlay_data = None
            ai_turn = self.env.turn
            self.ai_root_stats.set_data(stats_0, chosen=-1, ai_turn=ai_turn)
            self.ai_child_table.update()
            elapsed = time.time() - self.worker._t0
            self.status.set_thinking(elapsed)
        else:
            human_turn = self.env.turn
            best = int(np.argmax(visits))
            self.hint_root_stats.set_data(stats_0, chosen=best,
                                          ai_turn=human_turn)
            self.hint_child_table.update()
            if self._hint_visible:
                total = visits.sum()
                if total > 0:
                    n_pct = visits / total * 100
                    q_arr = stats_0['Q'].copy()
                    p1w = stats_0['P1W'].copy()
                    p2w = stats_0['P2W'].copy()
                    w_arr = p1w * 100 if self.player_color == 1 else p2w * 100
                    self.board.overlay_data = {'N': n_pct, 'Q': q_arr, 'W': w_arr}
                    self.board.overlay_best = best
                else:
                    self.board.overlay_data = None
        self.board.update()

    def _on_ai_ready(self, stats_0, visits, elapsed):
        self.worker.pause_and_wait()
        self.status.set_thinking(elapsed)
        self._stop_scan()

        ai_turn = self.env.turn
        action = int(np.argmax(visits))

        self.ai_root_stats.set_data(stats_0, chosen=action, ai_turn=ai_turn)
        self.ai_child_table.update()

        self.az_player.mcts.prune_roots(np.array([action], dtype=np.int32))

        # Pre-compute hint for next position
        hint_raw = self.az_player.mcts.get_root_stats()
        hint_s0 = {}
        for k, v in hint_raw.items():
            val = v[0]
            hint_s0[k] = val.copy() if hasattr(val, 'copy') else float(val)
        hint_v = self.az_player.mcts.get_visits_count()[0].copy()
        human_turn = -ai_turn
        if hint_s0['root_N'] > 0:
            best_hint = int(np.argmax(hint_v))
            self.hint_root_stats.set_data(hint_s0, chosen=best_hint,
                                          ai_turn=human_turn)
            self._update_status_mcts(hint_s0)
            if self._hint_visible:
                total = hint_v.sum()
                if total > 0:
                    n_pct = hint_v / total * 100
                    q_arr = hint_s0['Q'].copy()
                    p1w = hint_s0['P1W'].copy()
                    p2w = hint_s0['P2W'].copy()
                    w_arr = p1w * 100 if self.player_color == 1 else p2w * 100
                    self.board.overlay_data = {'N': n_pct, 'Q': q_arr, 'W': w_arr}
                    self.board.overlay_best = best_hint
        else:
            self.hint_root_stats.clear_data()
            self.board.overlay_data = None
        self.hint_child_table.update()

        # Apply AI move
        self._apply_move(action, ai_turn, is_ai=True)

    def _apply_move(self, action, current_player, is_ai):
        """Apply a move, update board, and handle game flow."""
        if action < 64:
            self.board.last_move = action_to_rc(action)
        else:
            self.board.last_move = 'pass'

        self.env.step(action)
        self.move_count += 1
        self.move_log.add_move(self.move_count, current_player, action)
        self.board.update_valid()
        self._update_analysis()
        self.board.update()

        if self.env.done():
            if is_ai:
                self.worker.pause_and_wait()
            self._show_result()
            return

        self._update_turn_label()
        self._update_pass_btn()

        if is_ai:
            # After AI move, start searching for human's turn
            env_copy_board = self.env.board[np.newaxis, ...]
            env_copy_turns = np.array([self.env.turn], dtype=np.int32)
            threshold = self.console.n_playout_spin.value()
            self.worker.set_position(self.az_player.mcts, self.az_player.pv_fn,
                                     env_copy_board, env_copy_turns,
                                     is_ai_turn=False, threshold=threshold)
            if not self._search_paused:
                self.worker.resume()
        else:
            # After human move, AI's turn
            self._start_scan()
            self._resume_search(is_ai_turn=True)

    def _show_result(self):
        winner = self.env.winPlayer()
        self.board.interactive = False
        self._update_pass_btn()
        self.board.update()

        if winner == self.player_color:
            self.status.set_turn("VICTORY", C.GREEN)
            self.status.set_result("YOU WIN", C.GREEN)
        elif winner == -self.player_color:
            self.status.set_turn("DEFEATED", C.RED_HEX)
            self.status.set_result("YOU LOSE", C.RED_HEX)
        else:
            self.status.set_turn("STALEMATE", C.YEL_HEX)
            self.status.set_result("DRAW", C.YEL_HEX)

    # ═══════════════════════════════════════════════════════════════════════
    # Human Input
    # ═══════════════════════════════════════════════════════════════════════

    def _human_pass(self):
        """Handle human pressing the PASS button."""
        if self.env.done() or self.env.turn != self.player_color:
            return
        valid = self.env.valid_move()
        if 64 not in valid:
            return

        self.worker.pause_and_wait()
        self.board.overlay_data = None
        self._save_history()
        self.az_player.mcts.prune_roots(np.array([64], dtype=np.int32))
        self._apply_move(64, self.env.turn, is_ai=False)

    def mousePressEvent(self, event):
        if self.env.done():
            return
        if self.env.turn != self.player_color:
            return
        local = self.board.mapFromParent(event.pos())
        cell = self.board.cell_at(local.x(), local.y())
        if cell is None:
            return
        r, c = cell
        action = rc_to_action(r, c)
        if action not in self.board._valid_set:
            return

        self.worker.pause_and_wait()
        self.board.overlay_data = None
        self._save_history()
        self.az_player.mcts.prune_roots(np.array([action], dtype=np.int32))
        self._apply_move(action, self.env.turn, is_ai=False)

    def _save_history(self):
        self._history.append((self.env.copy(), self.board.last_move, self.move_count,
                              self.ai_root_stats.snapshot(),
                              self.hint_root_stats.snapshot(),
                              self.move_log.snapshot()))
        self.undo_btn.setEnabled(True)

    # ═══════════════════════════════════════════════════════════════════════
    # Undo
    # ═══════════════════════════════════════════════════════════════════════

    def _undo(self):
        if not self._history:
            return
        if self.env.turn != self.player_color and not self.env.done():
            return
        self.worker.pause_and_wait()
        self._stop_scan()
        saved_env, saved_last, saved_count, saved_ai, saved_hint, saved_log = self._history.pop()
        self.env = saved_env
        self.board.env = saved_env
        self.az_player.mcts.reset_env(0)
        self.board.last_move = saved_last
        self.board.overlay_data = None
        self.board.update_valid()
        self.board.update()
        self.move_count = saved_count
        self.status.set_result("")
        self.ai_root_stats.restore(saved_ai)
        self.ai_child_table.update()
        self.hint_root_stats.restore(saved_hint)
        self.hint_child_table.update()
        self.move_log.restore(saved_log)
        self.undo_btn.setEnabled(bool(self._history))
        self._update_analysis()
        self._update_turn_label()
        self._update_pass_btn()
        if not self.env.done():
            is_ai = (self.env.turn != self.player_color)
            if is_ai:
                self._start_scan()
            self._resume_search(is_ai_turn=is_ai)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    app = QApplication([])
    app.setFont(QFont("Consolas", 11))
    gui = OthelloGUI()
    gui.show()
    app.exec_()
