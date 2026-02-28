"""AlphaZero Connect4 — Cyberpunk HUD GUI"""

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

ENV_NAME = 'Connect4'
MODEL_NAME = 'AZ'
ANIMATION_MS = 25
PARAMS_PATH = './params/{name}_{env}_{net}_{type}.pt'


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
# Theme — Cyberpunk / HUD
# ═══════════════════════════════════════════════════════════════════════════════

class C:
    # Backgrounds
    BG       = "#060a12"
    SURFACE  = "#0c1220"
    SURFACE2 = "#101828"
    BORDER   = "#0d3a4a"
    BORDER2  = "#1a5060"

    # Text
    TEXT     = "#c0e8f8"
    DIM      = "#4a8090"
    MUTED    = "#1e4050"
    GREEN_T  = "#00ff88"

    # Accents
    CYAN     = "#00e5ff"
    CYAN2    = "#00b8d4"
    MAGENTA  = "#e040fb"
    GREEN    = "#00e676"
    RED_HEX  = "#ff1744"
    YEL_HEX  = "#ffab00"

    # QColor objects
    CYAN_CLR     = QColor(0, 229, 255)
    CYAN_DIM     = QColor(0, 229, 255, 40)
    CYAN_GLOW    = QColor(0, 229, 255, 80)
    MAGENTA_CLR  = QColor(224, 64, 251)

    RED          = QColor(255, 40, 60)
    RED_LT       = QColor(255, 100, 110)
    RED_GLOW     = QColor(255, 40, 60, 60)
    YEL          = QColor(255, 180, 0)
    YEL_LT       = QColor(255, 220, 80)
    YEL_GLOW     = QColor(255, 180, 0, 60)

    BOARD_BG     = QColor(8, 14, 24)
    CELL_BG      = QColor(4, 8, 16)
    GRID_CORE    = QColor(0, 180, 212, 90)
    GRID_GLOW    = QColor(0, 229, 255, 25)
    HOVER        = QColor(0, 229, 255, 20)
    WIN_GLOW     = QColor(0, 229, 255, 150)


STYLESHEET = """
/* ── Base ── */
QWidget {
    background: #060a12;
    color: #c0e8f8;
    font-family: "Consolas", "Cascadia Code", monospace;
    font-size: 13px;
}
/* ── Tabs ── */
QTabWidget::pane {
    border: 1px solid #0d3a4a;
    border-radius: 4px;
    background: #0c1220;
    padding: 8px;
    top: -1px;
}
QTabBar::tab {
    background: transparent;
    color: #4a8090;
    padding: 7px 18px;
    border: none;
    border-bottom: 2px solid transparent;
    font-family: "Consolas", monospace;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QTabBar::tab:selected { color: #00e5ff; border-bottom-color: #00e5ff; }
QTabBar::tab:hover:!selected { color: #00b8d4; }
/* ── Combo / Spin ── */
QComboBox, QSpinBox {
    background: #0a1018;
    border: 1px solid #0d3a4a;
    border-radius: 3px;
    padding: 4px 8px;
    min-height: 26px;
    color: #00e5ff;
    font-family: "Consolas", monospace;
    selection-background-color: #0d3a4a;
}
QComboBox:hover, QSpinBox:hover { border-color: #00e5ff; }
QComboBox:focus, QSpinBox:focus { border-color: #00e5ff; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #0a1018;
    border: 1px solid #0d3a4a;
    selection-background-color: #0d3a4a;
    color: #00e5ff;
    outline: none;
}
/* ── Slider ── */
QSlider { min-height: 22px; }
QSlider::groove:horizontal { height: 3px; background: #0d3a4a; border-radius: 1px; }
QSlider::handle:horizontal {
    width: 12px; height: 12px; margin: -5px 0;
    background: #00e5ff; border-radius: 6px;
}
QSlider::handle:horizontal:hover { background: #40efff; }
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #004060, stop:1 #00e5ff);
    border-radius: 1px;
}
/* ── Buttons ── */
QPushButton {
    background: #0a1018;
    color: #00e5ff;
    border: 1px solid #0d3a4a;
    border-radius: 3px;
    padding: 6px 16px;
    font-family: "Consolas", monospace;
    font-weight: bold;
    font-size: 12px;
    min-height: 30px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
QPushButton:hover { background: #0d1828; border-color: #00e5ff; color: #40efff; }
QPushButton:pressed { background: #002030; }
QPushButton#primary { background: #003040; border-color: #00e5ff; }
QPushButton#primary:hover { background: #004060; border-color: #40efff; }
QPushButton#danger { border-color: #ff1744; color: #ff1744; }
QPushButton#danger:hover { background: #1a0010; border-color: #ff4070; }
/* ── Checkbox ── */
QCheckBox { spacing: 6px; background: transparent; color: #4a8090; font-family: "Consolas"; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #0d3a4a; border-radius: 2px;
    background: #0a1018;
}
QCheckBox::indicator:checked { background: #00e5ff; border-color: #00e5ff; }
QCheckBox:hover { color: #00e5ff; }
/* ── Scrollbar ── */
QScrollBar:vertical { width: 5px; background: transparent; }
QScrollBar::handle:vertical { background: #0d3a4a; border-radius: 2px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #00b8d4; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
/* ── TextEdit ── */
QTextEdit {
    background: #060a12;
    border: 1px solid #0d3a4a;
    border-radius: 3px;
    padding: 6px;
    color: #4a8090;
    font-family: "Consolas", monospace;
    font-size: 11px;
    selection-background-color: #0d3a4a;
}
/* ── Labels ── */
QLabel { background: transparent; }
/* ── Separators ── */
QFrame#sep { background: #0d3a4a; max-height: 1px; }
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


def _draw_glow_line(qp, x1, y1, x2, y2, color, core_w=1):
    """Draw a line with neon glow effect (3-pass)."""
    for w, alpha in [(core_w + 6, 15), (core_w + 3, 40), (core_w, 140)]:
        c = QColor(color)
        c.setAlpha(alpha)
        qp.setPen(QPen(c, w))
        qp.drawLine(int(x1), int(y1), int(x2), int(y2))


def _draw_corner_brackets(qp, rect, size=18, color=None, width=2):
    """Draw sci-fi corner bracket decorations."""
    c = color or C.CYAN_CLR
    pen = QPen(QColor(c.red(), c.green(), c.blue(), 120), width)
    qp.setPen(pen)
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    # Top-left
    qp.drawLine(int(x), int(y + size), int(x), int(y))
    qp.drawLine(int(x), int(y), int(x + size), int(y))
    # Top-right
    qp.drawLine(int(x + w - size), int(y), int(x + w), int(y))
    qp.drawLine(int(x + w), int(y), int(x + w), int(y + size))
    # Bottom-left
    qp.drawLine(int(x), int(y + h - size), int(x), int(y + h))
    qp.drawLine(int(x), int(y + h), int(x + size), int(y + h))
    # Bottom-right
    qp.drawLine(int(x + w - size), int(y + h), int(x + w), int(y + h))
    qp.drawLine(int(x + w), int(y + h - size), int(x + w), int(y + h))


# ═══════════════════════════════════════════════════════════════════════════════
# Board Widget — HUD Battle Grid
# ═══════════════════════════════════════════════════════════════════════════════

class BoardWidget(QWidget):
    CELL = 76
    MARGIN = 28

    def __init__(self, env, parent=None):
        super().__init__(parent)
        self.env = env
        w = self.CELL * 7 + self.MARGIN * 2
        h = self.CELL * 6 + self.MARGIN * 2
        self.setFixedSize(w, h)
        self.setMouseTracking(True)

        self.last_move = None
        self.win_cells = None
        self.hover_col = -1
        self.interactive = True
        self.ghost_color = None

        # Animation
        self.anim_row = -1
        self.anim_col = -1
        self.anim_color = None

        # Scan-line animation
        self.scan_y = -1
        self.scanning = False

        # MCTS overlay on empty cells
        self.overlay_data = None   # {'N': (7,), 'Q': (7,), 'W': (7,)} or None
        self.overlay_best = -1     # column with highest visits

    def _board(self):
        state = self.env.current_state()
        return (state[0, 0] - state[0, 1]).astype(int)

    # ── Paint ───────────────────────────────────────────────────────────────
    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        self._draw_bg(qp)
        self._draw_microgrid(qp)
        self._draw_hover_col(qp)
        self._draw_glow_grid(qp)
        self._draw_pieces(qp)
        self._draw_last_move(qp)
        self._draw_win_glow(qp)
        self._draw_overlay(qp)
        self._draw_ghost(qp)
        self._draw_anim(qp)
        self._draw_scan_line(qp)
        self._draw_brackets(qp)

    def _draw_bg(self, qp):
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 6, 6)
        qp.fillPath(path, C.BOARD_BG)
        # Subtle border
        qp.setPen(QPen(QColor(0, 229, 255, 30), 1))
        qp.setBrush(Qt.NoBrush)
        qp.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)

    def _draw_microgrid(self, qp):
        """Subtle dot-grid background pattern."""
        qp.setPen(Qt.NoPen)
        m = self.MARGIN
        bw, bh = self.CELL * 7, self.CELL * 6
        dot_color = QColor(0, 229, 255, 12)
        qp.setBrush(dot_color)
        for x in range(m, m + bw + 1, 19):
            for y in range(m, m + bh + 1, 19):
                qp.drawEllipse(QPointF(x, y), 0.7, 0.7)

    def _draw_hover_col(self, qp):
        if self.hover_col < 0 or not self.interactive:
            return
        x = self.MARGIN + self.hover_col * self.CELL
        m = self.MARGIN
        bh = self.CELL * 6
        # Vertical scan-beam gradient
        grad = QLinearGradient(x, m, x, m + bh)
        grad.setColorAt(0.0, QColor(0, 229, 255, 0))
        grad.setColorAt(0.3, QColor(0, 229, 255, 30))
        grad.setColorAt(0.7, QColor(0, 229, 255, 30))
        grad.setColorAt(1.0, QColor(0, 229, 255, 0))
        qp.fillRect(int(x), m, self.CELL, bh, grad)
        # Side beams
        for bx in [x, x + self.CELL]:
            _draw_glow_line(qp, bx, m + 10, bx, m + bh - 10, C.CYAN_CLR, 1)

    def _draw_glow_grid(self, qp):
        """Grid lines with multi-pass neon glow."""
        m, c = self.MARGIN, self.CELL
        # Glow pass
        glow_pen = QPen(C.GRID_GLOW, 3)
        qp.setPen(glow_pen)
        for col in range(1, 7):
            x = m + col * c
            qp.drawLine(x, m, x, m + 6 * c)
        for row in range(1, 6):
            y = m + row * c
            qp.drawLine(m, y, m + 7 * c, y)
        # Core pass
        core_pen = QPen(C.GRID_CORE, 1)
        qp.setPen(core_pen)
        for col in range(1, 7):
            x = m + col * c
            qp.drawLine(x, m, x, m + 6 * c)
        for row in range(1, 6):
            y = m + row * c
            qp.drawLine(m, y, m + 7 * c, y)
        # Outer border glow
        border_rect = QRectF(m - 1, m - 1, 7 * c + 2, 6 * c + 2)
        for w, alpha in [(5, 12), (3, 30), (1, 70)]:
            qp.setPen(QPen(QColor(0, 229, 255, alpha), w))
            qp.setBrush(Qt.NoBrush)
            qp.drawRect(border_rect)

    def _draw_pieces(self, qp):
        board = self._board()
        for r in range(6):
            for cc in range(7):
                v = board[r][cc]
                cx = self.MARGIN + cc * self.CELL + self.CELL // 2
                cy = self.MARGIN + r * self.CELL + self.CELL // 2
                self._draw_piece(qp, cx, cy, v)

    def _draw_piece(self, qp, cx, cy, value, alpha=255):
        rad = self.CELL // 2 - 8
        if value == 0:
            # Empty cell — dark pit with subtle ring
            qp.setBrush(C.CELL_BG)
            qp.setPen(QPen(QColor(0, 229, 255, 20), 1))
            qp.drawEllipse(QPointF(cx, cy), rad, rad)
            return

        dk = QColor(C.RED) if value == 1 else QColor(C.YEL)
        lt = QColor(C.RED_LT) if value == 1 else QColor(C.YEL_LT)
        glow_c = QColor(C.RED_GLOW) if value == 1 else QColor(C.YEL_GLOW)
        dk.setAlpha(alpha); lt.setAlpha(alpha)

        # Outer glow halo
        for i in range(3):
            gc = QColor(glow_c)
            gc.setAlpha(max(0, glow_c.alpha() - i * 18))
            qp.setBrush(Qt.NoBrush)
            qp.setPen(QPen(gc, 2))
            qp.drawEllipse(QPointF(cx, cy), rad + 3 + i * 3, rad + 3 + i * 3)

        # Core piece with gradient
        grad = QRadialGradient(cx - rad * 0.25, cy - rad * 0.25, rad * 1.1)
        grad.setColorAt(0, lt)
        grad.setColorAt(0.7, dk)
        grad.setColorAt(1, QColor(dk.red() // 2, dk.green() // 2, dk.blue() // 2, alpha))
        qp.setBrush(grad)
        qp.setPen(Qt.NoPen)
        qp.drawEllipse(QPointF(cx, cy), rad, rad)

        # Specular highlight
        spec = QRadialGradient(cx - rad * 0.2, cy - rad * 0.35, rad * 0.4)
        spec.setColorAt(0, QColor(255, 255, 255, 70))
        spec.setColorAt(1, QColor(255, 255, 255, 0))
        qp.setBrush(spec)
        qp.drawEllipse(QPointF(cx - rad * 0.2, cy - rad * 0.35), rad * 0.35, rad * 0.25)

    def _draw_last_move(self, qp):
        if self.last_move is None:
            return
        r, cc = self.last_move
        cx = self.MARGIN + cc * self.CELL + self.CELL // 2
        cy = self.MARGIN + r * self.CELL + self.CELL // 2
        rad = self.CELL // 2 - 4
        # Glow ring
        for w, a in [(6, 20), (3, 60), (2, 140)]:
            qp.setBrush(Qt.NoBrush)
            qp.setPen(QPen(QColor(0, 229, 255, a), w))
            qp.drawEllipse(QPointF(cx, cy), rad, rad)

    def _draw_win_glow(self, qp):
        if not self.win_cells:
            return
        for r, cc in self.win_cells:
            cx = self.MARGIN + cc * self.CELL + self.CELL // 2
            cy = self.MARGIN + r * self.CELL + self.CELL // 2
            rad = self.CELL // 2 - 2
            # Strong pulsing glow
            for w, a in [(10, 15), (6, 40), (3, 80), (2, 180)]:
                qp.setBrush(Qt.NoBrush)
                qp.setPen(QPen(QColor(0, 229, 255, a), w))
                qp.drawEllipse(QPointF(cx, cy), rad, rad)

    def _draw_overlay(self, qp):
        """Draw MCTS stats overlay on each valid column's drop position."""
        if self.overlay_data is None or not self.interactive:
            return
        od = self.overlay_data
        n_arr, q_arr, w_arr = od['N'], od['Q'], od['W']
        rad = self.CELL // 2 - 8

        for col in range(7):
            if n_arr[col] <= 0:
                continue
            row = self.find_drop_row(col)
            if row < 0:
                continue
            cx = self.MARGIN + col * self.CELL + self.CELL // 2
            cy = self.MARGIN + row * self.CELL + self.CELL // 2
            is_best = (col == self.overlay_best)

            # Semi-transparent circle
            fill_a = 40 if is_best else 18
            border_a = 80 if is_best else 35
            qp.setBrush(QColor(0, 229, 255, fill_a))
            qp.setPen(QPen(QColor(0, 229, 255, border_a), 1))
            qp.drawEllipse(QPointF(cx, cy), rad, rad)

            # Text colors
            main_clr = QColor(C.CYAN) if is_best else QColor(C.DIM)
            sub_clr = QColor(0, 229, 255, 160) if is_best else QColor(C.MUTED)

            # Line 1: N% (visit share)
            qp.setPen(main_clr)
            qp.setFont(QFont("Consolas", 9, QFont.Bold))
            n_text = f"{n_arr[col]:.0f}%" if n_arr[col] >= 10 else f"{n_arr[col]:.1f}%"
            qp.drawText(QRectF(cx - rad, cy - rad, rad * 2, rad * 0.9),
                         Qt.AlignCenter | Qt.AlignBottom, n_text)

            # Line 2: Q value
            qp.setPen(sub_clr)
            qp.setFont(QFont("Consolas", 8))
            q_val = -q_arr[col]
            q_text = f"{q_val:+.2f}" if abs(q_val) < 10 else f"{q_val:+.0f}"
            qp.drawText(QRectF(cx - rad, cy - rad * 0.15, rad * 2, rad * 0.7),
                         Qt.AlignCenter, q_text)

            # Line 3: W% (win rate)
            qp.setPen(sub_clr)
            qp.setFont(QFont("Consolas", 8))
            w_text = f"W:{w_arr[col]:.0f}"
            qp.drawText(QRectF(cx - rad, cy + rad * 0.15, rad * 2, rad * 0.85),
                         Qt.AlignCenter | Qt.AlignTop, w_text)

    def _draw_ghost(self, qp):
        if self.hover_col < 0 or not self.interactive or self.ghost_color is None:
            return
        row = self.find_drop_row(self.hover_col)
        if row < 0:
            return
        cx = self.MARGIN + self.hover_col * self.CELL + self.CELL // 2
        cy = self.MARGIN + row * self.CELL + self.CELL // 2
        rad = self.CELL // 2 - 8
        gc = QColor(self.ghost_color)
        gc.setAlpha(50)
        qp.setBrush(gc)
        qp.setPen(QPen(QColor(self.ghost_color.red(), self.ghost_color.green(),
                               self.ghost_color.blue(), 80), 1))
        qp.drawEllipse(QPointF(cx, cy), rad, rad)

    def _draw_anim(self, qp):
        if self.anim_row < 0 or self.anim_col < 0:
            return
        cx = self.MARGIN + self.anim_col * self.CELL + self.CELL // 2
        cy = self.MARGIN + self.anim_row * self.CELL + self.CELL // 2
        v = 1 if self.anim_color is C.RED else -1
        self._draw_piece(qp, cx, cy, v)

    def _draw_scan_line(self, qp):
        """Horizontal scanning line during AI thinking."""
        if not self.scanning or self.scan_y < 0:
            return
        m = self.MARGIN
        bw = self.CELL * 7
        y = m + self.scan_y
        # Glow beam
        grad = QLinearGradient(m, y, m + bw, y)
        grad.setColorAt(0, QColor(0, 229, 255, 0))
        grad.setColorAt(0.2, QColor(0, 229, 255, 60))
        grad.setColorAt(0.5, QColor(0, 229, 255, 100))
        grad.setColorAt(0.8, QColor(0, 229, 255, 60))
        grad.setColorAt(1, QColor(0, 229, 255, 0))
        qp.setPen(QPen(grad, 2))
        qp.drawLine(m, int(y), m + bw, int(y))
        # Wider soft glow
        for dy, a in [(-2, 20), (-1, 40), (1, 40), (2, 20)]:
            qp.setPen(QPen(QColor(0, 229, 255, a), 1))
            qp.drawLine(m, int(y + dy), m + bw, int(y + dy))

    def _draw_brackets(self, qp):
        """Corner bracket decorations."""
        m = self.MARGIN
        r = QRectF(m - 8, m - 8, self.CELL * 7 + 16, self.CELL * 6 + 16)
        _draw_corner_brackets(qp, r, size=14, color=C.CYAN_CLR, width=2)

    # ── Mouse ───────────────────────────────────────────────────────────────
    def mouseMoveEvent(self, event):
        col = self.col_at(event.x())
        if col != self.hover_col:
            self.hover_col = col
            self.update()

    def leaveEvent(self, _):
        if self.hover_col != -1:
            self.hover_col = -1
            self.update()

    # ── Utilities ───────────────────────────────────────────────────────────
    def col_at(self, x):
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

    def find_win_line(self):
        board = self._board()
        for r in range(6):
            for cc in range(7):
                v = board[r][cc]
                if v == 0:
                    continue
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    cells = []
                    for i in range(4):
                        nr, nc = r + dr * i, cc + dc * i
                        if 0 <= nr < 6 and 0 <= nc < 7 and board[nr][nc] == v:
                            cells.append((nr, nc))
                        else:
                            break
                    if len(cells) == 4:
                        return cells
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Display Widgets
# ═══════════════════════════════════════════════════════════════════════════════

class PolicyChart(QWidget):
    """HUD-style bar chart for column policy probabilities."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(130)
        self.probs = [0.0] * 7
        self.highlight = -1
        self.valid_mask = [True] * 7

    def set_data(self, probs, highlight=-1, valid_mask=None):
        self.probs = list(probs)
        self.highlight = highlight
        self.valid_mask = valid_mask or [True] * 7
        self.update()

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bar_zone = h - 22
        bar_w = max(12, (w - 16) // 7 - 8)
        gap = (w - bar_w * 7) / 8
        mx = max(self.probs) if max(self.probs) > 0 else 1.0

        # Background grid lines
        qp.setPen(QPen(QColor(0, 229, 255, 10), 1))
        for gy in range(0, bar_zone, 15):
            qp.drawLine(int(gap), bar_zone - gy, int(w - gap), bar_zone - gy)

        for i in range(7):
            x = gap + i * (bar_w + gap)
            p = self.probs[i]
            bh = max(2, int((bar_zone - 16) * p / mx)) if p > 0.001 else 2
            y = bar_zone - bh

            is_hl = (i == self.highlight)
            is_valid = self.valid_mask[i]

            # Bar gradient
            if not is_valid:
                clr_top = QColor(C.MUTED)
                clr_bot = QColor(C.MUTED)
            elif is_hl:
                clr_top = QColor(0, 229, 255)
                clr_bot = QColor(0, 100, 130)
            else:
                clr_top = QColor(C.DIM)
                clr_bot = QColor(C.MUTED)

            grad = QLinearGradient(x, y, x, bar_zone)
            grad.setColorAt(0, clr_top)
            grad.setColorAt(1, clr_bot)
            path = QPainterPath()
            path.addRoundedRect(QRectF(x, y, bar_w, bh), 2, 2)
            qp.fillPath(path, grad)

            # Glow on highlighted bar
            if is_hl and is_valid:
                for gw, ga in [(bar_w + 6, 15), (bar_w + 3, 25)]:
                    gc = QColor(0, 229, 255, ga)
                    gpath = QPainterPath()
                    gpath.addRoundedRect(QRectF(x - (gw - bar_w) / 2, y, gw, bh), 3, 3)
                    qp.fillPath(gpath, gc)

            # Prob label
            if p > 0.02:
                qp.setPen(QColor(C.GREEN_T) if is_hl else QColor(C.DIM))
                qp.setFont(QFont("Consolas", 8))
                text = f"{p:.0%}" if p >= 0.1 else f"{p:.1%}"
                qp.drawText(QRectF(x - 6, y - 14, bar_w + 12, 14),
                            Qt.AlignCenter, text)

            # Column label
            qp.setPen(QColor(C.CYAN) if is_hl else QColor(C.MUTED))
            qp.setFont(QFont("Consolas", 9))
            qp.drawText(QRectF(x, bar_zone + 2, bar_w, 18),
                        Qt.AlignCenter, str(i + 1))


class WinRateBar(QWidget):
    """Three-section bar with neon glow edge."""
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

        # Background
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, w, h), 3, 3)
        qp.setClipPath(path)
        qp.fillRect(0, 0, w, h, QColor(10, 16, 24))

        x = 0
        for ratio, clr in [
            (self.w_rate, C.RED), (self.d_rate, QColor(30, 60, 80)),
            (self.l_rate, C.YEL)
        ]:
            pw = int(w * ratio)
            if pw > 0:
                grad = QLinearGradient(x, 0, x, h)
                grad.setColorAt(0, QColor(clr.red(), clr.green(), clr.blue(), 200))
                grad.setColorAt(1, QColor(clr.red() // 2, clr.green() // 2, clr.blue() // 2, 200))
                qp.fillRect(int(x), 0, pw, h, grad)
            x += pw

        # Top edge glow
        qp.setClipping(False)
        qp.setPen(QPen(QColor(0, 229, 255, 40), 1))
        qp.drawLine(0, 0, w, 0)


class StepsBar(QWidget):
    """Progress bar with gradient glow fill."""
    MAX_STEPS = 42

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

        # Background
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, bw, h), 3, 3)
        qp.setClipPath(path)
        qp.fillRect(0, 0, bw, h, QColor(10, 16, 24))

        ratio = self.steps / self.MAX_STEPS if self.MAX_STEPS else 0
        fill_w = int(bw * ratio)
        if fill_w > 0:
            grad = QLinearGradient(0, 0, fill_w, 0)
            grad.setColorAt(0, QColor(0, 80, 100))
            grad.setColorAt(1, QColor(0, 229, 255))
            qp.fillRect(0, 0, fill_w, h, grad)

        # Value text
        qp.setClipping(False)
        qp.setPen(QColor(C.CYAN))
        qp.setFont(QFont("Consolas", 10))
        qp.drawText(bw + 4, 0, tw, h, Qt.AlignVCenter | Qt.AlignLeft, text)


class RootStatsWidget(QWidget):
    """MCTS root node statistics — visit distribution, Q values, WDL."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(130)
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
        """Return a lightweight copy of current display state."""
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
        """Restore from a snapshot (or clear if None)."""
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
        q_row_h = 12
        col_row_h = 14
        bar_zone = h - summary_h - q_row_h - col_row_h - 4

        bar_w = max(12, (w - 16) // 7 - 8)
        gap = (w - bar_w * 7) / 8
        total_v = self.visits.sum()
        max_v = self.visits.max() if total_v > 0 else 1

        self._draw_summary(qp, w, summary_h)

        # Background grid
        qp.setPen(QPen(QColor(224, 64, 251, 8), 1))
        for gy in range(0, int(bar_zone), 15):
            y0 = summary_h + bar_zone - gy
            qp.drawLine(int(gap), int(y0), int(w - gap), int(y0))

        for i in range(7):
            x = gap + i * (bar_w + gap)
            v = self.visits[i]
            bh = max(2, int((bar_zone - 10) * v / max_v)) if v > 0 else 2
            y = summary_h + bar_zone - bh
            is_ch = (i == self.chosen)

            if v <= 0:
                ct, cb = QColor(C.MUTED), QColor(C.MUTED)
            elif is_ch:
                ct, cb = QColor(0, 229, 255), QColor(0, 80, 110)
            else:
                ct, cb = QColor(180, 60, 220), QColor(60, 15, 80)

            grad = QLinearGradient(x, y, x, summary_h + bar_zone)
            grad.setColorAt(0, ct)
            grad.setColorAt(1, cb)
            path = QPainterPath()
            path.addRoundedRect(QRectF(x, y, bar_w, bh), 2, 2)
            qp.fillPath(path, grad)

            if is_ch and v > 0:
                for gw, ga in [(bar_w + 6, 12), (bar_w + 3, 22)]:
                    gp = QPainterPath()
                    gp.addRoundedRect(QRectF(x - (gw - bar_w) / 2, y, gw, bh), 3, 3)
                    qp.fillPath(gp, QColor(0, 229, 255, ga))

            # Visit % above bar
            if v > 0 and total_v > 0:
                pct = v / total_v
                qp.setPen(QColor(C.GREEN_T) if is_ch else QColor(C.DIM))
                qp.setFont(QFont("Consolas", 7))
                text = f"{pct:.0%}" if pct >= 0.1 else f"{pct:.1%}"
                qp.drawText(QRectF(x - 6, y - 12, bar_w + 12, 12),
                            Qt.AlignCenter, text)

            # Q value row (negated → AI/parent perspective)
            q_y = summary_h + bar_zone + 2
            if v > 0:
                q = -self.q_values[i]
                if q > 0.05:
                    qp.setPen(QColor(C.GREEN))
                elif q < -0.05:
                    qp.setPen(QColor(C.RED_HEX))
                else:
                    qp.setPen(QColor(C.DIM))
                qp.setFont(QFont("Consolas", 7))
                qp.drawText(QRectF(x - 10, q_y, bar_w + 20, q_row_h),
                            Qt.AlignCenter, f"{q:+.2f}")

            # Column label
            col_y = q_y + q_row_h
            qp.setPen(QColor(C.CYAN) if is_ch else QColor(C.MUTED))
            qp.setFont(QFont("Consolas", 9))
            qp.drawText(QRectF(x, col_y, bar_w, col_row_h),
                        Qt.AlignCenter, str(i + 1))

        if self.dimmed:
            qp.fillRect(QRectF(0, 0, w, h), QColor(6, 10, 18, 190))

    def _draw_summary(self, qp, w, sh):
        d_pct = self.wdl[0] * 100
        if self.ai_turn == 1:
            w_pct, l_pct = self.wdl[1] * 100, self.wdl[2] * 100
        else:
            w_pct, l_pct = self.wdl[2] * 100, self.wdl[1] * 100

        parts = [
            (f"N:{int(self.root_n)}", C.CYAN),
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
    """HUD table showing per-action child node statistics."""
    ROW_H = 16
    HDR_H = 18
    COLS = ['Col', 'N', 'N%', 'Q', 'W%', 'D%', 'L%', 'M', 'P']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dimmed = False
        self._stats = None   # reference to RootStatsWidget
        self._update_height()

    def _update_height(self):
        # header + 7 data rows + 2px padding
        self.setFixedHeight(self.HDR_H + self.ROW_H * 7 + 4)

    def set_source(self, root_stats_widget):
        """Link to RootStatsWidget to read data from."""
        self._stats = root_stats_widget

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        s = self._stats

        # Background frame
        qp.setPen(QPen(QColor(0, 229, 255, 30), 1))
        qp.setBrush(QColor(8, 14, 24, 200))
        qp.drawRoundedRect(QRectF(0, 0, w, h), 3, 3)

        if s is None or s.visits is None or s.root_n <= 0:
            qp.setPen(QColor(C.MUTED))
            qp.setFont(QFont("Consolas", 9))
            qp.drawText(QRectF(0, 0, w, h), Qt.AlignCenter,
                         "-- awaiting search --")
            return

        n_actions = len(s.visits)
        total_v = s.visits.sum()

        # Column widths — proportional to available width
        #  Col  N     N%    Q      W%    D%    L%    M     P
        ratios = [0.06, 0.14, 0.09, 0.11, 0.11, 0.10, 0.11, 0.11, 0.10]
        pad = 6
        usable = w - pad * 2
        col_x = [pad]
        col_w = []
        for r in ratios:
            cw = int(usable * r)
            col_w.append(cw)
            col_x.append(col_x[-1] + cw)
        # Give remainder to last column
        col_w[-1] = usable - sum(col_w[:-1])

        # ── Header ──
        qp.setFont(QFont("Consolas", 8))
        qp.setPen(QColor(C.DIM))
        for ci, name in enumerate(self.COLS):
            rect = QRectF(col_x[ci], 2, col_w[ci], self.HDR_H)
            qp.drawText(rect, Qt.AlignCenter, name)
        # Header underline
        y_line = self.HDR_H
        qp.setPen(QPen(QColor(0, 229, 255, 30), 1))
        qp.drawLine(pad, int(y_line), int(w - pad), int(y_line))

        # ── Sort rows by visits descending ──
        order = np.argsort(-s.visits)

        # Determine AI-perspective WDL per action
        ai = s.ai_turn
        for rank, idx in enumerate(order):
            y = self.HDR_H + rank * self.ROW_H
            n = int(s.visits[idx])
            is_chosen = (idx == s.chosen)

            # Row background highlight for chosen action
            if is_chosen:
                qp.fillRect(QRectF(pad, y, usable, self.ROW_H),
                             QColor(0, 229, 255, 18))

            qp.setFont(QFont("Consolas", 8))

            # Prepare cell values (Q negated → AI/parent perspective)
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
                (str(idx + 1),      C.CYAN if is_chosen else C.MUTED),
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
            ]

            for ci, (txt, color) in enumerate(cells):
                qp.setPen(QColor(color))
                rect = QRectF(col_x[ci], y, col_w[ci], self.ROW_H)
                qp.drawText(rect, Qt.AlignCenter, txt)

        if self.dimmed:
            qp.fillRect(QRectF(0, 0, w, h), QColor(6, 10, 18, 190))


# ═══════════════════════════════════════════════════════════════════════════════
# Status Panel — HUD Readout
# ═══════════════════════════════════════════════════════════════════════════════

class StatusPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(16)

        # ── Left: turn + time + result ──
        left = QVBoxLayout()
        left.setSpacing(2)
        self.turn_lbl = QLabel("")
        self.turn_lbl.setAlignment(Qt.AlignCenter)
        self.turn_lbl.setTextFormat(Qt.RichText)
        f = QFont("Consolas", 12)
        f.setBold(True)
        self.turn_lbl.setFont(f)
        left.addWidget(self.turn_lbl)

        self.time_lbl = QLabel("▸ --")
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
        self.win_lbl = self._rate_label("WIN", C.RED_HEX)
        self.draw_lbl = self._rate_label("DRAW", C.DIM)
        self.lose_lbl = self._rate_label("LOSE", C.YEL_HEX)
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
        steps_title = QLabel(f"<font color='{C.DIM}' style='font-family:Consolas;font-size:10px;'>▸ REMAINING</font>")
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
        """Draw HUD frame around the panel."""
        super().paintEvent(event)
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        # Border
        qp.setPen(QPen(QColor(0, 229, 255, 40), 1))
        qp.setBrush(QColor(12, 18, 32, 200))
        qp.drawRoundedRect(r, 4, 4)
        # Corner brackets
        _draw_corner_brackets(qp, r, size=10, color=C.CYAN_CLR, width=1)

    @staticmethod
    def _rate_label(prefix, color):
        lbl = QLabel(f"<font color='{color}' style='font-family:Consolas;font-size:10px;'>"
                     f"{prefix}</font><br><font color='{color}'>--%</font>")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setTextFormat(Qt.RichText)
        return lbl

    def set_mcts_rates(self, win, draw, lose):
        """Primary WDL display from MCTS search statistics."""
        self.win_lbl.setText(
            f"<font color='{C.RED_HEX}' style='font-family:Consolas;font-size:10px;'>WIN</font>"
            f"<br><font color='{C.RED_HEX}'>{win:.1f}%</font>")
        self.draw_lbl.setText(
            f"<font color='{C.DIM}' style='font-family:Consolas;font-size:10px;'>DRAW</font>"
            f"<br><font color='{C.DIM}'>{draw:.1f}%</font>")
        self.lose_lbl.setText(
            f"<font color='{C.YEL_HEX}' style='font-family:Consolas;font-size:10px;'>LOSE</font>"
            f"<br><font color='{C.YEL_HEX}'>{lose:.1f}%</font>")
        self.wdl_bar.set_rates(win / 100, draw / 100, lose / 100)

    def set_nn_rates(self, win, draw, lose):
        """Secondary NN WDL shown as dim subtitle below the bar."""
        self.nn_wdl_lbl.setText(
            f"<font color='{C.MUTED}' style='font-family:Consolas;font-size:9px;'>"
            f"nn {win:.1f} / {draw:.1f} / {lose:.1f}</font>")

    def set_mcts_steps(self, m):
        """Primary remaining-steps display from MCTS M value."""
        self.steps_bar.set_steps(m)

    def set_nn_steps(self, s):
        """Secondary NN steps shown as dim subtitle."""
        self.nn_steps_lbl.setText(
            f"<font color='{C.MUTED}' style='font-family:Consolas;font-size:9px;'>"
            f"nn {s:.0f}</font>")

    def clear_mcts(self):
        """Reset MCTS displays to initial state."""
        for lbl, prefix, color in [(self.win_lbl, 'WIN', C.RED_HEX),
                                    (self.draw_lbl, 'DRAW', C.DIM),
                                    (self.lose_lbl, 'LOSE', C.YEL_HEX)]:
            lbl.setText(f"<font color='{color}' style='font-family:Consolas;"
                        f"font-size:10px;'>{prefix}</font><br>"
                        f"<font color='{color}'>--%</font>")
        self.wdl_bar.set_rates(0, 0, 0)
        self.steps_bar.set_steps(0)
        self.nn_wdl_lbl.setText("")
        self.nn_steps_lbl.setText("")

    def set_thinking(self, sec):
        self.time_lbl.setText(f"▸ {sec:.2f}s" if sec >= 0 else "▸ --")

    def set_turn(self, text, color):
        self.turn_lbl.setText(f"<font color='{color}'>{text}</font>")

    def set_result(self, text, color=None):
        if text:
            self.result_lbl.setText(f"<font color='{color or C.TEXT}'>{text}</font>")
        else:
            self.result_lbl.setText("")


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter Console — Terminal Style
# ═══════════════════════════════════════════════════════════════════════════════

class ParameterConsole(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Title with HUD prefix
        title = QLabel(f"<font color='{C.CYAN}' style='font-family:Consolas;"
                       f"font-size:11px;letter-spacing:3px;'>▸ PARAMETER CONSOLE</font>")
        title.setTextFormat(Qt.RichText)
        root.addWidget(title)

        # Tabs
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
            ("NETWORK", "network_cb", ["CNN", "ViT"]),
            ("WEIGHTS", "model_type_cb", ["current", "best"]),
            ("PLAYER", "player_cb", ["Human First (X)", "AI First (X)"]),
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

    def add_move(self, num, player, col):
        color = C.RED_HEX if player == 1 else C.YEL_HEX
        symbol = "X" if player == 1 else "O"
        self._lines.append(
            f'<span style="color:{C.MUTED}">#{num:2d}</span> '
            f'<span style="color:{color}"><b>{symbol}</b></span> '
            f'<span style="color:{C.DIM}">&rarr; Col {col + 1}</span>')
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
    """Runs MCTS playouts continuously in the background.

    Signals
    -------
    progress(dict, object)
        Emitted every CHUNK playouts with (stats_0, visits).
    ai_ready(dict, object, float)
        Emitted when root_N >= threshold during AI turn.
    """
    CHUNK = 50

    progress = pyqtSignal(dict, object)          # (stats_0, visits)
    ai_ready = pyqtSignal(dict, object, float)   # (stats_0, visits, elapsed)

    def __init__(self, parent=None):
        super().__init__(parent)
        # search state (written only while paused)
        self._bmcts = None
        self._pv_fn = None
        self._board = None
        self._turns = None
        self._is_ai_turn = False
        self._threshold = 500
        self._t0 = 0.0
        self._ai_acted = False
        # synchronisation
        self._paused = True
        self._stop_flag = False
        self._wake = threading.Event()      # wakes run-loop
        self._idle = threading.Event()      # signalled when worker is idle
        self._idle.set()

    # ── called from main thread ───────────────────────────────────────────
    def set_position(self, bmcts, pv_fn, board, turns, is_ai_turn, threshold):
        """Set new search position.  **Must** be called while worker is paused."""
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
        """Ask worker to pause and block until it is actually idle."""
        self._paused = True
        self._wake.set()        # in case it's sleeping
        self._idle.wait()       # block until idle

    def stop(self):
        self._stop_flag = True
        self._paused = True
        self._wake.set()

    # ── background thread ─────────────────────────────────────────────────
    def run(self):
        while not self._stop_flag:
            # ── sleep until resumed ──
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

            # ── one chunk of playouts ──
            bm.batch_playout(pv, self._board, self._turns,
                             n_playout=self.CHUNK)

            if self._paused or self._stop_flag:
                continue

            # ── read & deep-copy stats ──
            raw = bm.get_root_stats()
            stats_0 = {}
            for k, v in raw.items():
                val = v[0]
                stats_0[k] = val.copy() if hasattr(val, 'copy') else float(val)
            visits = bm.get_visits_count()[0].copy()

            # ── emit progress ──
            self.progress.emit(stats_0, visits)

            # ── check AI threshold ──
            if self._is_ai_turn and not self._ai_acted:
                if stats_0['root_N'] >= self._threshold:
                    self._ai_acted = True
                    self._paused = True          # auto-pause
                    elapsed = time.time() - self._t0
                    self.ai_ready.emit(stats_0, visits, elapsed)

        # thread exits
        self._idle.set()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════════════════════

class Connect4GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaZero Connect4 — HUD")
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
            mlh_slope=Def.mlh_slope, mlh_cap=Def.mlh_cap,
            mlh_threshold=Def.mlh_thr)
        self.player_color = 1
        self.move_count = 0
        self.animating = False

        # ── Widgets ─────────────────────────────────────────────────────────
        self.board = BoardWidget(self.env)
        self.status = StatusPanel()
        self.console = ParameterConsole()

        # Dual analysis: AI last move + human hint
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
        self.hint_btn.setChecked(True)       # on by default
        self._hint_visible = True

        # ── Layout ──────────────────────────────────────────────────────────
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(8)

        # ── Top row: board + console+buttons ────────────────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(16)
        top_row.addWidget(self.board)

        right_col = QVBoxLayout()
        right_col.setSpacing(4)
        right_col.addWidget(self.console, stretch=1)
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(6)
        btn_row1.addWidget(self.undo_btn)
        btn_row1.addWidget(self.restart_btn)
        btn_row1.addWidget(self.pause_btn)
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(6)
        btn_row2.addWidget(self.reset_btn)
        btn_row2.addWidget(self.hint_btn)
        btn_row2.addWidget(self.log_btn)
        right_col.addLayout(btn_row1)
        right_col.addLayout(btn_row2)
        top_row.addLayout(right_col)
        main_layout.addLayout(top_row)

        # ── Status (full width) ─────────────────────────────────────────────
        main_layout.addWidget(self.status)

        # ── Bottom row: AI panel | Hint panel (side by side) ────────────────
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)

        # AI last move panel
        ai_panel = QVBoxLayout()
        ai_panel.setSpacing(3)
        ai_title = QLabel(
            f"<font color='{C.CYAN}' style='font-family:Consolas;"
            f"font-size:11px;letter-spacing:2px;'>▸ AI LAST MOVE</font>")
        ai_title.setTextFormat(Qt.RichText)
        ai_panel.addWidget(ai_title)
        ai_panel.addWidget(self.ai_root_stats)
        ai_panel.addWidget(self.ai_child_table)
        bottom_row.addLayout(ai_panel)

        # Vertical separator
        vsep = QFrame()
        vsep.setFrameShape(QFrame.VLine)
        vsep.setStyleSheet(f"color: {C.BORDER};")
        vsep.setFixedWidth(1)
        bottom_row.addWidget(vsep)

        # Hint panel (wrapped in container for blur effect)
        self.hint_container = QWidget()
        hint_panel = QVBoxLayout(self.hint_container)
        hint_panel.setContentsMargins(0, 0, 0, 0)
        hint_panel.setSpacing(3)
        hint_title = QLabel(
            f"<font color='{C.GREEN_T}' style='font-family:Consolas;"
            f"font-size:11px;letter-spacing:2px;'>▸ YOUR ANALYSIS</font>")
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

        # ── Window size ─────────────────────────────────────────────────────
        total_w = self.board.width() + 320 + 48
        self.setFixedSize(total_w, 960)

        # ── MoveLog overlay drawer ──────────────────────────────────────────
        self._build_log_drawer()

        # ── Timers ──────────────────────────────────────────────────────────
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._step_anim)
        self.anim_target_row = -1
        self.anim_callback = None

        self.reload_timer = QTimer()
        self.reload_timer.setSingleShot(True)
        self.reload_timer.timeout.connect(self._reload_and_restart)

        self.settings_timer = QTimer()
        self.settings_timer.setSingleShot(True)
        self.settings_timer.timeout.connect(self._reload_and_restart)

        self.param_timer = QTimer()
        self.param_timer.setSingleShot(True)
        self.param_timer.timeout.connect(self._update_search_params)

        # Scan-line timer for AI thinking
        self.scan_timer = QTimer()
        self.scan_timer.timeout.connect(self._step_scan)

        # Continuous background MCTS search worker
        self.worker = ContinuousSearchWorker()
        self.worker.progress.connect(self._on_progress)
        self.worker.ai_ready.connect(self._on_ai_ready)
        self.worker.start()  # starts in paused state

        self._history = []

        # ── Load model ──────────────────────────────────────────────────────
        self._reload_model()

        # ── Connect signals ─────────────────────────────────────────────────
        # Model changes → full reload + restart
        _model_delayed = lambda _=None: self.settings_timer.start(400)
        self.console.network_cb.currentIndexChanged.connect(_model_delayed)
        self.console.model_type_cb.currentIndexChanged.connect(_model_delayed)

        # Search params → live setter (no tree destruction)
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
        self.reset_btn.clicked.connect(self.console.reset_defaults)
        self.pause_btn.clicked.connect(self._toggle_search_pause)
        self.hint_btn.clicked.connect(self._toggle_hint)
        self.log_btn.clicked.connect(self._toggle_log)

        self._start_game()

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        super().closeEvent(event)

    # ── Window background with grid ─────────────────────────────────────────
    def paintEvent(self, event):
        qp = QPainter(self)
        qp.fillRect(self.rect(), QColor(C.BG))
        # Subtle crosshair grid
        qp.setPen(QPen(QColor(0, 229, 255, 6), 1))
        for x in range(0, self.width(), 30):
            qp.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 30):
            qp.drawLine(0, y, self.width(), y)
        # Window corner brackets
        r = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        _draw_corner_brackets(qp, r, size=24, color=C.CYAN_CLR, width=1)

    # ═══════════════════════════════════════════════════════════════════════
    # MoveLog Drawer (overlay)
    # ═══════════════════════════════════════════════════════════════════════

    def _build_log_drawer(self):
        """Create an overlay panel for MoveLog, hidden by default."""
        d = QFrame(self)
        d.setStyleSheet(
            f"QFrame {{ background: {C.SURFACE}; "
            f"border-left: 1px solid {C.BORDER2}; }}")
        d.setFixedWidth(260)
        lay = QVBoxLayout(d)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        hdr = QHBoxLayout()
        title = QLabel(
            f"<font color='{C.CYAN}' style='font-family:Consolas;"
            f"font-size:11px;letter-spacing:2px;'>▸ MOVE LOG</font>")
        title.setTextFormat(Qt.RichText)
        hdr.addWidget(title)
        hdr.addStretch()
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet(
            f"QPushButton {{ border:none; color:{C.DIM}; font-size:14px; }}"
            f"QPushButton:hover {{ color:{C.CYAN}; }}")
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
        """Push search parameters to the live MCTS tree (no rebuild)."""
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
        # Cache size — recreate Python-side cache only
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
        self.board.win_cells = None
        self.board.anim_row = -1
        self.board.anim_col = -1
        self.board.anim_color = None
        self.board.interactive = True
        self.board.ghost_color = QColor(C.RED) if self.player_color == 1 else QColor(C.YEL)
        self.board.scanning = False
        self.board.scan_y = -1
        self.board.overlay_data = None
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
        self._update_analysis()
        self._update_turn_label()

        # Reset pause state
        self._search_paused = False
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("PAUSE")

        # Start continuous search immediately
        is_ai_first = (self.env.turn != self.player_color)
        if is_ai_first:
            self._start_scan()
        self._resume_search(is_ai_turn=is_ai_first)

    def _update_turn_label(self):
        if self.env.done():
            return
        if self.env.turn == self.player_color:
            color = C.RED_HEX if self.player_color == 1 else C.YEL_HEX
            self.status.set_turn("YOUR TURN", color)
            self.board.interactive = True
        else:
            self.status.set_turn("AI ACTIVE", C.CYAN)
            self.board.interactive = False

    def _update_analysis(self):
        """Update status panel NN WDL and steps from network forward pass."""
        with torch.no_grad():
            state = self.env.current_state()
            t = torch.from_numpy(state).float().to(self.net.device).unsqueeze(0)
            if t.dim() == 5:
                t = t.squeeze(1)
            _, vl, sl = self.net(t)

            vp = vl[0].exp().cpu().tolist()
            draw_pct = vp[0] * 100
            if self.player_color == 1:
                win_pct, lose_pct = vp[1] * 100, vp[2] * 100
            else:
                win_pct, lose_pct = vp[2] * 100, vp[1] * 100
            self.status.set_nn_rates(win_pct, draw_pct, lose_pct)

            sp = sl[0].exp().cpu()
            expected = (sp * torch.arange(len(sp), dtype=torch.float32)).sum().item()
            self.status.set_nn_steps(expected)

    def _update_status_mcts(self, stats_0):
        """Update status panel primary WDL/M from MCTS root stats."""
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
        max_y = self.board.CELL * 6
        self.board.scan_y += 4
        if self.board.scan_y > max_y:
            self.board.scan_y = 0
        self.board.update()

    # ── Continuous search helpers ──────────────────────────────────────────
    def _resume_search(self, is_ai_turn):
        """Set worker position from current env state and resume."""
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
            # Restore overlay from current hint stats
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
        """Handle user changing the sims threshold (no MCTS recreation)."""
        new_thr = self.console.n_playout_spin.value()
        self.worker.pause_and_wait()
        self.worker._threshold = new_thr
        # If AI turn and already past new threshold → act immediately
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
        """Periodic update from worker — refresh the live stats panel."""
        if self.animating:
            return
        self._update_status_mcts(stats_0)
        if self.env.turn != self.player_color:
            # AI thinking — update AI panel live
            self.board.overlay_data = None
            ai_turn = self.env.turn
            self.ai_root_stats.set_data(stats_0, chosen=-1, ai_turn=ai_turn)
            self.ai_child_table.update()
            elapsed = time.time() - self.worker._t0
            self.status.set_thinking(elapsed)
        else:
            # Human turn — update hint panel live
            human_turn = self.env.turn
            best = int(np.argmax(visits))
            self.hint_root_stats.set_data(stats_0, chosen=best,
                                          ai_turn=human_turn)
            self.hint_child_table.update()
            # Feed board overlay (only if hint visible)
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
        """Worker reached threshold — AI picks its action."""
        self.worker.pause_and_wait()          # already auto-paused
        self.status.set_thinking(elapsed)
        self._stop_scan()

        ai_turn = self.env.turn
        action = int(np.argmax(visits))

        # Final AI stats with chosen action highlighted
        self.ai_root_stats.set_data(stats_0, chosen=action, ai_turn=ai_turn)
        self.ai_child_table.update()

        # Prune tree → root is now human's position
        self.az_player.mcts.prune_roots(np.array([action], dtype=np.int32))

        # Extract hint from pruned tree
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
            # Set board overlay from hint data (only if hint visible)
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

        # Compute next board (env hasn't stepped yet, but tree already pruned)
        env_copy = self.env.copy()
        env_copy.step(action)
        next_board = env_copy.board[np.newaxis, ...]
        next_turns = np.array([env_copy.turn], dtype=np.int32)
        threshold = self.console.n_playout_spin.value()

        # Resume worker on human's position during animation
        if not env_copy.done():
            self.worker.set_position(self.az_player.mcts, self.az_player.pv_fn,
                                     next_board, next_turns,
                                     is_ai_turn=False, threshold=threshold)
            if not self._search_paused:
                self.worker.resume()

        # Animate piece drop
        row = self.board.find_drop_row(action)
        if row >= 0:
            self.board.last_move = (row, action)
            color = C.RED if -self.player_color == 1 else C.YEL
            self._start_anim(row, action, color,
                             lambda a=action: self._after_ai_anim(a))

    def _after_ai_anim(self, action):
        """Called after AI's drop animation finishes."""
        current_player = self.env.turn
        self.env.step(action)
        self.move_count += 1
        self.move_log.add_move(self.move_count, current_player, action)
        self._update_analysis()
        self.board.update()

        if self.env.done():
            self.worker.pause_and_wait()
            self._show_result()
            return

        self._update_turn_label()
        # Worker already running for hint since _on_ai_ready

    def _after_human_anim(self, col):
        """Called after human's drop animation finishes."""
        current_player = self.env.turn
        self.az_player.mcts.prune_roots(np.array([col], dtype=np.int32))
        self.env.step(col)
        self.move_count += 1
        self.move_log.add_move(self.move_count, current_player, col)
        self._update_analysis()
        self.board.update()

        if self.env.done():
            self._show_result()
            return

        self._update_turn_label()
        self._start_scan()
        self._resume_search(is_ai_turn=True)

    def _show_result(self):
        winner = self.env.winPlayer()
        self.board.interactive = False
        self.board.win_cells = self.board.find_win_line()
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
    # Animation
    # ═══════════════════════════════════════════════════════════════════════

    def _start_anim(self, target_row, col, color, callback):
        self.animating = True
        self.anim_target_row = target_row
        self.anim_callback = callback
        self.board.anim_row = -1
        self.board.anim_col = col
        self.board.anim_color = color
        self.anim_timer.start(ANIMATION_MS)

    def _step_anim(self):
        if self.board.anim_row < self.anim_target_row:
            self.board.anim_row += 1
            self.board.update()
        else:
            self.anim_timer.stop()
            self.board.anim_row = -1
            self.board.update()
            self.animating = False
            self.anim_callback()

    # ═══════════════════════════════════════════════════════════════════════
    # Undo
    # ═══════════════════════════════════════════════════════════════════════

    def _undo(self):
        if not self._history or self.animating:
            return
        if self.env.turn != self.player_color:
            return                            # can't undo while AI is moving
        self.worker.pause_and_wait()
        self._stop_scan()
        saved_env, saved_last, saved_count, saved_ai, saved_hint, saved_log = self._history.pop()
        self.env = saved_env
        self.board.env = saved_env
        self.az_player.mcts.reset_env(0)      # tree invalidated by undo
        self.board.last_move = saved_last
        self.board.win_cells = None
        self.board.anim_row = -1
        self.board.anim_col = -1
        self.board.overlay_data = None
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
        # Resume continuous search from restored position
        if not self.env.done():
            is_ai = (self.env.turn != self.player_color)
            if is_ai:
                self._start_scan()
            self._resume_search(is_ai_turn=is_ai)

    # ═══════════════════════════════════════════════════════════════════════
    # Mouse Input
    # ═══════════════════════════════════════════════════════════════════════

    def mousePressEvent(self, event):
        if self.animating or self.env.done():
            return
        if self.env.turn != self.player_color:
            return
        local = self.board.mapFromParent(event.pos())
        col = self.board.col_at(local.x())
        if col < 0 or col not in self.env.valid_move():
            return
        row = self.board.find_drop_row(col)
        if row < 0:
            return
        self.worker.pause_and_wait()
        self.board.overlay_data = None
        self._history.append((self.env.copy(), self.board.last_move, self.move_count,
                              self.ai_root_stats.snapshot(),
                              self.hint_root_stats.snapshot(),
                              self.move_log.snapshot()))
        self.undo_btn.setEnabled(True)
        self.board.last_move = (row, col)
        color = C.RED if self.player_color == 1 else C.YEL
        self._start_anim(row, col, color, lambda c=col: self._after_human_anim(c))


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    app = QApplication([])
    app.setFont(QFont("Consolas", 11))
    gui = Connect4GUI()
    gui.show()
    app.exec_()
