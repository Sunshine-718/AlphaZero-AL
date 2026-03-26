"""AlphaZero Othello — Frosted Glass GUI"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

from src.player import Human, AlphaZeroPlayer
from src.symmetry import (get_sym_config, apply_sym_board, apply_sym_action,
                          inverse_sym_visits)
_SYM_IDS = get_sym_config('Othello')['sym_ids']
_AUTO_SYM_ENSEMBLE = True
_AUTO_N_TREES = len(_SYM_IDS)
from src.environments import load
# On Windows, importing PyQt before torch can break torch DLL initialization.
import torch
import time
import math
import threading
import traceback
import numpy as np

from gui_common import (
    BaseMoveLog,
    NoWheelComboBox,
    NoWheelSlider,
    NoWheelSpinBox,
    SymmetrySearchWorker,
    _draw_glass,
    _make_slider,
    _scan_experiments as _scan_experiments_common,
    _sep,
    _sv,
    aggregate_root_stats_sym_ensemble,
    load_model_weights_only,
)

from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF, pyqtSignal, QThread
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QPen,
    QRadialGradient, QPainterPath, QLinearGradient, QConicalGradient)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QSlider,
    QPushButton, QCheckBox, QFrame, QTabWidget,
    QSizePolicy, QScrollArea, QTextEdit,
    QGraphicsBlurEffect)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

ENV_NAME = 'Othello'
PARAMS_DIR = './params'
CHUNK = 50


def _scan_experiments(env_name=ENV_NAME):
    return _scan_experiments_common(env_name, PARAMS_DIR)


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


def _parse_book(text):
    """Parse a book line like 'F5 D6 C3 ...' into a list of action indices."""
    actions = []
    for token in text.split():
        col = ord(token[0].upper()) - ord('A')
        row = int(token[1]) - 1
        actions.append(row * 8 + col)
    return actions


# Othello weak solution (black wins by 2 discs with perfect play from both sides)
BOOK_LINE = _parse_book(
    "F5 D6 C3 D3 C4 F4 F6 F3 E6 E7 D7 C5 B6 D8 C6 C7 "
    "D2 B5 A5 A6 A7 G5 E3 B4 C8 G6 G4 C2 E8 D1 F7 E2 "
    "G3 H4 F1 E1 F2 G1 B1 F8 G8 B3 H3 B2 H5 B7 A3 A4 "
    "A1 A2 C1 H2 H1 G2 B8 A8 G7 H8 H7 H6")


class Def:
    n_playout = 40000
    c_init = 1.4
    c_base = 2000
    fpu = 0.2
    alpha = 0.
    noise_eps = 0.
    cache = 10000
    vl_batch = 4
    score_utility_factor = 0.15
    score_scale = 8.0
    reuse_tree = True
    no_search = False
    use_book = False
    time_budget = 0.0  # 秒，0=禁用（按 sims 停），>0 时间到即落子

# Game modes
MODE_HVA = 0  # Human vs AI
MODE_HVH = 1  # Human vs Human
MODE_AVA = 2  # AI vs AI


# ═══════════════════════════════════════════════════════════════════════════════
# Theme — Frosted Glass / Glassmorphism
# ═══════════════════════════════════════════════════════════════════════════════

class C:
    # Backgrounds
    BG = "#0b0b1e"
    SURFACE = "rgba(255, 255, 255, 10)"
    SURFACE2 = "rgba(255, 255, 255, 15)"
    BORDER = "rgba(255, 255, 255, 25)"
    BORDER2 = "rgba(255, 255, 255, 40)"

    # Text
    TEXT = "#e0e0f5"
    DIM = "#9090b0"
    MUTED = "#505070"
    GREEN_T = "#4ade80"

    # Accents
    ACCENT = "#a78bfa"
    CYAN = "#38bdf8"
    MAGENTA = "#c084fc"
    GREEN = "#4ade80"
    RED_HEX = "#fb7185"
    YEL_HEX = "#fbbf24"

    # QColor objects — accent

    # Piece colors  — Black & White for Othello
    BLACK = QColor(30, 30, 50)
    BLACK_LT = QColor(80, 80, 110)
    BLACK_GLOW = QColor(100, 100, 160, 50)
    WHITE = QColor(230, 230, 245)
    WHITE_LT = QColor(255, 255, 255)
    WHITE_GLOW = QColor(230, 230, 245, 50)

    # Board
    BOARD_BG = QColor(12, 12, 35, 200)
    CELL_BG = QColor(8, 8, 25, 180)
    GRID_CORE = QColor(255, 255, 255, 18)
    GRID_GLOW = QColor(167, 139, 250, 12)
    HOVER = QColor(167, 139, 250, 25)

    # Glass

    # Board surface — dark green for Othello
    BOARD_GREEN = QColor(0, 80, 50, 180)
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





# ═══════════════════════════════════════════════════════════════════════════════
# Attention Map Extractor (zero-intrusion via forward hooks)
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionExtractor:
    """Extract attention weights from the Attention layer via forward hooks.

    Hooks q_norm and k_norm outputs to capture normalized Q/K,
    then computes attention = softmax(Q @ K^T / sqrt(head_dim)).
    Supports symmetry-ensemble: batch K symmetry-transformed states,
    inverse-permute each attention matrix, then average.
    """

    def __init__(self):
        self._q = None   # (B, seq, H, D) — q_norm output
        self._k = None   # (B, seq, H, D) — k_norm output
        self._gate_raw = None  # (B, seq, H) — raw gate logits (before sigmoid)
        self._num_heads = None
        self._handles = []
        self._capture_thread_id = None
        # Precompute inverse permutation tables for Othello symmetries
        # All Othello Klein-4 symmetries are self-inverse (involutions)
        from src.symmetry import SYM_REGISTRY
        reg = SYM_REGISTRY['Othello']
        self._sym_ids = reg['sym_ids']       # [0, 2, 6, 7]
        self._perms = reg['board_perms']     # {sym_id: np.ndarray(64,)}

    def _hook_q(self, module, input, output):
        if threading.get_ident() != self._capture_thread_id:
            return
        self._q = output.detach()

    def _hook_k(self, module, input, output):
        if threading.get_ident() != self._capture_thread_id:
            return
        self._k = output.detach()

    def _hook_gate(self, module, input, output):
        if threading.get_ident() != self._capture_thread_id:
            return
        gate = output.detach()
        # Newer Othello nets expose gate logits directly via gate_proj: (B, S, H).
        # Older experiments may still use a fused qkvg projection, where gate lives
        # in the last H channels.
        if self._num_heads is not None and gate.shape[-1] != self._num_heads:
            gate = gate[..., -self._num_heads:]
        self._gate_raw = gate  # (B, S, H)

    def attach(self, net):
        """Register hooks on net.hidden[-1].attn.{q_norm, k_norm, gate_proj/qkvg_proj}."""
        self.detach()
        attn = net.hidden[-1].attn
        self._num_heads = attn.num_heads
        self._handles.append(attn.q_norm.register_forward_hook(self._hook_q))
        self._handles.append(attn.k_norm.register_forward_hook(self._hook_k))
        if hasattr(attn, 'gate_proj'):
            self._handles.append(attn.gate_proj.register_forward_hook(self._hook_gate))
        elif hasattr(attn, 'qkvg_proj'):
            self._handles.append(attn.qkvg_proj.register_forward_hook(self._hook_gate))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._capture_thread_id = None
        self._q = self._k = self._gate_raw = None

    def begin_capture(self):
        self._capture_thread_id = threading.get_ident()
        self._q = self._k = self._gate_raw = None

    def end_capture(self):
        self._capture_thread_id = None

    def get_weights(self):
        """Compute attention weights from captured Q, K.

        If batch size == K (symmetry ensemble), inverse-permute each
        attention matrix and average. Otherwise return single result.

        Returns: np.ndarray (num_heads, 64, 64) or None.
        """
        if self._q is None or self._k is None:
            return None
        if self._q.shape != self._k.shape:
            return None
        # q_norm output: (B, S, H, D) — S=64 or 65 (with pass token)
        B, S = self._q.shape[0], self._q.shape[1]
        q = self._q.permute(0, 2, 1, 3)  # (B, H, S, D)
        k = self._k.permute(0, 2, 1, 3)  # (B, H, S, D)
        head_dim = q.shape[-1]
        # (B, H, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1).cpu().numpy()

        K = len(self._sym_ids)
        if B == K and K > 1:
            # Symmetry ensemble: inverse-permute rows & cols, then average
            # Extend perm to handle pass token (position 64 maps to itself)
            acc = weights[0]  # sym_id=0 is identity
            for i in range(1, K):
                perm = self._perms[self._sym_ids[i]]
                if S > 64:
                    perm = np.append(perm, np.arange(64, S))
                w = weights[i]
                acc = acc + w[:, perm][:, :, perm]
            weights = acc / K
        else:
            weights = weights[0]  # (H, S, S)

        # Strip pass token — return only board positions (H, 64, 64)
        return weights[:, :64, :64]

    def get_gate_scores(self):
        """Compute gate scores with symmetry ensemble.

        Returns: np.ndarray (64,) — mean gate score per position, or None.
        """
        if self._gate_raw is None:
            return None
        # gate_raw: (B, S, H) — S=64 or 65 (with pass token)
        gate = torch.sigmoid(self._gate_raw).cpu().numpy()  # (B, S, H)
        K = len(self._sym_ids)
        if gate.shape[0] == K and K > 1:
            acc = gate[0, :64]  # (64, H) — strip pass token
            for i in range(1, K):
                perm = self._perms[self._sym_ids[i]]
                acc = acc + gate[i, :64][perm]  # inverse-permute board positions only
            gate_avg = acc / K  # (64, H)
        else:
            gate_avg = gate[0, :64]  # (64, H)
        return gate_avg.mean(axis=1)  # (64,) — average across heads


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

        # Attention heatmap
        self.attn_weights = None   # (num_heads, 64, 64) or None
        self.gate_scores = None    # (64,) or None — per-position gate score
        self.attn_head = -1        # -1=mean, 0..H-1=specific head, 'gate'=gate overlay
        self.attn_visible = False  # controlled by checkbox

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
        self._draw_attention(qp)
        self._draw_valid_moves(qp)
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
            dk = QColor(C.BLACK)
            lt = QColor(C.BLACK_LT)
            glow_c = QColor(C.BLACK_GLOW)
        else:
            # White piece
            dk = QColor(C.WHITE)
            lt = QColor(C.WHITE_LT)
            glow_c = QColor(C.WHITE_GLOW)
        dk.setAlpha(alpha)
        lt.setAlpha(alpha)

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

    def _draw_attention(self, qp):
        """Draw attention heatmap or gate score overlay on the board."""
        if not self.attn_visible:
            return

        # Gate mode: draw per-position gate score (no query dependency)
        if self.attn_head == 'gate':
            if self.gate_scores is None:
                return
            heat = self.gate_scores  # (64,) already in [0, 1]
            m, cs = self.MARGIN, self.CELL
            for idx in range(64):
                r, c = idx // 8, idx % 8
                val = float(heat[idx])
                # Red (closed) to Green (open)
                red = int((1 - val) * 200)
                green = int(val * 200)
                alpha = 100 + int(abs(val - 0.5) * 2 * 80)  # stronger near 0 or 1
                x = m + c * cs
                y = m + r * cs
                qp.fillRect(int(x), int(y), cs, cs, QColor(red, green, 60, alpha))
            return

        # Attention mode
        if self.attn_weights is None:
            return
        w = self.attn_weights  # (H, 64, 64)
        # Select head
        if self.attn_head < 0:
            attn = w.mean(axis=0)  # (64, 64)
        else:
            attn = w[min(self.attn_head, w.shape[0] - 1)]  # (64, 64)
        # Query selection: hover cell or global mean
        if self.hover_cell is not None:
            qr, qc = self.hover_cell
            heat = attn[qr * 8 + qc]  # (64,) — attention from query to all keys
        else:
            heat = attn.mean(axis=0)  # (64,) — average over all queries
        # Normalize to [0, 1] for visualization
        vmin, vmax = heat.min(), heat.max()
        if vmax - vmin > 1e-8:
            heat = (heat - vmin) / (vmax - vmin)
        else:
            heat = np.zeros_like(heat)
        # Draw heatmap cells
        m, cs = self.MARGIN, self.CELL
        for idx in range(64):
            r, c = idx // 8, idx % 8
            val = float(heat[idx])
            if val < 0.02:
                continue
            alpha = int(val * 140)
            x = m + c * cs
            y = m + r * cs
            qp.fillRect(int(x), int(y), cs, cs, QColor(167, 139, 250, alpha))
        # Draw query marker (small diamond) if hovering
        if self.hover_cell is not None:
            qr, qc = self.hover_cell
            cx = m + qc * cs + cs // 2
            cy = m + qr * cs + cs // 2
            d = 6
            path = QPainterPath()
            path.moveTo(cx, cy - d)
            path.lineTo(cx + d, cy)
            path.lineTo(cx, cy + d)
            path.lineTo(cx - d, cy)
            path.closeSubpath()
            qp.setPen(Qt.NoPen)
            qp.setBrush(QColor(255, 255, 255, 200))
            qp.drawPath(path)

    def _draw_valid_moves(self, qp):
        """Draw small dots on all legal move cells (always visible)."""
        if not self.interactive or not self._valid_set:
            return
        # Skip if overlay is active — overlay already marks legal moves
        if self.overlay_data is not None:
            return
        for action in self._valid_set:
            if action >= 64:  # skip pass action
                continue
            r, c = action // 8, action % 8
            cx = self.MARGIN + c * self.CELL + self.CELL // 2
            cy = self.MARGIN + r * self.CELL + self.CELL // 2
            qp.setBrush(QColor(167, 139, 250, 40))
            qp.setPen(Qt.NoPen)
            qp.drawEllipse(QPointF(cx, cy), 5, 5)

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
        self.player_color = 1

    def set_rates(self, win, draw, lose):
        self.w_rate, self.d_rate, self.l_rate = win, draw, lose
        self.update()

    def set_player_color(self, player_color):
        self.player_color = 1 if player_color == 1 else -1
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
        win_clr = C.BLACK if self.player_color == 1 else C.WHITE
        lose_clr = C.WHITE if self.player_color == 1 else C.BLACK
        for ratio, clr in [
            (self.w_rate, win_clr), (self.d_rate, QColor(C.CYAN)),
            (self.l_rate, lose_clr)
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


class UtilBar(QWidget):
    MAX_DIFF = 64.0  # max disc diff for bar display

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(14)
        self.util = 0.0      # atan-mapped utility from MCTS
        self.score_scale = 8.0
        self.player_color = 1

    def set_player_color(self, player_color):
        self.player_color = 1 if player_color == 1 else -1
        self.update()

    def set_util(self, util):
        self.util = max(-0.999, min(float(util), 0.999))
        self.update()

    def _to_diff(self):
        return self.score_scale * math.tan(self.util * math.pi / 2.0)

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        diff = self._to_diff()
        text = f"{diff:+.1f}"
        fm = qp.fontMetrics()
        tw = fm.horizontalAdvance(text) + 8
        bw = max(12, self.width() - tw)
        h = self.height()
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, bw, h), 4, 4)
        qp.setClipPath(path)
        qp.fillRect(0, 0, bw, h, QColor(255, 255, 255, 8))
        center = bw // 2
        qp.fillRect(center, 0, 1, h, QColor(255, 255, 255, 35))
        ratio = min(abs(diff) / self.MAX_DIFF, 1.0)
        fill_w = int(center * ratio)
        if fill_w > 0:
            if diff > 0:
                base = C.BLACK if self.player_color == 1 else C.WHITE
                x0, x1 = center, center + fill_w
            else:
                base = C.WHITE if self.player_color == 1 else C.BLACK
                x0, x1 = center - fill_w, center
            grad = QLinearGradient(x0, 0, x1, 0)
            grad.setColorAt(0, QColor(base.red(), base.green(), base.blue(), 210))
            grad.setColorAt(1, QColor(base.red() // 2, base.green() // 2, base.blue() // 2, 210))
            qp.fillRect(int(x0), 0, max(1, int(x1 - x0)), h, grad)
        qp.setClipping(False)
        if diff > 0:
            text_color = C.BLACK if self.player_color == 1 else C.WHITE
        elif diff < 0:
            text_color = C.WHITE if self.player_color == 1 else C.BLACK
        else:
            text_color = QColor(C.ACCENT)
        qp.setPen(text_color)
        qp.setFont(QFont("Consolas", 10))
        qp.drawText(bw + 4, 0, tw, h, Qt.AlignVCenter | Qt.AlignLeft, text)


class WinRateTrend(QWidget):
    """Q 值趋势图：按手数绘制 Q = (win-lose)/100 曲线，范围 [-1, 1]。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(80)
        self.player_color = 1
        # 每个元素: (move_num, win%, draw%, lose%)  — 均为玩家视角 0~100
        self._data = []

    def set_player_color(self, c):
        self.player_color = 1 if c == 1 else -1
        self.update()

    def record(self, move_num, win_pct, draw_pct, lose_pct):
        self._data.append((move_num, win_pct, draw_pct, lose_pct))
        self.update()

    def truncate(self, move_num):
        """回退到 move_num（undo 用）。"""
        self._data = [d for d in self._data if d[0] <= move_num]
        self.update()

    def clear(self):
        self._data.clear()
        self.update()

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # 背景
        _draw_glass(qp, self.rect(), radius=8, fill_alpha=10, border_alpha=22)

        n = len(self._data)
        margin_l, margin_r, margin_t, margin_b = 28, 6, 10, 14
        cw = w - margin_l - margin_r
        ch = h - margin_t - margin_b

        if cw < 10 or ch < 10:
            return

        # Q 值 [-1, 1] → Y 坐标
        def y_of_q(q):
            return margin_t + (1.0 - q) / 2.0 * ch

        # 0 参考线（均势）
        qp.setPen(QPen(QColor(255, 255, 255, 20), 1, Qt.DashLine))
        y0 = y_of_q(0)
        qp.drawLine(margin_l, int(y0), margin_l + cw, int(y0))

        # Y 轴刻度
        qp.setPen(QColor(255, 255, 255, 40))
        qp.setFont(QFont("Consolas", 7))
        qp.drawText(0, margin_t - 2, margin_l - 4, 12, Qt.AlignRight | Qt.AlignVCenter, "+1")
        qp.drawText(0, int(y0) - 6, margin_l - 4, 12, Qt.AlignRight | Qt.AlignVCenter, "0")
        qp.drawText(0, margin_t + ch - 10, margin_l - 4, 12, Qt.AlignRight | Qt.AlignVCenter, "-1")

        if n == 0:
            qp.setPen(QColor(255, 255, 255, 30))
            qp.setFont(QFont("Consolas", 9))
            qp.drawText(self.rect(), Qt.AlignCenter, "Q VALUE TREND")
            return

        # X 轴坐标映射
        max_move = max(d[0] for d in self._data)
        def x_of(mv):
            if max_move <= 1:
                return margin_l + cw // 2
            return margin_l + int((mv / max_move) * cw)

        # Q = (win% - lose%) / 100，范围 [-1, 1]
        line_clr = QColor(C.CYAN)
        pts = [QPointF(x_of(d[0]), y_of_q((d[1] - d[3]) / 100.0)) for d in self._data]

        if n == 1:
            # 单点：画圆点
            qp.setPen(Qt.NoPen)
            qp.setBrush(line_clr)
            qp.drawEllipse(pts[0], 3, 3)
        else:
            # 曲线与零线之间半透明填充
            fill_path = QPainterPath()
            fill_path.moveTo(QPointF(pts[0].x(), y_of_q(0)))
            for pt in pts:
                fill_path.lineTo(pt)
            fill_path.lineTo(QPointF(pts[-1].x(), y_of_q(0)))
            fill_path.closeSubpath()
            qp.fillPath(fill_path, QColor(line_clr.red(), line_clr.green(), line_clr.blue(), 25))

            # Q 值曲线
            qp.setPen(QPen(line_clr, 1.5))
            for i in range(n - 1):
                qp.drawLine(pts[i], pts[i + 1])

        # 末端圆点
        qp.setPen(Qt.NoPen)
        qp.setBrush(line_clr)
        qp.drawEllipse(pts[-1], 2.5, 2.5)

        # X 轴标签
        qp.setPen(QColor(255, 255, 255, 40))
        qp.setFont(QFont("Consolas", 7))
        if n > 0:
            qp.drawText(x_of(self._data[0][0]) - 8, margin_t + ch + 1, 20, 12,
                         Qt.AlignCenter, str(self._data[0][0]))
        if n > 1:
            qp.drawText(x_of(max_move) - 8, margin_t + ch + 1, 20, 12,
                         Qt.AlignCenter, str(max_move))


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
        self.per_tree_n = 0
        self.root_q = 0.0
        self.root_m = 0.0
        self.wdl = None
        self.chosen = -1
        self.ai_turn = 1
        self.score_scale = 8.0

    @staticmethod
    def _util_to_diff(util, scale):
        """Inverse-atan: convert score utility back to approximate disc diff."""
        clamped = max(-0.999, min(float(util), 0.999))
        return scale * math.tan(clamped * math.pi / 2.0)

    def set_data(self, stats, chosen=-1, ai_turn=1):
        self.visits = stats['N'].copy()
        self.q_values = stats['Q'].copy()
        self.prior = stats['prior'].copy()
        self.child_m = stats['M'].copy()
        self.child_d = stats['D'].copy()
        self.child_p1w = stats['P1W'].copy()
        self.child_p2w = stats['P2W'].copy()
        self.root_n = float(stats['root_N'])
        self.per_tree_n = float(stats.get('per_tree_N', stats['root_N']))
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
        self.per_tree_n = 0
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
                    root_n=self.root_n, per_tree_n=self.per_tree_n,
                    root_q=self.root_q, root_m=self.root_m,
                    wdl=self.wdl.copy(), chosen=self.chosen,
                    ai_turn=self.ai_turn, score_scale=self.score_scale)

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
        self.per_tree_n = snap.get('per_tree_n', snap['root_n'])
        self.root_q = snap['root_q']
        self.root_m = snap['root_m']
        self.wdl = snap['wdl']
        self.chosen = snap['chosen']
        self.ai_turn = snap['ai_turn']
        self.score_scale = snap.get('score_scale', 8.0)
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

        ptn = int(self.per_tree_n)
        rn = int(self.root_n)
        if ptn > 0 and ptn != rn:
            n_trees = max(1, round(rn / ptn))
            n_text = f"N:{ptn}\u00d7{n_trees}={rn}"
        else:
            n_text = f"N:{rn}"

        parts = [
            (n_text, C.ACCENT),
            (f"Q:{self.root_q:+.2f}", C.GREEN_T),
            (f"Df:{self._util_to_diff(self.root_m, self.score_scale):+.1f}", C.YEL_HEX),
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

    COLS = ['Pos', 'N', 'N%', 'Q', 'W%', 'D%', 'L%', 'Df', 'P']

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

        ratios = [0.07, 0.12, 0.08, 0.10, 0.10, 0.09, 0.10, 0.10, 0.09]
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
            m_val = s._util_to_diff(s.child_m[idx], s.score_scale)
            prior = s.prior[idx] * 100

            cells = [
                (action_label(idx), C.ACCENT if is_chosen else C.MUTED),
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
                (f"{m_val:+.1f}" if n > 0 else '-',
                    C.YEL_HEX if n > 0 else C.MUTED),
                (f"{prior:.1f}" if prior > 0.05 else '<.1',
                    C.MAGENTA if prior > 5 else C.DIM),
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
        self.player_color = 1
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
        util_title = QLabel(f"<font color='{C.DIM}' style='font-family:Consolas;font-size:10px;'>DISC DIFF</font>")
        util_title.setAlignment(Qt.AlignCenter)
        util_title.setTextFormat(Qt.RichText)
        right.addWidget(util_title)
        self.util_bar = UtilBar()
        self.util_bar.setMinimumWidth(100)
        self.util_bar.set_player_color(self.player_color)
        right.addWidget(self.util_bar)
        self.nn_diff_lbl = QLabel("")
        self.nn_diff_lbl.setAlignment(Qt.AlignCenter)
        self.nn_diff_lbl.setTextFormat(Qt.RichText)
        right.addWidget(self.nn_diff_lbl)
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

    def set_player_color(self, player_color):
        self.player_color = 1 if player_color == 1 else -1
        self.wdl_bar.set_player_color(player_color)
        self.util_bar.set_player_color(player_color)

    def set_disc_count(self, black, white):
        if self.player_color == 1:
            my_count, opp_count = black, white
            my_color = C.BLACK_LT.name()
            opp_color = C.WHITE.name()
        else:
            my_count, opp_count = white, black
            my_color = C.WHITE.name()
            opp_color = C.BLACK_LT.name()
        self.disc_lbl.setText(
            f"<font color='{my_color}' style='font-family:Consolas;font-size:11px;'>"
            f"\u25cf {my_count}</font>"
            f"<font color='{C.DIM}' style='font-family:Consolas;font-size:11px;'> - </font>"
            f"<font color='{opp_color}' style='font-family:Consolas;font-size:11px;'>"
            f"\u25cf {opp_count}</font>")

    def set_nn_rates(self, win, draw, lose):
        self.nn_wdl_lbl.setText(
            f"<font color='{C.MUTED}' style='font-family:Consolas;font-size:9px;'>"
            f"nn {win:.1f} / {draw:.1f} / {lose:.1f}</font>")

    def set_mcts_util(self, util):
        self.util_bar.set_util(util)

    def set_nn_diff(self, diff):
        self.nn_diff_lbl.setText(
            f"<font color='{C.MUTED}' style='font-family:Consolas;font-size:9px;'>"
            f"nn disc {diff:+.1f}</font>")

    def clear_mcts(self):
        for lbl, prefix, color in [(self.win_lbl, 'WIN', C.GREEN),
                                   (self.draw_lbl, 'DRAW', C.CYAN),
                                   (self.lose_lbl, 'LOSE', C.RED_HEX)]:
            lbl.setText(f"<font color='{color}' style='font-family:Consolas;"
                        f"font-size:10px;'>{prefix}</font><br>"
                        f"<font color='{color}'>--%</font>")
        self.wdl_bar.set_rates(0, 0, 0)
        self.util_bar.set_util(0)
        self.nn_wdl_lbl.setText("")
        self.nn_diff_lbl.setText("")

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
        self._build_aux_tab()

    def _wrap_scroll_tab(self, content):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.viewport().setStyleSheet("background: transparent;")
        scroll.setWidget(content)
        return scroll

    def _build_game_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        def _label(text):
            lbl = QLabel(
                f"<font color='{C.DIM}' style='font-family:Consolas;font-size:10px;"
                f"letter-spacing:1px;'>{text}</font>")
            return lbl

        # MODE
        lay.addWidget(_label("MODE"))
        self.mode_cb = NoWheelComboBox()
        self.mode_cb.addItems(["Human vs AI", "Human vs Human", "AI vs AI"])
        lay.addWidget(self.mode_cb)

        # PLAYER (hidden in AvA)
        self._player_label = _label("PLAYER")
        lay.addWidget(self._player_label)
        self.player_cb = NoWheelComboBox()
        self.player_cb.addItems(["Human Black (first)", "Human White (second)"])
        lay.addWidget(self.player_cb)

        # WEIGHTS (primary — full filename from params/)
        self._weights_label = _label("WEIGHTS")
        lay.addWidget(self._weights_label)
        self.model_type_cb = NoWheelComboBox()
        self.model_type_cb.addItems(_scan_experiments())
        lay.addWidget(self.model_type_cb)

        # ⚪ WEIGHTS (second — only shown in AvA for White AI)
        self._weights2_label = _label("⚪ WEIGHTS")
        lay.addWidget(self._weights2_label)
        self.model_type_cb2 = NoWheelComboBox()
        self.model_type_cb2.addItems(_scan_experiments())
        lay.addWidget(self.model_type_cb2)
        self._weights2_label.hide()
        self.model_type_cb2.hide()

        lay.addStretch()
        self.tabs.addTab(w, "GAME")

    def _update_game_visibility(self, mode):
        """Show/hide GAME tab widgets based on mode."""
        is_ava = (mode == MODE_AVA)
        self._player_label.setVisible(not is_ava)
        self.player_cb.setVisible(not is_ava)
        self._weights_label.setText(
            f"<font color='{C.DIM}' style='font-family:Consolas;font-size:10px;"
            f"letter-spacing:1px;'>{'⚫ WEIGHTS' if is_ava else 'WEIGHTS'}</font>")
        self._weights2_label.setVisible(is_ava)
        self.model_type_cb2.setVisible(is_ava)

    def _build_mcts_tab(self):
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(8, 8, 8, 6)
        lay.setSpacing(10)

        row = QHBoxLayout()
        lbl = QLabel("sims")
        lbl.setFixedWidth(76)
        lbl.setStyleSheet(f"color: {C.DIM}; font-size: 11px; font-family: Consolas;")
        lbl.setToolTip("MCTS simulations per move")
        row.addWidget(lbl)
        self.n_playout_spin = NoWheelSpinBox()
        self.n_playout_spin.setRange(1, 999999)
        self.n_playout_spin.setValue(Def.n_playout)
        row.addWidget(self.n_playout_spin)
        lay.addLayout(row)

        row_tb = QHBoxLayout()
        lbl_tb = QLabel("time(s)")
        lbl_tb.setFixedWidth(76)
        lbl_tb.setStyleSheet(f"color: {C.DIM}; font-size: 11px; font-family: Consolas;")
        lbl_tb.setToolTip("Time budget per AI move in seconds.\n"
                          "0 = disabled (use sims count).\n"
                          "When set, AI moves after this many seconds\n"
                          "or when sims reached, whichever comes first.")
        row_tb.addWidget(lbl_tb)
        self.time_budget_spin = QDoubleSpinBox()
        self.time_budget_spin.setRange(0, 9999)
        self.time_budget_spin.setDecimals(1)
        self.time_budget_spin.setSingleStep(0.5)
        self.time_budget_spin.setValue(Def.time_budget)
        self.time_budget_spin.setSpecialValueText("off")
        row_tb.addWidget(self.time_budget_spin)
        lay.addLayout(row_tb)

        row3 = QHBoxLayout()
        lbl3 = QLabel("vl_batch")
        lbl3.setFixedWidth(76)
        lbl3.setStyleSheet(f"color: {C.DIM}; font-size: 11px; font-family: Consolas;")
        lbl3.setToolTip("Virtual Loss batch size per tree per iteration.\n"
                         ">1 enables VL tree parallelism for faster NN batching.")
        row3.addWidget(lbl3)
        self.vl_batch_spin = NoWheelSpinBox()
        self.vl_batch_spin.setRange(1, 256)
        self.vl_batch_spin.setValue(Def.vl_batch)
        row3.addWidget(self.vl_batch_spin)
        lay.addLayout(row3)

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

        auto_sym = QLabel(
            f"<font color='{C.MUTED}' style='font-family:Consolas;font-size:10px;'>"
            f"symmetry ensemble: auto ({_AUTO_N_TREES} views)</font>")
        auto_sym.setTextFormat(Qt.RichText)
        lay.addWidget(auto_sym)

        self.book_check = QCheckBox("BOOK")
        self.book_check.setChecked(Def.use_book)
        self.book_check.setToolTip(
            "Follow the weak solution book line.\n"
            "If opponent deviates, switch to AlphaZero for the rest of the game.")
        lay.addWidget(self.book_check)

        self.reuse_tree_check = QCheckBox("REUSE TREE")
        self.reuse_tree_check.setChecked(Def.reuse_tree)
        self.reuse_tree_check.setToolTip("Reuse subtree from previous move (prune) vs fresh tree each move (reset)")
        lay.addWidget(self.reuse_tree_check)

        self.no_search_check = QCheckBox("NO SEARCH")
        self.no_search_check.setChecked(Def.no_search)
        self.no_search_check.setToolTip("Use raw NN policy without MCTS search (single root expansion only)")
        lay.addWidget(self.no_search_check)

        lay.addStretch()
        self.tabs.addTab(self._wrap_scroll_tab(content), "MCTS")

    def _build_aux_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        info = QLabel(f"<font color='{C.DIM}' style='font-family:Consolas;font-size:10px;'>"
                      f"Score Utility (KataGo-style)</font>")
        info.setTextFormat(Qt.RichText)
        lay.addWidget(info)

        self.score_factor_sl = _make_slider(lay, "score_factor", 0, 1, 0.05, Def.score_utility_factor,
                                             tooltip="KataGo-style score utility weight")
        self.score_scale_sl = _make_slider(lay, "score_scale", 1, 32, 1, Def.score_scale,
                                            tooltip="Score atan mapping scale")

        # ── Attention Map ──
        sep = _sep()
        lay.addWidget(sep)
        attn_info = QLabel(f"<font color='{C.DIM}' style='font-family:Consolas;font-size:10px;'>"
                           f"Attention Heatmap</font>")
        attn_info.setTextFormat(Qt.RichText)
        lay.addWidget(attn_info)

        attn_row = QHBoxLayout()
        self.attn_check = QCheckBox("Show")
        self.attn_check.setChecked(False)
        self.attn_check.setToolTip("Overlay attention heatmap on board")
        attn_row.addWidget(self.attn_check)

        self.attn_head_cb = QComboBox()
        self.attn_head_cb.setToolTip("Select attention head or gate score to visualize")
        self.attn_head_cb.setFixedWidth(90)
        self.set_attention_head_count(0)
        attn_row.addWidget(self.attn_head_cb)
        attn_row.addStretch()
        lay.addLayout(attn_row)

        lay.addStretch()
        self.tabs.addTab(w, "Aux")

    def set_attention_head_count(self, num_heads):
        num_heads = max(0, int(num_heads))
        current_text = self.attn_head_cb.currentText() or "Mean"
        old_block = self.attn_head_cb.blockSignals(True)
        self.attn_head_cb.clear()
        self.attn_head_cb.addItems(["Mean"] + [f"Head {i}" for i in range(num_heads)] + ["Gate"])

        idx = self.attn_head_cb.findText(current_text)
        if idx < 0 and current_text.startswith("Head "):
            try:
                head_idx = int(current_text.split()[-1])
                idx = min(head_idx + 1, num_heads) if num_heads > 0 else 0
            except ValueError:
                idx = -1
        if idx < 0:
            idx = self.attn_head_cb.count() - 1 if current_text == "Gate" else 0

        self.attn_head_cb.setCurrentIndex(max(0, idx))
        self.attn_head_cb.blockSignals(old_block)

    def reset_defaults(self):
        self.mode_cb.setCurrentIndex(MODE_HVA)
        self.model_type_cb.setCurrentIndex(0)
        self.model_type_cb2.setCurrentIndex(0)
        self.player_cb.setCurrentIndex(0)
        self.n_playout_spin.setValue(Def.n_playout)
        self.time_budget_spin.setValue(Def.time_budget)
        self.vl_batch_spin.setValue(Def.vl_batch)
        self.c_init_sl.setValue(int(Def.c_init * self.c_init_sl._scale))
        self.c_base_sl.setValue(int(Def.c_base * self.c_base_sl._scale))
        self.fpu_sl.setValue(int(Def.fpu * self.fpu_sl._scale))
        self.alpha_sl.setValue(int(Def.alpha * self.alpha_sl._scale))
        self.eps_sl.setValue(int(Def.noise_eps * self.eps_sl._scale))
        self.cache_sl.setValue(int(Def.cache * self.cache_sl._scale))
        self.book_check.setChecked(Def.use_book)
        self.reuse_tree_check.setChecked(Def.reuse_tree)
        self.no_search_check.setChecked(Def.no_search)
        self.score_factor_sl.setValue(int(Def.score_utility_factor * self.score_factor_sl._scale))
        self.score_scale_sl.setValue(int(Def.score_scale * self.score_scale_sl._scale))
        self.attn_head_cb.setCurrentIndex(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Move Log
# ═══════════════════════════════════════════════════════════════════════════════

class MoveLog(BaseMoveLog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Consolas", 10))

    def format_move(self, num, player, action):
        color = C.TEXT if player == 1 else C.DIM
        symbol = "\u25cf" if player == 1 else "\u25cb"  # ● / ○
        return (
            f'<span style="color:{C.MUTED}">#{num:2d}</span> '
            f'<span style="color:{color}"><b>{symbol}</b></span> '
            f'<span style="color:{C.DIM}">&rarr; {action_label(action)}</span>')



# ═══════════════════════════════════════════════════════════════════════════════
# MCTS Search Worker (background thread)
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate_root_stats_sym_ensemble(raw, sym_ids):
    return aggregate_root_stats_sym_ensemble(raw, sym_ids, 'Othello')


class _LegacyContinuousSearchWorker(QThread):
    CHUNK = CHUNK

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
        self._n_trees = 1
        self._sym_ensemble = False
        self._vl_batch = 1
        self._time_budget = 0.0
        self._t0 = 0.0
        self._ai_acted = False
        self._paused = True
        self._stop_flag = False
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()


class ContinuousSearchWorker(SymmetrySearchWorker):
    def __init__(self, parent=None):
        super().__init__('Othello', _SYM_IDS, chunk=CHUNK, parent=parent)

    def set_position(self, bmcts, pv_fn, board, turns, is_ai_turn, threshold,
                     n_trees=1, sym_ensemble=False, vl_batch=1, time_budget=0.0):
        self._bmcts = bmcts
        self._pv_fn = pv_fn
        self._board = np.ascontiguousarray(board, dtype=np.int8)
        self._turns = np.ascontiguousarray(turns, dtype=np.int32)
        self._is_ai_turn = is_ai_turn
        self._threshold = threshold
        self._n_trees = n_trees
        self._sym_ensemble = sym_ensemble
        self._vl_batch = vl_batch
        self._time_budget = time_budget
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
                             n_playout=self.CHUNK, vl_batch=self._vl_batch)

            if self._paused or self._stop_flag:
                continue

            raw = bm.get_root_stats()
            all_visits = bm.get_visits_count()

            if self._sym_ensemble:
                stats_0 = _aggregate_root_stats_sym_ensemble(raw, _SYM_IDS)
                # 逆变换每棵树的 visits 回原始坐标后求和
                transformed_visits = all_visits.copy()
                for i, sid in enumerate(_SYM_IDS):
                    transformed_visits[i] = inverse_sym_visits(all_visits[i], sid, 'Othello')
                visits = transformed_visits.sum(axis=0).copy()

            self.progress.emit(stats_0, visits)

            if self._is_ai_turn and not self._ai_acted:
                per_tree_n = float(raw['root_N'][0])
                elapsed = time.time() - self._t0

                # 时间预算触发：到时间即落子（至少搜索 8 步）
                time_up = (self._time_budget > 0
                           and elapsed >= self._time_budget
                           and per_tree_n >= 8)

                # Early exit：剩余预算全部加到第二名也追不上第一名
                sorted_v = np.sort(visits)[::-1]
                only_one = len(sorted_v) < 2 or sorted_v[1] == 0
                remaining = max(0, self._threshold - per_tree_n) * self._n_trees
                visit_converged = (per_tree_n >= 8
                                   and (only_one
                                        or sorted_v[0] - sorted_v[1] >= remaining))

                converged = visit_converged or time_up
                if converged:
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
            fpu_reduction=Def.fpu, use_symmetry=False,
            game_name='Othello',
            score_utility_factor=Def.score_utility_factor,
            score_scale=Def.score_scale,
            vl_batch=Def.vl_batch,
            n_trees=1,
            sym_ensemble=_AUTO_SYM_ENSEMBLE)
        self.az_player2 = None   # second MCTS player for AvA white side
        self.player_color = 1    # 1 = Black (first player)
        self.mode = MODE_HVA
        self.net2 = None         # second network for AvA white side
        self._p2_weights_linked = True
        self._n_trees = _AUTO_N_TREES
        self.move_count = 0
        self._book_on = False    # book active for current game
        self._book_idx = 0       # next move index in BOOK_LINE
        self._book_sym = 0       # symmetry id mapping actual board → book board

        # ── Widgets ─────────────────────────────────────────────────────────
        self.board = BoardWidget(self.env)
        self.status = StatusPanel()
        self.trend = WinRateTrend()
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

        # Win rate trend chart
        main_layout.addWidget(self.trend)

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
        # + trend(80) + spacing(8) + bottom_panels(~300) + bottom_margin(12)
        total_h = (12 + self.board.height() + 8 + 85 + 8
                   + 80 + 8 + 20 + 125 + 150 + 12)
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

        # ── Attention extractor ──────────────────────────────────────────
        self._attn_extractor = AttentionExtractor()
        self._attn_net = None

        self._reload_model()

        # ── Connect signals ─────────────────────────────────────────────────
        self.console.model_type_cb.currentIndexChanged.connect(
            self._on_primary_model_changed)

        def _param_delayed(_=None): return self.param_timer.start(150)
        self.console.n_playout_spin.valueChanged.connect(self._on_sims_changed)
        self.console.time_budget_spin.valueChanged.connect(self._on_sims_changed)
        self.console.c_init_sl.valueChanged.connect(_param_delayed)
        self.console.c_base_sl.valueChanged.connect(_param_delayed)
        self.console.fpu_sl.valueChanged.connect(_param_delayed)
        self.console.alpha_sl.valueChanged.connect(_param_delayed)
        self.console.eps_sl.valueChanged.connect(_param_delayed)
        self.console.cache_sl.valueChanged.connect(_param_delayed)
        self.console.score_factor_sl.valueChanged.connect(_param_delayed)
        self.console.score_scale_sl.valueChanged.connect(_param_delayed)
        self.console.vl_batch_spin.valueChanged.connect(_param_delayed)

        self.console.mode_cb.currentIndexChanged.connect(self._on_mode_changed)
        self.console.player_cb.currentIndexChanged.connect(self._on_player_changed)
        self.console.model_type_cb2.currentIndexChanged.connect(
            self._on_p2_model_changed)
        self.restart_btn.clicked.connect(self._reload_and_restart)
        self.undo_btn.clicked.connect(self._undo)
        self.pass_btn.clicked.connect(self._human_pass)
        self.reset_btn.clicked.connect(self.console.reset_defaults)
        self.pause_btn.clicked.connect(self._toggle_search_pause)
        self.hint_btn.clicked.connect(self._toggle_hint)
        self.log_btn.clicked.connect(self._toggle_log)
        self.console.attn_check.stateChanged.connect(self._on_attn_toggle)
        self.console.attn_head_cb.currentIndexChanged.connect(self._on_attn_head_changed)

        self._start_game()

    def _on_attn_toggle(self, state):
        self.board.attn_visible = bool(state)
        self.board.update()

    def _sync_attn_head_selector(self):
        num_heads = 0
        try:
            num_heads = int(self.net.hidden[-1].attn.num_heads)
        except Exception:
            pass
        self.console.set_attention_head_count(num_heads)
        self._on_attn_head_changed(self.console.attn_head_cb.currentIndex())

    def _on_attn_head_changed(self, idx):
        label = self.console.attn_head_cb.itemText(idx)
        if label == "Gate":
            self.board.attn_head = 'gate'
        elif label.startswith("Head "):
            try:
                self.board.attn_head = int(label.split()[-1])
            except ValueError:
                self.board.attn_head = -1
        else:
            self.board.attn_head = -1
        self.board.update()

    def closeEvent(self, event):
        self._attn_extractor.detach()
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
        entry = self.console.model_type_cb.currentText()  # e.g. '001/current'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = self.env_module.CNN(lr=0, device=device)
        net.eval()
        self.net = net
        path = os.path.join(PARAMS_DIR, ENV_NAME, entry)
        try:
            load_model_weights_only(self.net, path, device)
        except Exception:
            print(f"Failed to load model from {path}")
            traceback.print_exc()
            self.status.set_result("MODEL ERROR", C.RED_HEX)

        # Re-attach attention hooks to new model
        self._attn_extractor.attach(self.net)
        self._attn_net = self.net
        self._sync_attn_head_selector()

        p = self.az_player
        p._c_base = int(_sv(self.console.c_base_sl))
        p._fpu_reduction = _sv(self.console.fpu_sl)
        p._noise_eps = _sv(self.console.eps_sl)
        p.noise_eps_init = _sv(self.console.eps_sl)
        p._use_symmetry = False
        p._cache_size = int(_sv(self.console.cache_sl))
        p._score_utility_factor = _sv(self.console.score_factor_sl)
        p._score_scale = _sv(self.console.score_scale_sl)
        self.ai_root_stats.score_scale = p._score_scale
        self.hint_root_stats.score_scale = p._score_scale
        self.status.util_bar.score_scale = p._score_scale
        p.n_trees = 1
        p._vl_batch = self.console.vl_batch_spin.value()
        p.sym_ensemble = _AUTO_SYM_ENSEMBLE

        p.reload(self.net,
                 c_puct=_sv(self.console.c_init_sl),
                 n_playout=self.console.n_playout_spin.value(),
                 alpha=_sv(self.console.alpha_sl),
                 is_self_play=0)
        p.eval()
        self._n_trees = _AUTO_N_TREES
        self.player_color = 1 if self.console.player_cb.currentIndex() == 0 else -1
        self.mode = self.console.mode_cb.currentIndex()
        self.console._update_game_visibility(self.mode)
        if self.mode == MODE_AVA:
            self._ensure_net2()

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
        # Symmetry ensemble is always handled in Python at the GUI layer.
        m.set_use_symmetry(False)
        scale = _sv(self.console.score_scale_sl)
        m.set_score_utility_params(_sv(self.console.score_factor_sl), scale)
        self.ai_root_stats.score_scale = scale
        self.hint_root_stats.score_scale = scale
        self.status.util_bar.score_scale = scale
        new_cache = int(_sv(self.console.cache_sl))
        old_cache = getattr(m, 'cache_size', 0) or 0
        if new_cache != old_cache:
            from src.Cache import LRUCache
            m.cache = LRUCache(new_cache) if new_cache > 0 else None
            m.cache_size = new_cache
        self.az_player._vl_batch = self.console.vl_batch_spin.value()
        self.worker._vl_batch = self.az_player._vl_batch
        # Sync search params to player2 if it exists
        if self.az_player2 is not None:
            m2 = self.az_player2.mcts
            m2.set_c_init(_sv(self.console.c_init_sl))
            m2.set_c_base(_sv(self.console.c_base_sl))
            m2.set_alpha(_sv(self.console.alpha_sl))
            m2.set_fpu_reduction(_sv(self.console.fpu_sl))
            m2.set_noise_epsilon(_sv(self.console.eps_sl))
            m2.set_use_symmetry(False)
            m2.set_score_utility_params(_sv(self.console.score_factor_sl), scale)
            new_cache2 = int(_sv(self.console.cache_sl))
            old_cache2 = getattr(m2, 'cache_size', 0) or 0
            if new_cache2 != old_cache2:
                from src.Cache import LRUCache
                m2.cache = LRUCache(new_cache2) if new_cache2 > 0 else None
                m2.cache_size = new_cache2
            self.az_player2._vl_batch = self.az_player._vl_batch
        if not self._search_paused:
            self.worker.resume()

    # ═══════════════════════════════════════════════════════════════════════
    # Mode / Player switching (no board reset)
    # ═══════════════════════════════════════════════════════════════════════

    def _is_ai_turn(self):
        """Return True if the current turn should be played by AI."""
        if self.mode == MODE_HVH:
            return False
        if self.mode == MODE_AVA:
            return True
        return self.env.turn != self.player_color   # MODE_HVA

    def _active_player(self):
        """Return the MCTS player for the current turn."""
        if self.mode == MODE_AVA and self.az_player2 is not None and self.env.turn == -1:
            return self.az_player2
        return self.az_player

    def _sync_p2_weights_selector(self, force=False):
        if not force and not self._p2_weights_linked:
            return
        idx = self.console.model_type_cb.currentIndex()
        if self.console.model_type_cb2.currentIndex() == idx:
            self._p2_weights_linked = True
            return
        prev = self.console.model_type_cb2.blockSignals(True)
        try:
            self.console.model_type_cb2.setCurrentIndex(idx)
        finally:
            self.console.model_type_cb2.blockSignals(prev)
        self._p2_weights_linked = True

    def _on_primary_model_changed(self, _=None):
        self._sync_p2_weights_selector()
        self.settings_timer.start(400)

    def _on_p2_model_changed(self, _=None):
        self._p2_weights_linked = (
            self.console.model_type_cb2.currentIndex()
            == self.console.model_type_cb.currentIndex()
        )
        self._reload_net2_delayed()

    def _pv_fn_for_turn(self):
        """Return the correct pv_fn for the current turn, considering AvA."""
        if self.mode == MODE_AVA and self.net2 is not None and self.env.turn == -1:
            return self.net2
        return self.az_player.pv_fn

    def _analysis_net_for_turn(self):
        if self.mode == MODE_AVA and self.net2 is not None and self.env.turn == -1:
            return self.net2
        return self.net

    def _on_mode_changed(self):
        new_mode = self.console.mode_cb.currentIndex()
        if new_mode == self.mode:
            return
        self.worker.pause_and_wait()
        self._stop_scan()
        self.mode = new_mode
        self.console._update_game_visibility(new_mode)

        if new_mode == MODE_AVA:
            self._sync_p2_weights_selector()
            self._ensure_net2()

        # Reset MCTS trees — fresh search from current position
        for i in range(self._n_trees):
            self.az_player.mcts.reset_env(i)

        self._update_turn_label()
        self._update_pass_btn()
        self._update_analysis()

        if self.env.done():
            return

        next_ai = self._is_ai_turn()
        # First move of the game: skip search, pick randomly
        if next_ai and self.move_count == 0:
            self._play_random_opening()
            return
        if next_ai:
            self._start_scan()
        self._resume_search(is_ai_turn=next_ai)

    def _on_player_changed(self):
        """Swap human side without restarting. Only effective in HvA / HvH."""
        new_color = 1 if self.console.player_cb.currentIndex() == 0 else -1
        if new_color == self.player_color:
            return
        self.worker.pause_and_wait()
        self._stop_scan()
        self.player_color = new_color

        self.board.ghost_color = QColor(C.BLACK) if new_color == 1 else QColor(C.WHITE)
        self.status.set_player_color(new_color)
        self.trend.set_player_color(new_color)

        # Reset MCTS trees for fresh search
        for i in range(self._n_trees):
            self.az_player.mcts.reset_env(i)

        self._update_turn_label()
        self._update_pass_btn()
        self._update_analysis()

        if self.env.done():
            return

        next_ai = self._is_ai_turn()
        if next_ai:
            self._start_scan()
        self._resume_search(is_ai_turn=next_ai)

    def _ensure_net2(self):
        """Load (or reload) the second network + MCTS player for AvA White side."""
        entry2 = self.console.model_type_cb2.currentText()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net2 = self.env_module.CNN(lr=0, device=device)
        net2.eval()
        path = os.path.join(PARAMS_DIR, ENV_NAME, entry2)
        try:
            load_model_weights_only(net2, path, device)
        except Exception:
            print(f"Failed to load model2 from {path}")
            traceback.print_exc()
            self.status.set_result("MODEL2 ERROR", C.RED_HEX)
        self.net2 = net2

        # Create a second MCTS player mirroring az_player's config
        p = self.az_player
        self.az_player2 = AlphaZeroPlayer(
            None, c_init=None, c_base=p._c_base, n_playout=None,
            alpha=p._alpha, noise_epsilon=p._noise_eps,
            is_selfplay=0, cache_size=p._cache_size,
            fpu_reduction=p._fpu_reduction, use_symmetry=False,
            game_name=p._game_name, board_converter=p._board_converter,
            score_utility_factor=p._score_utility_factor,
            score_scale=p._score_scale,
            value_decay=p._value_decay,
            vl_batch=p._vl_batch,
            n_trees=1,
            sym_ensemble=_AUTO_SYM_ENSEMBLE)
        p2 = self.az_player2
        p2.sym_ensemble = _AUTO_SYM_ENSEMBLE
        p2.n_trees = 1
        p2.reload(net2,
                  c_puct=p.mcts.mcts.config.c_init,
                  n_playout=p._n_playout,
                  alpha=p._alpha,
                  is_self_play=0)
        p2.eval()

    def _reload_net2_delayed(self):
        """Reload White AI network without restart (AvA only)."""
        if self.mode != MODE_AVA:
            return
        self.worker.pause_and_wait()
        self._stop_scan()
        self._ensure_net2()
        if not self.env.done():
            next_ai = self._is_ai_turn()
            if next_ai:
                self._start_scan()
            self._resume_search(is_ai_turn=next_ai)

    # ═══════════════════════════════════════════════════════════════════════
    # Game Flow
    # ═══════════════════════════════════════════════════════════════════════

    def _start_game(self):
        self.worker.pause_and_wait()
        self._stop_scan()
        self.env.reset()
        for i in range(self._n_trees):
            self.az_player.mcts.reset_env(i)
        if self.az_player2 is not None:
            for i in range(_AUTO_N_TREES):
                self.az_player2.mcts.reset_env(i)
        self.board.last_move = None
        self.board.interactive = True
        self.board.ghost_color = QColor(C.BLACK) if self.player_color == 1 else QColor(C.WHITE)
        self.board.scanning = False
        self.board.scan_y = -1
        self.board.overlay_data = None
        self.board.update_valid()
        self.board.update()

        self.status.set_player_color(self.player_color)
        self.trend.set_player_color(self.player_color)
        self.status.set_result("")
        self.status.set_thinking(-1)
        self.status.clear_mcts()
        self.ai_root_stats.clear_data()
        self.ai_child_table.update()
        self.hint_root_stats.clear_data()
        self.hint_child_table.update()
        self.move_count = 0
        self.move_log.clear_log()
        self.trend.clear()
        self._history.clear()
        self.undo_btn.setEnabled(False)
        self._update_pass_btn()
        self._update_analysis()
        self._update_turn_label()

        self._book_on = self.console.book_check.isChecked()
        self._book_idx = 0
        self._book_sym = 0

        self._search_paused = False
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("PAUSE")

        is_ai_first = self._is_ai_turn()
        # First move: all 4 legal positions are symmetric-equivalent, pick randomly
        if is_ai_first:
            self._play_random_opening()
            return
        self._resume_search(is_ai_turn=False)

    def _update_turn_label(self):
        if self.env.done():
            return
        if self.mode == MODE_HVH:
            if self.env.turn == 1:
                self.status.set_turn("BLACK'S TURN", C.TEXT)
            else:
                self.status.set_turn("WHITE'S TURN", C.DIM)
            self.board.interactive = True
        elif self.mode == MODE_AVA:
            if self.env.turn == 1:
                self.status.set_turn("BLACK AI", C.TEXT)
            else:
                self.status.set_turn("WHITE AI", C.DIM)
            self.board.interactive = False
        else:  # MODE_HVA
            if self.env.turn == self.player_color:
                color = C.TEXT if self.player_color == 1 else C.DIM
                self.status.set_turn("YOUR TURN", color)
                self.board.interactive = True
            else:
                self.status.set_turn("AI ACTIVE", C.ACCENT)
                self.board.interactive = False

    def _update_pass_btn(self):
        """Enable PASS only when it's a human's turn and pass is the only valid move."""
        if self.env.done() or self.mode == MODE_AVA:
            self.pass_btn.setEnabled(False)
            return
        if self.mode == MODE_HVA and self.env.turn != self.player_color:
            self.pass_btn.setEnabled(False)
            return
        valid = self.env.valid_move()
        # Pass (action 64) is valid only when no board moves exist
        self.pass_btn.setEnabled(64 in valid and len(valid) == 1)

    def _update_analysis(self):
        # Disc count (use absolute state, not relative _board())
        state = self.env.current_state()
        black_count = int(np.sum(state[0, 0] > 0))
        white_count = int(np.sum(state[0, 1] > 0))
        self.status.set_disc_count(black_count, white_count)

        with torch.no_grad():
            net = self._analysis_net_for_turn()
            if self._attn_net is not net:
                self._attn_extractor.attach(net)
                self._attn_net = net
            base = self.env.current_state()  # (1, 3, 8, 8)
            # Build symmetry-augmented batch (K states) for attention ensemble
            sym_states = []
            for s in _SYM_IDS:
                st = np.stack([apply_sym_board(base[0, c], s, 'Othello')
                               for c in range(3)], axis=0)  # (3, 8, 8)
                sym_states.append(st)
            batch = np.stack(sym_states)  # (K, 3, 8, 8)
            t = torch.from_numpy(batch).float().to(net.device)
            self._attn_extractor.begin_capture()
            try:
                _, vl, sl, *_ = net(t)
            finally:
                self._attn_extractor.end_capture()

            # Use identity (index 0) for NN stats display
            vp = vl[0].exp().cpu().tolist()
            draw_pct = vp[0] * 100
            if self.player_color == self.env.turn:
                win_pct, lose_pct = vp[1] * 100, vp[2] * 100
            else:
                win_pct, lose_pct = vp[2] * 100, vp[1] * 100
            self.status.set_nn_rates(win_pct, draw_pct, lose_pct)

            aux_0 = sl[0].detach().cpu()
            offset = float(getattr(net, 'aux_target_offset', 0))
            if aux_0.ndim == 0 or aux_0.numel() == 1:
                # Current Othello models emit a normalized disc-diff scalar.
                expected = aux_0.item() * offset if offset > 0 else aux_0.item()
            else:
                # Backward compatibility for older aux heads that emitted log-prob distributions.
                sp = aux_0.exp()
                expected = (sp * torch.arange(sp.numel(), dtype=torch.float32)).sum().item() - offset
            if self.player_color != self.env.turn:
                expected = -expected
            self.status.set_nn_diff(expected)

            # Extract attention weights and gate scores — hooks captured K
            # symmetry variants, get_weights/get_gate_scores inverse-permute
            # and average them
            self.board.attn_weights = self._attn_extractor.get_weights()
            self.board.gate_scores = self._attn_extractor.get_gate_scores()

    def _update_status_mcts(self, stats_0):
        d = float(stats_0['root_D'])
        p1w = float(stats_0['root_P1W'])
        p2w = float(stats_0['root_P2W'])
        m = float(stats_0['root_M'])
        if self.player_color == 1:
            win_pct, lose_pct = p1w * 100, p2w * 100
        else:
            win_pct, lose_pct = p2w * 100, p1w * 100
        if self.player_color != self.env.turn:
            m = -m
        self.status.set_mcts_rates(win_pct, d * 100, lose_pct)
        self.status.set_mcts_util(m)

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
    def _nn_direct_eval(self, is_ai_turn):
        """NO SEARCH 模式：NN 前向 + 合法 mask，绕过 MCTS。支持对称增强。"""
        pv_fn = self._pv_fn_for_turn()
        if len(_SYM_IDS) > 1:
            # 对称增强：对所有对称变换取平均
            base = self.env.current_state()  # (1, 3, R, C)
            sym_states = []
            for s in _SYM_IDS:
                st = np.stack([apply_sym_board(base[0, c], s, 'Othello')
                               for c in range(3)], axis=0)  # (3, R, C)
                sym_states.append(st)
            states = np.stack(sym_states)  # (K, 3, R, C)
            all_probs, all_wdl, _ = pv_fn.predict(states)  # (K, A), (K, 3)
            # 逆变换每个对称的 policy 再取平均
            for i, s in enumerate(_SYM_IDS):
                all_probs[i] = inverse_sym_visits(all_probs[i], s, 'Othello')
            probs = all_probs.mean(axis=0)  # (A,)
            wdl = all_wdl.mean(axis=0)      # (3,)
        mask = np.array(self.env.valid_mask(), dtype=bool)
        probs[~mask] = 0.0
        total = probs.sum()
        if total > 0:
            probs /= total

        # 转换 WDL：相对 [draw, win_tomove, loss_tomove] → 绝对 [p1w, p2w]
        d, w_rel, l_rel = float(wdl[0]), float(wdl[1]), float(wdl[2])
        if self.env.turn == 1:
            p1w, p2w = w_rel, l_rel
        else:
            p1w, p2w = l_rel, w_rel
        if self.player_color == 1:
            win_pct, lose_pct = p1w * 100, p2w * 100
        else:
            win_pct, lose_pct = p2w * 100, p1w * 100
        self.status.set_mcts_rates(win_pct, d * 100, lose_pct)

        action = int(np.argmax(probs))

        if is_ai_turn:
            self._save_history()
            self._prune_or_reset(action)
            self._apply_move(action, self.env.turn, is_ai=True)
        else:
            # Hint: 显示 NN 原始策略在棋盘上
            if self._hint_visible:
                n_pct = probs * 100
                self.board.overlay_data = {
                    'N': n_pct,
                    'Q': np.full_like(probs, float(wdl[1] - wdl[2])),
                    'W': np.where(mask, win_pct, 0.0),
                }
                self.board.overlay_best = action
                self.board.update()

    def _resume_search(self, is_ai_turn):
        if self.console.no_search_check.isChecked():
            self._nn_direct_eval(is_ai_turn)
            return
        p = self._active_player()
        K = len(_SYM_IDS)
        board = np.stack([apply_sym_board(self.env.board, s, 'Othello')
                          for s in _SYM_IDS])
        turns = np.full(K, self.env.turn, dtype=np.int32)
        threshold = self.console.n_playout_spin.value()
        tb = self.console.time_budget_spin.value()
        pv_fn = self._pv_fn_for_turn()
        self.worker.set_position(p.mcts, pv_fn, board, turns,
                                 is_ai_turn, threshold, n_trees=K,
                                 sym_ensemble=True,
                                 vl_batch=p._vl_batch, time_budget=tb)
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
                next_ai = self._is_ai_turn()
                if next_ai:
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
        self.worker._time_budget = self.console.time_budget_spin.value()
        if self.worker._is_ai_turn and not self.worker._ai_acted:
            ap = self._active_player()
            raw = ap.mcts.get_root_stats()
            root_n = float(raw['root_N'][0])
            if root_n >= new_thr:
                all_visits = ap.mcts.get_visits_count()
                stats_0 = _aggregate_root_stats_sym_ensemble(raw, _SYM_IDS)
                transformed = all_visits.copy()
                for i, sid in enumerate(_SYM_IDS):
                    transformed[i] = inverse_sym_visits(all_visits[i], sid, 'Othello')
                visits = transformed.sum(axis=0).copy()
                elapsed = time.time() - self.worker._t0
                self._on_ai_ready(stats_0, visits, elapsed)
                return
        if not self._search_paused:
            self.worker.resume()

    # ── Continuous search callbacks ────────────────────────────────────────
    def _on_progress(self, stats_0, visits):
        self._update_status_mcts(stats_0)
        # In AvA, always show as AI panel; in HvH, always show as hint panel
        show_ai = (self.mode == MODE_AVA
                    or (self.mode == MODE_HVA and self.env.turn != self.player_color))
        if show_ai:
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

        self._save_history()
        self._prune_or_reset(action)

        if self.console.reuse_tree_check.isChecked():
            # Pre-compute hint from the NEXT player's pruned subtree.
            # env.turn hasn't changed yet (step not called), so next = opponent.
            if self.mode == MODE_AVA and self.az_player2 is not None:
                # AvA: each player has own tree; pick the opponent's
                hint_player = self.az_player2 if ai_turn == 1 else self.az_player
            else:
                hint_player = self.az_player
            hint_raw = hint_player.mcts.get_root_stats()
            all_hint_v = hint_player.mcts.get_visits_count()
            hint_s0 = _aggregate_root_stats_sym_ensemble(
                hint_raw, _SYM_IDS)
            transformed = all_hint_v.copy()
            for i, sid in enumerate(_SYM_IDS):
                transformed[i] = inverse_sym_visits(all_hint_v[i], sid, 'Othello')
            hint_v = transformed.sum(axis=0).copy()
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

    def _prune_or_reset_player(self, player, action):
        """Prune a single player's tree to action's subtree, or reset."""
        K = _AUTO_N_TREES
        if not self.console.reuse_tree_check.isChecked():
            for i in range(K):
                player.mcts.reset_env(i)
        else:
            sym_actions = np.array(
                [apply_sym_action(action, s, 'Othello') for s in _SYM_IDS],
                dtype=np.int32)
            player.mcts.prune_roots(sym_actions)

    def _prune_or_reset(self, action):
        """Prune tree(s) to action's subtree, or reset if reuse is disabled."""
        self._prune_or_reset_player(self.az_player, action)
        # AvA: also prune the other player's tree
        if self.mode == MODE_AVA and self.az_player2 is not None:
            self._prune_or_reset_player(self.az_player2, action)

    def _record_trend(self):
        """Record current WDL from status bar into trend chart."""
        bar = self.status.wdl_bar
        if bar.w_rate + bar.d_rate + bar.l_rate > 0:
            self.trend.record(self.move_count,
                              bar.w_rate * 100, bar.d_rate * 100, bar.l_rate * 100)

    def _apply_move(self, action, current_player, is_ai):
        """Apply a move, update board, and handle game flow."""
        if action < 64:
            self.board.last_move = action_to_rc(action)
        else:
            self.board.last_move = 'pass'

        # Book tracking: check if this move matches the book line
        if self._book_on:
            if self._book_idx < len(BOOK_LINE):
                if self._book_idx == 0:
                    # First move: detect which symmetry maps it to BOOK_LINE[0]
                    matched = False
                    for s in _SYM_IDS:
                        if apply_sym_action(action, s, 'Othello') == BOOK_LINE[0]:
                            self._book_sym = s
                            self._book_idx = 1
                            matched = True
                            break
                    if not matched:
                        self._book_on = False
                elif apply_sym_action(action, self._book_sym, 'Othello') == BOOK_LINE[self._book_idx]:
                    self._book_idx += 1
                else:
                    self._book_on = False  # deviation → switch to AlphaZero
            else:
                self._book_on = False

        self.env.step(action)
        self.move_count += 1
        self._record_trend()
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

        # Book: if still on-book and it's AI's turn, play instantly
        next_ai = self._is_ai_turn()
        if self._book_on and next_ai:
            self._play_book_move()
            return

        # Normal AlphaZero search
        if next_ai:
            self._start_scan()
        self._resume_search(is_ai_turn=next_ai)

    def _play_random_opening(self):
        """First move of the game: randomly pick one of the 4 equivalent openings."""
        import random
        # All 4 legal first moves are symmetric equivalents of F5
        sym = random.choice(_SYM_IDS)
        action = apply_sym_action(BOOK_LINE[0], sym, 'Othello')
        ai_turn = self.env.turn
        label = action_label(action).upper()
        self.status.set_thinking(0)
        # Book: pre-set sym so _apply_move's first-move detection confirms it
        if self._book_on:
            self._book_sym = sym
            self.status.set_result(f"BOOK: {label}")
        self._save_history()
        self._prune_or_reset(action)
        self._apply_move(action, ai_turn, is_ai=True)

    def _play_book_move(self):
        """Play the next book move instantly (no MCTS search)."""
        if self._book_idx >= len(BOOK_LINE):
            self._book_on = False
            # Fallback to normal search
            next_ai = self._is_ai_turn()
            if next_ai:
                self._start_scan()
            self._resume_search(is_ai_turn=next_ai)
            return
        # Map canonical book action back to actual board coordinates
        # Each Othello symmetry is its own inverse (involution), so apply same sym
        action = apply_sym_action(BOOK_LINE[self._book_idx], self._book_sym, 'Othello')
        ai_turn = self.env.turn
        label = action_label(action).upper()
        self.status.set_thinking(0)
        self.status.set_result(f"BOOK: {label}")
        self._save_history()
        self._prune_or_reset(action)
        self._apply_move(action, ai_turn, is_ai=True)

    def _show_result(self):
        winner = self.env.winPlayer()
        self.board.interactive = False
        self._update_pass_btn()
        self.board.update()

        # 终局：用确定结果替换最后一个 MCTS 估计值
        end_move = self.move_count
        if self.trend._data and self.trend._data[-1][0] == end_move:
            self.trend._data.pop()
        if self.mode == MODE_AVA:
            # AI vs AI — neutral result text
            if winner == 1:
                self.trend.record(end_move, 100, 0, 0)
                self.status.set_turn("BLACK WINS", C.TEXT)
                self.status.set_result("BLACK WINS", C.TEXT)
            elif winner == -1:
                self.trend.record(end_move, 0, 0, 100)
                self.status.set_turn("WHITE WINS", C.DIM)
                self.status.set_result("WHITE WINS", C.DIM)
            else:
                self.trend.record(end_move, 0, 100, 0)
                self.status.set_turn("STALEMATE", C.YEL_HEX)
                self.status.set_result("DRAW", C.YEL_HEX)
        else:
            # HvA / HvH — from player's perspective
            if winner == self.player_color:
                self.trend.record(end_move, 100, 0, 0)
                self.status.set_turn("VICTORY", C.GREEN)
                self.status.set_result("YOU WIN", C.GREEN)
            elif winner == -self.player_color:
                self.trend.record(end_move, 0, 0, 100)
                self.status.set_turn("DEFEATED", C.RED_HEX)
                self.status.set_result("YOU LOSE", C.RED_HEX)
            else:
                self.trend.record(end_move, 0, 100, 0)
                self.status.set_turn("STALEMATE", C.YEL_HEX)
                self.status.set_result("DRAW", C.YEL_HEX)

    # ═══════════════════════════════════════════════════════════════════════
    # Human Input
    # ═══════════════════════════════════════════════════════════════════════

    def _human_pass(self):
        """Handle human pressing the PASS button."""
        if self.env.done() or self.mode == MODE_AVA:
            return
        if self.mode == MODE_HVA and self.env.turn != self.player_color:
            return
        valid = self.env.valid_move()
        if 64 not in valid:
            return

        self.worker.pause_and_wait()
        self.board.overlay_data = None
        self._save_history()
        self._prune_or_reset(64)
        self._apply_move(64, self.env.turn, is_ai=False)

    def mousePressEvent(self, event):
        if self.env.done() or self.mode == MODE_AVA:
            return
        if self.mode == MODE_HVA and self.env.turn != self.player_color:
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
        self._prune_or_reset(action)
        self._apply_move(action, self.env.turn, is_ai=False)

    def _save_history(self):
        self._history.append((self.env.copy(), self.board.last_move, self.move_count,
                              self.ai_root_stats.snapshot(),
                              self.hint_root_stats.snapshot(),
                              self.move_log.snapshot(),
                              self._book_on, self._book_idx, self._book_sym))
        self.undo_btn.setEnabled(True)

    # ═══════════════════════════════════════════════════════════════════════
    # Undo
    # ═══════════════════════════════════════════════════════════════════════

    def _undo(self):
        if not self._history:
            return
        # In HvA, undo only when it's human's turn (or game is done)
        if self.mode == MODE_HVA and self.env.turn != self.player_color and not self.env.done():
            return
        self.worker.pause_and_wait()
        self._stop_scan()
        *saved, saved_book_on, saved_book_idx, saved_book_sym = self._history.pop()
        saved_env, saved_last, saved_count, saved_ai, saved_hint, saved_log = saved
        self._book_on = saved_book_on
        self._book_idx = saved_book_idx
        self._book_sym = saved_book_sym
        self.env = saved_env
        self.board.env = saved_env
        for i in range(self._n_trees):
            self.az_player.mcts.reset_env(i)
        if self.az_player2 is not None:
            for i in range(_AUTO_N_TREES):
                self.az_player2.mcts.reset_env(i)
        self.board.last_move = saved_last
        self.board.overlay_data = None
        self.board.update_valid()
        self.board.update()
        self.move_count = saved_count
        self.trend.truncate(saved_count)
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
            if self.mode == MODE_AVA:
                # AVA: undo 后自动暂停，防止 AI 立刻再下一步
                self._search_paused = True
                self.pause_btn.setChecked(True)
                self.pause_btn.setText("RESUME")
            else:
                next_ai = self._is_ai_turn()
                if next_ai:
                    self._start_scan()
                self._resume_search(is_ai_turn=next_ai)


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
