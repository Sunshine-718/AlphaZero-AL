import os
import threading
import time

import numpy as np
from PyQt5.QtCore import Qt, QRectF, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPainterPath, QPen, QLinearGradient
from PyQt5.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QScrollArea, QSlider, QSpinBox, QTextEdit

from src.symmetry import inverse_sym_stats, inverse_sym_visits


STYLESHEET = """
/* Base */
QWidget {
    background: transparent;
    color: #e0e0f5;
    font-family: "Consolas", "Cascadia Code", monospace;
    font-size: 13px;
}
/* Tabs */
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
/* Combo / Spin */
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
/* Slider */
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
/* Buttons */
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
/* Checkbox */
QCheckBox { spacing: 6px; background: transparent; color: #9090b0; font-family: "Consolas"; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid rgba(255, 255, 255, 20); border-radius: 3px;
    background: rgba(255, 255, 255, 6);
}
QCheckBox::indicator:checked { background: #a78bfa; border-color: #a78bfa; }
QCheckBox:hover { color: #c4b5fd; }
/* Scrollbar */
QScrollBar:vertical { width: 5px; background: transparent; }
QScrollBar::handle:vertical { background: rgba(255, 255, 255, 20); border-radius: 2px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: rgba(167, 139, 250, 60); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
/* TextEdit */
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
/* Labels */
QLabel { background: transparent; }
/* Separators */
QFrame#sep { background: rgba(255, 255, 255, 12); max-height: 1px; }
"""


def _scan_experiments(env_name, params_dir="./params"):
    env_dir = os.path.join(params_dir, env_name)
    entries = []
    if not os.path.isdir(env_dir):
        return entries

    exp_ids = []
    for name in os.listdir(env_dir):
        if os.path.isdir(os.path.join(env_dir, name)) and name.isdigit():
            exp_ids.append(name)
    exp_ids.sort(key=int, reverse=True)

    for exp_id in exp_ids:
        for variant in ("current", "best"):
            model_file = os.path.join(env_dir, exp_id, variant, "model.pt")
            if os.path.exists(model_file):
                entries.append(f"{exp_id}/{variant}")
    return entries


def _sv(slider):
    return slider.value() / slider._scale


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()


def _make_slider(layout, label, lo, hi, step, default, decimals=2, tooltip=""):
    row = QHBoxLayout()
    name = QLabel(label)
    name.setFixedWidth(76)
    name.setStyleSheet("color: #9090b0; font-size: 11px; font-family: Consolas;")
    if tooltip:
        name.setToolTip(tooltip)
    row.addWidget(name)

    scale = 10 ** decimals
    slider = NoWheelSlider(Qt.Horizontal)
    slider._scale = scale
    slider._decimals = decimals
    slider.setRange(int(lo * scale), int(hi * scale))
    slider.setSingleStep(max(1, int(step * scale)))
    slider.setValue(int(default * scale))
    row.addWidget(slider, stretch=1)

    fmt = f"{{:.{decimals}f}}" if decimals else "{:.0f}"
    value_label = QLabel(fmt.format(default))
    value_label.setFixedWidth(55)
    value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    value_label.setStyleSheet(
        "color: #4ade80; font-family: Consolas; font-size: 11px;")
    slider.valueChanged.connect(lambda value: value_label.setText(fmt.format(value / scale)))
    row.addWidget(value_label)
    layout.addLayout(row)
    return slider


def _sep():
    frame = QFrame()
    frame.setObjectName("sep")
    frame.setFrameShape(QFrame.HLine)
    frame.setFixedHeight(1)
    return frame


def _draw_glass(qp, rect, radius=10, fill_alpha=10, border_alpha=25):
    rect_f = QRectF(rect)
    path = QPainterPath()
    path.addRoundedRect(rect_f, radius, radius)

    qp.fillPath(path, QColor(255, 255, 255, fill_alpha))

    qp.save()
    qp.setClipPath(path)
    highlight_h = min(50, rect_f.height() * 0.3)
    highlight = QLinearGradient(rect_f.x(), rect_f.y(), rect_f.x(), rect_f.y() + highlight_h)
    highlight.setColorAt(0, QColor(255, 255, 255, 18))
    highlight.setColorAt(1, QColor(255, 255, 255, 0))
    qp.fillRect(QRectF(rect_f.x(), rect_f.y(), rect_f.width(), highlight_h), highlight)
    qp.restore()

    qp.setPen(QPen(QColor(255, 255, 255, border_alpha), 1))
    qp.setBrush(Qt.NoBrush)
    qp.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)


def _draw_soft_glow(qp, x1, y1, x2, y2, color, core_w=1):
    for width, alpha in ((core_w + 4, 10), (core_w + 2, 30), (core_w, 100)):
        glow = QColor(color)
        glow.setAlpha(alpha)
        qp.setPen(QPen(glow, width))
        qp.drawLine(int(x1), int(y1), int(x2), int(y2))


class BaseMoveLog(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self._lines = []

    def format_move(self, num, player, move):
        raise NotImplementedError

    def add_move(self, num, player, move):
        self._lines.append(self.format_move(num, player, move))
        self.setHtml("<br>".join(self._lines))
        scroll_bar = self.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def clear_log(self):
        self._lines.clear()
        self.clear()

    def snapshot(self):
        return list(self._lines)

    def restore(self, lines):
        self._lines = list(lines)
        self.setHtml("<br>".join(self._lines))
        scroll_bar = self.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())


def aggregate_root_stats_sym_ensemble(raw, sym_ids, game_name):
    merged = inverse_sym_stats(raw, sym_ids, game_name)
    stats = {
        "root_N": float(merged["root_N"][0]),
        "per_tree_N": float(raw["root_N"][0]),
    }
    for key in ("root_Q", "root_M", "root_D", "root_P1W", "root_P2W"):
        stats[key] = float(merged[key][0])
    for key in ("N", "Q", "prior", "noise", "M", "D", "P1W", "P2W"):
        stats[key] = merged[key][0].copy()
    return stats


class SymmetrySearchWorker(QThread):
    progress = pyqtSignal(dict, object)
    ai_ready = pyqtSignal(dict, object, float)

    def __init__(self, game_name, sym_ids, *, chunk=50, parent=None):
        super().__init__(parent)
        self._game_name = game_name
        self._sym_ids = tuple(sym_ids)
        self.CHUNK = chunk
        self._bmcts = None
        self._pv_fn = None
        self._board = None
        self._turns = None
        self._is_ai_turn = False
        self._threshold = 500
        self._n_trees = len(self._sym_ids)
        self._vl_batch = 1
        self._time_budget = 0.0
        self._t0 = 0.0
        self._ai_acted = False
        self._paused = True
        self._stop_flag = False
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()

    def set_position(self, bmcts, pv_fn, board, turns, is_ai_turn, threshold,
                     n_trees=None, vl_batch=1, time_budget=0.0):
        self._bmcts = bmcts
        self._pv_fn = pv_fn
        self._board = np.ascontiguousarray(board, dtype=np.int8)
        self._turns = np.ascontiguousarray(turns, dtype=np.int32)
        self._is_ai_turn = is_ai_turn
        self._threshold = threshold
        self._n_trees = len(self._sym_ids) if n_trees is None else n_trees
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

    def _aggregate_visits(self, all_visits):
        transformed = all_visits.copy()
        for i, sid in enumerate(self._sym_ids):
            transformed[i] = inverse_sym_visits(all_visits[i], sid, self._game_name)
        return transformed.sum(axis=0).copy()

    def run(self):
        while not self._stop_flag:
            if self._paused:
                self._idle.set()
                self._wake.wait()
                self._wake.clear()
                continue
            if self._stop_flag:
                break

            bmcts = self._bmcts
            pv_fn = self._pv_fn
            if bmcts is None or pv_fn is None:
                self._paused = True
                continue

            bmcts.batch_playout(
                pv_fn,
                self._board,
                self._turns,
                n_playout=self.CHUNK,
                vl_batch=self._vl_batch,
            )

            if self._paused or self._stop_flag:
                continue

            raw = bmcts.get_root_stats()
            all_visits = bmcts.get_visits_count()
            stats_0 = aggregate_root_stats_sym_ensemble(raw, self._sym_ids, self._game_name)
            visits = self._aggregate_visits(all_visits)

            self.progress.emit(stats_0, visits)

            if self._is_ai_turn and not self._ai_acted:
                per_tree_n = float(raw["root_N"][0])
                elapsed = time.time() - self._t0

                time_up = (
                    self._time_budget > 0
                    and elapsed >= self._time_budget
                    and per_tree_n >= 8
                )

                sorted_v = np.sort(visits)[::-1]
                only_one = len(sorted_v) < 2 or sorted_v[1] == 0
                remaining = max(0, self._threshold - per_tree_n) * self._n_trees
                visit_converged = (
                    per_tree_n >= 8
                    and (only_one or sorted_v[0] - sorted_v[1] >= remaining)
                )

                if visit_converged or time_up:
                    self._ai_acted = True
                    self._paused = True
                    self.ai_ready.emit(stats_0, visits, time.time() - self._t0)

        self._idle.set()
