"""
查看 dataset.pt 中关键棋盘的 prob / winner 分布，以及 NN 对关键棋盘的输出。
统计全量数据并生成可视化图片。

用法:
    python tools/inspect_buffer.py
    python tools/inspect_buffer.py --game Othello
    python tools/inspect_buffer.py --buffer dataset/dataset.pt
    python tools/inspect_buffer.py --model params/AZ_Connect4_CNN_best.pt
    python tools/inspect_buffer.py --font "Microsoft YaHei"
    python tools/inspect_buffer.py --font-path /path/to/NotoSansCJK-Regular.ttc
    python tools/inspect_buffer.py --no-buffer   # 只看 NN
    python tools/inspect_buffer.py --no-nn       # 只看 buffer
    python tools/inspect_buffer.py --output figs # 指定图片输出目录
"""
import sys, os, argparse, warnings, importlib
from datetime import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import track

console = Console(record=True)

_ZH_FONTS = [
    'Microsoft YaHei',
    'SimHei',
    'PingFang SC',
    'Heiti SC',
    'WenQuanYi Micro Hei',
    'Noto Sans CJK SC',
    'Source Han Sans SC',
    'Songti SC',
    'Kaiti SC',
]
_ZH_FONT_HINTS = ['CJK', 'YaHei', 'SimHei', 'PingFang', 'Heiti', 'WenQuan', 'Song', 'Kai', 'Han Sans']

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

WINNER_LABELS = {1: 'P1 wins', -1: 'P2 wins', 0: 'Draw'}
WINNER_COLORS = {1: '#4C72B0', -1: '#DD8452', 0: '#55A868'}
WINNER_RICH_STYLES = {
    1:  'bold blue',
    -1: 'bold dark_orange',
    0:  'bold green',
}


# ────────────────────────── Game Configs ──────────────────────────

def _c4_label(i):
    return f'col{i}'


def _oth_label(i):
    if i == 64:
        return 'pass'
    return f'{chr(ord("a") + i % 8)}{i // 8 + 1}'


def _make_c4_empty():
    return np.zeros((6, 7), dtype=np.float32)


def _make_c4_first_move():
    b = np.zeros((6, 7), dtype=np.float32)
    b[5, 3] = 1
    return b


def _make_oth_initial():
    b = np.zeros((8, 8), dtype=np.float32)
    b[3, 3] = -1; b[3, 4] = 1
    b[4, 3] = 1;  b[4, 4] = -1
    return b


def _make_oth_first_move():
    b = _make_oth_initial()
    b[2, 3] = 1; b[3, 3] = 1  # flip
    return b


# 轨道编号 → 位置语义标签
_OTHELLO_ORBIT_LABELS = {
    0: 'corner',
    1: 'C-square',
    2: 'edge-2',
    3: 'edge-3',
    4: 'X-square',
    5: 'inner-1',
    6: 'inner-2',
    7: 'XX-square',
    8: 'near-center',
    9: 'center',
}

_COL_NAMES = ['edge', 'near', 'inner', 'center']
_C4_ORBIT_LABELS = {
    r * 4 + c: f'r{r}-{_COL_NAMES[c]}' for r in range(6) for c in range(4)
}

GAME_CONFIGS = {
    'Connect4': dict(
        action_size=7,
        board_shape=(6, 7),
        action_label=_c4_label,
        patterns=[
            ('空棋盘',           _make_c4_empty,      1,  'empty_board_X'),
            ('X 下中间列后 (O)', _make_c4_first_move, -1, 'after_center_O'),
        ],
        step_bins=[(0, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 35), (36, 40), (41, 42)],
        default_model='params/AZ_Connect4_CNN_current.pt',
        best_model='params/AZ_Connect4_CNN_best.pt',
        network_module='src.environments.Connect4.Network',
        utils_module='src.environments.Connect4.utils',
        env_module='src.environments.Connect4',
        plot_top_k=7,
        orbit_labels=_C4_ORBIT_LABELS,
    ),
    'Othello': dict(
        action_size=65,
        board_shape=(8, 8),
        action_label=_oth_label,
        patterns=[
            ('初始局面 (Black)',    _make_oth_initial,    1,  'initial_board_B'),
            ('Black d3 后 (White)', _make_oth_first_move, -1, 'after_d3_W'),
        ],
        step_bins=[(0, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60)],
        default_model='params/AZ_Othello_CNN_current.pt',
        best_model='params/AZ_Othello_CNN_best.pt',
        network_module='src.environments.Othello.Network',
        utils_module='src.environments.Othello.utils',
        env_module='src.environments.Othello',
        plot_top_k=10,
        orbit_labels=_OTHELLO_ORBIT_LABELS,
    ),
}


def setup_matplotlib_font(font_name=None, font_path=None):
    """配置 matplotlib 中文字体。"""
    chosen = None

    if font_path:
        if not os.path.exists(font_path):
            console.print(f'[bold red][!] 指定字体文件不存在: {font_path}[/bold red]')
        else:
            try:
                fm.fontManager.addfont(font_path)
                chosen = fm.FontProperties(fname=font_path).get_name()
            except Exception as e:
                console.print(f'[bold red][!] 加载字体文件失败: {font_path} ({e})[/bold red]')

    available = {f.name for f in fm.fontManager.ttflist}
    if chosen is None and font_name:
        if font_name in available:
            chosen = font_name
        else:
            console.print(f'[bold red][!] 指定字体名未找到: {font_name}[/bold red]')

    if chosen is None:
        for f in _ZH_FONTS:
            if f in available:
                chosen = f
                break

    if chosen is None:
        lower_map = {name.lower(): name for name in available}
        for name_l, name in lower_map.items():
            if any(h.lower() in name_l for h in _ZH_FONT_HINTS):
                chosen = name
                break

    plt.rcParams['axes.unicode_minus'] = False
    if chosen is not None:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
        console.print(f'[dim][font] 使用中文字体: {chosen}[/dim]')
        return

    # 环境里没有中文字体时，避免保存图片时大量重复 glyph 警告刷屏
    warnings.filterwarnings(
        'ignore',
        message=r'Glyph .* missing from font\(s\) DejaVu Sans',
        category=UserWarning,
    )
    console.print('[yellow][font] 未检测到可用中文字体，中文标题可能显示为方框。可用 --font-path 指定 .ttf/.otf。[/yellow]')


def board_matches(state_int8, board, turn):
    ch0 = state_int8[0]
    ch1 = state_int8[1]
    t = 1 if state_int8[2, 0, 0] >= 0 else -1
    if t != turn:
        return False
    p1_match = (ch0 == (board == turn).astype(np.int8)).all()
    p2_match = (ch1 == (board == -turn).astype(np.int8)).all()
    return bool(p1_match and p2_match)


# ────────────────────────── Rich helpers ──────────────────────────

def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute Shannon entropy for each row: -sum(p * log(p))."""
    p = probs.clamp(min=1e-8)
    return -torch.sum(p * p.log(), dim=-1)


def fmt_prob_rich(prob, action_label, top_k=None) -> Text:
    """Format a policy vector as rich Text, highlighting the max action.

    For small action spaces (top_k is None) all actions are shown.
    For large action spaces, only the top_k actions are displayed.
    """
    n = len(prob)
    if top_k is None or top_k >= n:
        indices = list(range(n))
    else:
        indices = list(np.argsort(prob)[::-1][:top_k])
    best = int(prob.argmax())
    parts = Text()
    for j, i in enumerate(indices):
        segment = f'{action_label(i)}:{prob[i]*100:5.1f}%'
        if i == best:
            parts.append(segment, style='bold cyan')
        else:
            parts.append(segment)
        if j < len(indices) - 1:
            parts.append('  ')
    return parts


def make_winner_table(ew: torch.Tensor, title: str = 'Winner Distribution') -> Table:
    total = len(ew)
    table = Table(title=title, show_header=True, header_style='bold')
    table.add_column('Winner', style='bold')
    table.add_column('Count', justify='right')
    table.add_column('%', justify='right')
    for label, val in [('P1 wins', 1), ('P2 wins', -1), ('Draw', 0)]:
        cnt = int((ew == val).sum())
        pct = f'{cnt / total * 100:5.1f}' if total > 0 else '  0.0'
        style = WINNER_RICH_STYLES.get(val, '')
        table.add_row(Text(label, style=style), str(cnt), pct)
    return table


def make_top_n_table(idx, ep, ew, es, top_n, action_label, top_k=None) -> Table:
    show = min(top_n, len(idx))
    table = Table(title=f'前 {show} 条样本', show_header=True, header_style='bold')
    table.add_column('idx', justify='right')
    table.add_column('winner', justify='right')
    table.add_column('steps', justify='right')
    table.add_column('policy')
    for k in range(show):
        i = idx[k]
        w_val = int(ew[k])
        w_text = Text(WINNER_LABELS[w_val], style=WINNER_RICH_STYLES.get(w_val, ''))
        s = str(int(es[k])) if es is not None else '?'
        policy_text = fmt_prob_rich(ep[k].numpy(), action_label, top_k=top_k)
        table.add_row(str(i), w_text, s, policy_text)
    return table


# ────────────────────────── 绘图 ──────────────────────────

def plot_pattern(desc, ep, ew, es, nn_prob, output_dir, fname, gcfg):
    """为一个 pattern 生成 2x2 子图并保存。"""
    action_size = gcfg['action_size']
    action_label = gcfg['action_label']
    top_k = gcfg['plot_top_k']
    show_all = top_k >= action_size

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{desc}  (n={len(ep)})', fontsize=15, fontweight='bold')

    # ── (a) Winner 饼图 ──
    ax = axes[0, 0]
    counts, labels, colors = [], [], []
    for val, label in [(1, 'P1 wins'), (0, 'Draw'), (-1, 'P2 wins')]:
        c = int((ew == val).sum())
        if c > 0:
            counts.append(c)
            labels.append(f'{label}\n{c} ({c/len(ew)*100:.1f}%)')
            colors.append(WINNER_COLORS[val])
    ax.pie(counts, labels=labels, colors=colors, startangle=90,
           textprops={'fontsize': 10})
    ax.set_title('Winner Distribution')

    # ── (b) Policy 柱状图 (mean ± std) + NN overlay ──
    ax = axes[0, 1]
    mean_p = ep.mean(dim=0).numpy()
    std_p = ep.std(dim=0).numpy()

    if show_all:
        plot_idx = np.arange(action_size)
    else:
        plot_idx = np.argsort(mean_p)[::-1][:top_k]

    x = np.arange(len(plot_idx))
    col_labels = [action_label(i) for i in plot_idx]
    ax.bar(x, mean_p[plot_idx], yerr=std_p[plot_idx], capsize=4,
           color='#4C72B0', alpha=0.8, label='Buffer MCTS mean')
    if nn_prob is not None:
        ax.scatter(x, nn_prob[plot_idx], color='#C44E52', s=80, zorder=5,
                   marker='D', label='NN raw policy')
    ax.set_xticks(x)
    ax.set_xticklabels(col_labels, rotation=0 if show_all else 45,
                       ha='center' if show_all else 'right')
    for j, i in enumerate(plot_idx):
        ax.text(j, mean_p[i] + std_p[i] + 0.01, f'{mean_p[i]*100:.1f}%',
                ha='center', fontsize=8)
    ax.set_ylabel('Probability')
    ax.set_title('Policy Distribution' if show_all
                 else f'Policy Distribution (top {top_k})')
    ax.legend(fontsize=9)

    # ── (c) Steps-to-end 直方图 (按 winner 堆叠) ──
    ax = axes[1, 0]
    if es is not None:
        s_np = es.numpy()
        bins = np.arange(s_np.min() - 0.5, s_np.max() + 1.5, 1)
        stack_data, stack_labels, stack_colors = [], [], []
        for val, label in [(1, 'P1 wins'), (0, 'Draw'), (-1, 'P2 wins')]:
            mask = (ew == val).numpy()
            if mask.any():
                stack_data.append(s_np[mask])
                stack_labels.append(label)
                stack_colors.append(WINNER_COLORS[val])
        ax.hist(stack_data, bins=bins, stacked=True, color=stack_colors,
                label=stack_labels, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Steps to End')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No steps data', ha='center', va='center',
                transform=ax.transAxes)
    ax.set_title('Steps-to-End Distribution')

    # ── (d) Policy 箱线图 ──
    ax = axes[1, 1]
    bp_data = [ep[:, i].numpy() for i in plot_idx]
    boxplot_kwargs = dict(
        patch_artist=True,
        showfliers=True,
        flierprops=dict(markersize=2, alpha=0.3),
    )
    try:
        bp = ax.boxplot(bp_data, tick_labels=col_labels, **boxplot_kwargs)
    except TypeError:
        bp = ax.boxplot(bp_data, labels=col_labels, **boxplot_kwargs)
    for patch in bp['boxes']:
        patch.set_facecolor('#A1C9F4')
    if nn_prob is not None:
        ax.scatter(range(1, len(plot_idx) + 1), nn_prob[plot_idx],
                   color='#C44E52', s=60, zorder=5, marker='D',
                   label='NN raw policy')
        ax.legend(fontsize=9)
    if not show_all:
        ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_ylabel('Probability')
    ax.set_title('Policy Spread (Box Plot)' if show_all
                 else f'Policy Spread (Top {top_k} Box Plot)')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{fname}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    console.print(f'    [dim]\\[saved] {path}[/dim]')


# ────────────────────────── Buffer ──────────────────────────

def analyze_buffer(path, top_n, output_dir, nn_policies, gcfg):
    if not os.path.exists(path):
        console.print(f'[bold red][!] Buffer 不存在: {path}[/bold red]')
        return

    action_size = gcfg['action_size']
    action_label = gcfg['action_label']
    top_k = gcfg['plot_top_k']
    patterns = gcfg['patterns']
    step_bins = gcfg['step_bins']
    show_all = top_k >= action_size

    data = torch.load(path, map_location='cpu', weights_only=True)
    states  = data['state']
    probs   = data['prob']
    winners = data['winner']
    steps   = data.get('steps_to_end')
    ptr     = data['_ptr']
    capacity = states.shape[0]
    n = min(ptr, capacity)

    # ── Global Diagnostics ──
    file_size = os.path.getsize(path)
    is_full = ptr >= capacity
    has_wrapped = ptr > capacity

    all_tensors = [states, probs, winners]
    if steps is not None:
        all_tensors.append(steps)
    total_bytes = sum(t.nelement() * t.element_size() for t in all_tensors)

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column('Key', style='bold')
    info_table.add_column('Value')
    info_table.add_row('File', path)
    info_table.add_row('File size', f'{file_size / 1024 / 1024:.2f} MB')
    info_table.add_row('Memory footprint', f'{total_bytes / 1024 / 1024:.2f} MB')
    info_table.add_row('Capacity', str(capacity))
    info_table.add_row('ptr', str(ptr))
    info_table.add_row('Valid samples', f'{n}  ({n / capacity * 100:.1f}%)')
    info_table.add_row('Buffer full', '[green]Yes[/green]' if is_full else '[yellow]No[/yellow]')
    info_table.add_row('ptr wrapped', '[green]Yes[/green]' if has_wrapped else '[dim]No[/dim]')
    info_table.add_row('')  # spacer
    tensor_info = [
        ('state',        states),
        ('prob',         probs),
        ('winner',       winners),
    ]
    if steps is not None:
        tensor_info.append(('steps_to_end', steps))
    for name, t in tensor_info:
        info_table.add_row(f'  {name}', f'{list(t.shape)}  dtype={t.dtype}')

    console.print()
    console.print(Panel(info_table, title='[bold]Buffer Metadata[/bold]', border_style='blue'))

    # Global winner distribution
    valid_winners = winners[:n].view(-1)
    console.print(make_winner_table(valid_winners, title='Global Winner Distribution'))

    # Global steps-to-end stats
    if steps is not None:
        valid_steps = steps[:n].view(-1).float()
        steps_table = Table(title='Global Steps-to-End', show_header=True, header_style='bold')
        steps_table.add_column('Stat', style='bold')
        steps_table.add_column('Value', justify='right')
        steps_table.add_row('Mean', f'{valid_steps.mean():.1f}')
        steps_table.add_row('Std', f'{valid_steps.std():.1f}')
        steps_table.add_row('Min', f'{int(valid_steps.min())}')
        steps_table.add_row('Max', f'{int(valid_steps.max())}')
        console.print(steps_table)

    # Global policy entropy
    valid_probs = probs[:n]
    ent = compute_entropy(valid_probs)
    max_probs = valid_probs.max(dim=1).values
    concentration_pct = float((max_probs > 0.8).float().mean() * 100)

    ent_table = Table(title='Global Policy Entropy', show_header=True, header_style='bold')
    ent_table.add_column('Stat', style='bold')
    ent_table.add_column('Value', justify='right')
    ent_table.add_row('Mean', f'{ent.mean():.4f}')
    ent_table.add_row('Std', f'{ent.std():.4f}')
    ent_table.add_row('Min', f'{ent.min():.4f}')
    ent_table.add_row('Max', f'{ent.max():.4f}')
    ent_table.add_row('Concentration (max>80%)', f'{concentration_pct:.1f}%')
    console.print(ent_table)

    # ── Per-pattern analysis ──
    for desc, make_board, turn, fname in patterns:
        board = make_board()
        board_np = board.astype(np.int8)

        idx = []
        for i in track(range(n), description=f'  Scanning "{desc}"...', console=console, transient=True):
            if board_matches(states[i], board_np, turn):
                idx.append(i)

        player = 'X' if turn == 1 else 'O'

        if not idx:
            console.print(Panel(
                '[yellow]未找到匹配样本。[/yellow]',
                title=f'[bold]{desc} (turn={player})[/bold]',
                border_style='yellow',
            ))
            continue

        idx_t = torch.tensor(idx)
        ep = probs[idx_t]
        ew = winners[idx_t].view(-1)
        es = steps[idx_t].view(-1) if steps is not None else None

        match_ratio = len(idx) / n * 100
        pat_ent = compute_entropy(ep)

        # Summary panel
        pat_table = Table(show_header=False, box=None, padding=(0, 2))
        pat_table.add_column('Key', style='bold')
        pat_table.add_column('Value')
        pat_table.add_row('样本数', str(len(idx)))
        pat_table.add_row('Match ratio', f'{match_ratio:.2f}% of buffer')
        if es is not None:
            pat_table.add_row('Steps-to-end',
                f'mean={es.float().mean():.1f}, std={es.float().std():.1f}, min={es.min()}, max={es.max()}')
        pat_table.add_row('Policy entropy',
            f'mean={pat_ent.mean():.4f}, std={pat_ent.std():.4f}')

        console.print(Panel(pat_table, title=f'[bold]{desc} (turn={player})[/bold]', border_style='cyan'))

        # Winner distribution
        console.print(make_winner_table(ew))

        # Steps-to-end distribution (binned, by winner)
        if es is not None:
            es_int = es.int()
            rows = []
            max_total = 0
            for lo, hi in step_bins:
                mask = (es_int >= lo) & (es_int <= hi)
                p1   = int(((ew == 1) & mask).sum())
                draw = int(((ew == 0) & mask).sum())
                p2   = int(((ew == -1) & mask).sum())
                total = p1 + draw + p2
                max_total = max(max_total, total)
                rows.append((lo, hi, p1, draw, p2, total))

            dist_table = Table(title='Steps-to-End Distribution', show_header=True, header_style='bold')
            dist_table.add_column('Range', style='bold', justify='right')
            dist_table.add_column('P1 wins', justify='right', style='blue')
            dist_table.add_column('Draw', justify='right', style='green')
            dist_table.add_column('P2 wins', justify='right', style='dark_orange')
            dist_table.add_column('Total', justify='right')
            dist_table.add_column('', min_width=25)
            for lo, hi, p1, draw, p2, total in rows:
                bar_len = int(total / max_total * 25) if max_total > 0 else 0
                bar = '█' * bar_len
                dist_table.add_row(f'{lo:2d}-{hi:2d}', str(p1), str(draw), str(p2), str(total), bar)
            console.print(dist_table)

        # Policy mean ± std table
        mp = ep.mean(dim=0).numpy()
        sp = ep.std(dim=0).numpy()

        if show_all:
            show_indices = list(range(action_size))
        else:
            show_indices = list(np.argsort(mp)[::-1][:top_k])

        policy_table = Table(
            title='Policy 均值 ± std' if show_all
            else f'Policy 均值 ± std (top {top_k})',
            show_header=True, header_style='bold')
        for i in show_indices:
            policy_table.add_column(action_label(i), justify='center')
        mean_cells = []
        best_col = int(mp.argmax())
        for i in show_indices:
            val = f'{mp[i]*100:.1f}%'
            if i == best_col:
                mean_cells.append(f'[bold cyan]{val}[/bold cyan]')
            else:
                mean_cells.append(val)
        policy_table.add_row(*mean_cells)
        std_cells = [f'[dim]±{sp[i]*100:.1f}%[/dim]' for i in show_indices]
        policy_table.add_row(*std_cells)
        console.print(policy_table)

        # Top-N samples
        display_top_k = top_k if not show_all else None
        console.print(make_top_n_table(idx, ep, ew, es, top_n, action_label, top_k=display_top_k))

        # 绘图
        nn_prob = nn_policies.get(fname) if nn_policies else None
        plot_pattern(desc, ep, ew, es, nn_prob, output_dir, fname, gcfg)
        console.print()


# ────────────────────────── Embedding 诊断 ──────────────────────────

def _cosine_sim(a, b):
    """Cosine similarity between two 1-D tensors."""
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def _fmt_top_dims(vec, k=6):
    top_idx = torch.argsort(vec.abs(), descending=True)[:k].tolist()
    return ', '.join(f'd{idx}={vec[idx]:+.3f}' for idx in top_idx)


def analyze_embeddings(net, output_dir, gcfg):
    """分析并可视化 Piece / Position Embedding。"""
    piece_w = net.piece_emb.weight.detach().cpu()    # (n_pieces, d)
    pos_w = net.pos_emb.weight.detach().cpu()        # (n_orbits, d)
    has_phase = hasattr(net, 'phase_emb')
    if has_phase:
        phase_w = net.phase_emb.weight.detach().cpu()  # (2, d)
    has_player = hasattr(net, 'player_emb')
    if has_player:
        player_w = net.player_emb.weight.detach().cpu()  # (2, d)
    orbit_map = net.orbit_map.cpu().numpy()
    n_orbits = pos_w.shape[0]
    n_pieces = piece_w.shape[0]
    rows, cols = gcfg['board_shape']
    orbit_labels_dict = gcfg.get('orbit_labels', {})

    PIECE_NAMES = ['Empty', 'Own', 'Opp'] if n_pieces == 3 else ['Own', 'Opp']
    PLAYER_NAMES = ['P1 (Black)', 'P2 (White)']

    # ── 1. Piece Embedding 表格 ──
    piece_norms = piece_w.norm(dim=1)
    piece_table = Table(title='Piece Embedding', show_header=True, header_style='bold')
    piece_table.add_column('', style='bold')
    piece_table.add_column('L2 Norm', justify='right')
    for name in PIECE_NAMES:
        piece_table.add_column(f'cos({name})', justify='right')
    for i, name in enumerate(PIECE_NAMES):
        sims = [f'{_cosine_sim(piece_w[i], piece_w[j]):+.4f}' for j in range(n_pieces)]
        piece_table.add_row(name, f'{piece_norms[i]:.4f}', *sims)

    own_idx, opp_idx = (1, 2) if n_pieces == 3 else (0, 1)
    own_opp_dist = float((piece_w[own_idx] - piece_w[opp_idx]).norm())
    piece_table.add_section()
    piece_table.add_row('L2 dist Own-Opp', f'{own_opp_dist:.4f}', *[''] * n_pieces)
    if n_pieces == 3:
        own_empty_dist = float((piece_w[1] - piece_w[0]).norm())
        opp_empty_dist = float((piece_w[2] - piece_w[0]).norm())
        piece_table.add_row('L2 dist Own-Empty', f'{own_empty_dist:.4f}', '', '', '')
        piece_table.add_row('L2 dist Opp-Empty', f'{opp_empty_dist:.4f}', '', '', '')
    console.print(piece_table)

    piece_dims_table = Table(title='Piece Embedding Top Dimensions', show_header=True, header_style='bold')
    piece_dims_table.add_column('Piece', style='bold')
    piece_dims_table.add_column('Top-|value| dims')
    for i, name in enumerate(PIECE_NAMES):
        piece_dims_table.add_row(name, _fmt_top_dims(piece_w[i]))
    console.print(piece_dims_table)

    # ── 2. Phase Embedding 表格（仅在存在时显示） ──
    if has_phase:
        phase_names = ['Opening', 'Endgame']
        phase_norms = phase_w.norm(dim=1)
        phase_cos = _cosine_sim(phase_w[0], phase_w[1])
        phase_dist = float((phase_w[0] - phase_w[1]).norm())

        phase_table = Table(title='Phase Embedding', show_header=True, header_style='bold')
        phase_table.add_column('', style='bold')
        phase_table.add_column('L2 Norm', justify='right')
        phase_table.add_column('Top-|value| dims')
        for i, name in enumerate(phase_names):
            phase_table.add_row(name, f'{phase_norms[i]:.4f}', _fmt_top_dims(phase_w[i]))
        phase_table.add_section()
        phase_table.add_row('cos(Open, End)', f'{phase_cos:+.4f}', '')
        phase_table.add_row('L2 dist(Open, End)', f'{phase_dist:.4f}', '')
        console.print(phase_table)

    # ── 3. Player Embedding 表格（仅在存在时显示） ──
    if has_player:
        player_norms = player_w.norm(dim=1)
        p_cos = _cosine_sim(player_w[0], player_w[1])
        p_dist = float((player_w[0] - player_w[1]).norm())
        player_table = Table(title='Player Embedding', show_header=True, header_style='bold')
        player_table.add_column('', style='bold')
        player_table.add_column('L2 Norm', justify='right')
        for i, name in enumerate(PLAYER_NAMES):
            player_table.add_row(name, f'{player_norms[i]:.4f}')
        player_table.add_section()
        player_table.add_row('cos(P1, P2)', f'{p_cos:+.4f}')
        player_table.add_row('L2 dist(P1, P2)', f'{p_dist:.4f}')
        console.print(player_table)

    # ── 4. Position Embedding 表格（关键轨道） ──
    pos_norms = pos_w.norm(dim=1)
    ref_orbit = pos_w[0]

    pos_table = Table(title='Position Embedding (key orbits)', show_header=True, header_style='bold')
    pos_table.add_column('Orbit', justify='right', style='bold')
    pos_table.add_column('Label')
    pos_table.add_column('L2 Norm', justify='right')
    ref_label = orbit_labels_dict.get(0, 'orbit-0')
    pos_table.add_column(f'cos({ref_label})', justify='right')

    sorted_orbits = torch.argsort(pos_norms, descending=True).tolist()
    for orbit_id in sorted_orbits:
        label = orbit_labels_dict.get(orbit_id, '')
        norm_val = f'{pos_norms[orbit_id]:.4f}'
        cos_val = f'{_cosine_sim(pos_w[orbit_id], ref_orbit):+.4f}'
        pos_table.add_row(str(orbit_id), label, norm_val, cos_val)
    console.print(pos_table)

    # ── 5. 绘图 ──
    os.makedirs(output_dir, exist_ok=True)

    # 5-ref. 轨道参考图
    orbit_board = np.array(orbit_map).reshape(rows, cols)
    fig, ax = plt.subplots(figsize=(max(6, cols * 0.8), max(5, rows * 0.8)))
    cmap = plt.cm.get_cmap('tab10' if n_orbits <= 10 else 'tab20', n_orbits)
    im = ax.imshow(orbit_board, cmap=cmap, vmin=-0.5, vmax=n_orbits - 0.5)
    for r in range(rows):
        for c in range(cols):
            oid = orbit_board[r, c]
            label = orbit_labels_dict.get(oid, '')
            txt = f'{oid}\n{label}' if label else str(oid)
            ax.text(c, r, txt, ha='center', va='center', fontsize=7, fontweight='bold')
    ax.set_xticks(range(cols), [f'c{i}' for i in range(cols)])
    ax.set_yticks(range(rows), [f'r{i}' for i in range(rows)])
    ax.set_title(f'Orbit Map ({n_orbits} orbits)')
    fig.colorbar(im, ax=ax, shrink=0.8, ticks=range(n_orbits))
    plt.tight_layout()
    path = os.path.join(output_dir, 'emb_orbit_map.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    console.print(f'    [dim]\\[saved] {path}[/dim]')

    # 5a. Position L2 Norm 热力图
    board_norms = pos_norms[orbit_map].reshape(rows, cols).numpy()
    fig, ax = plt.subplots(figsize=(max(6, cols * 0.8), max(5, rows * 0.8)))
    im = ax.imshow(board_norms, cmap='YlOrRd')
    for r in range(rows):
        for c in range(cols):
            ax.text(c, r, f'{board_norms[r, c]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_xticks(range(cols), [f'c{i}' for i in range(cols)])
    ax.set_yticks(range(rows), [f'r{i}' for i in range(rows)])
    ax.set_title('Position Embedding L2 Norm')
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, 'emb_position_norm.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    console.print(f'    [dim]\\[saved] {path}[/dim]')

    # 5b. Position cosine similarity to orbit-0 热力图
    cos_to_ref = torch.nn.functional.cosine_similarity(
        pos_w, ref_orbit.unsqueeze(0), dim=1
    )
    board_cos = cos_to_ref[orbit_map].reshape(rows, cols).numpy()
    fig, ax = plt.subplots(figsize=(max(6, cols * 0.8), max(5, rows * 0.8)))
    im = ax.imshow(board_cos, cmap='RdBu_r', vmin=-1, vmax=1)
    for r in range(rows):
        for c in range(cols):
            ax.text(c, r, f'{board_cos[r, c]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_xticks(range(cols), [f'c{i}' for i in range(cols)])
    ax.set_yticks(range(rows), [f'r{i}' for i in range(rows)])
    ax.set_title(f'Position Embedding cos(orbit, {ref_label})')
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, 'emb_position_cosine.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    console.print(f'    [dim]\\[saved] {path}[/dim]')

    # 5c. Position Embedding PCA 散点图
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pos_2d = pca.fit_transform(pos_w.numpy())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pos_2d[:, 0], pos_2d[:, 1], s=100, c=pos_norms.numpy(),
               cmap='YlOrRd', edgecolors='black', linewidths=0.5, zorder=5)
    for i in range(n_orbits):
        label = orbit_labels_dict.get(i, str(i))
        ax.annotate(label,
                    (pos_2d[i, 0], pos_2d[i, 1]),
                    textcoords='offset points', xytext=(6, 6), fontsize=8)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Position Embedding PCA')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'emb_position_pca.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    console.print(f'    [dim]\\[saved] {path}[/dim]')

    # 5d. PCA 前两个主成分的棋盘热力图
    orbit_map_t = torch.tensor(orbit_map, dtype=torch.long)
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, rows * 0.8)))
    for k, ax in enumerate(axes):
        pc_vals = torch.from_numpy(pos_2d[:, k])
        board_pc = pc_vals[orbit_map_t].reshape(rows, cols).numpy()
        vmax = max(abs(board_pc.min()), abs(board_pc.max()))
        im = ax.imshow(board_pc, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        for r in range(rows):
            for c in range(cols):
                ax.text(c, r, f'{board_pc[r, c]:.2f}', ha='center', va='center', fontsize=7)
        ax.set_xticks(range(cols), [f'c{i}' for i in range(cols)])
        ax.set_yticks(range(rows), [f'r{i}' for i in range(rows)])
        ax.set_title(f'PC{k+1} ({pca.explained_variance_ratio_[k]*100:.1f}%)')
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Position Embedding — PCA Heatmaps')
    plt.tight_layout()
    path = os.path.join(output_dir, 'emb_position_pca_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    console.print(f'    [dim]\\[saved] {path}[/dim]')

    # 5e. 轨道余弦相似度矩阵
    pos_norm = pos_w / pos_w.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos_matrix = (pos_norm @ pos_norm.T).numpy()
    orbit_labels = [orbit_labels_dict.get(i, str(i)) for i in range(n_orbits)]
    font_size = max(4, min(7, 120 // n_orbits))
    fig, ax = plt.subplots(figsize=(max(9, n_orbits * 0.5), max(8, n_orbits * 0.45)))
    im = ax.imshow(cos_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    for r in range(n_orbits):
        for c in range(n_orbits):
            ax.text(c, r, f'{cos_matrix[r, c]:.1f}', ha='center', va='center', fontsize=font_size)
    ax.set_xticks(range(n_orbits), orbit_labels, rotation=45, ha='right', fontsize=max(5, font_size))
    ax.set_yticks(range(n_orbits), orbit_labels, fontsize=max(5, font_size))
    ax.set_title('Position Embedding — Orbit Cosine Similarity')
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, 'emb_position_cos_matrix.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    console.print(f'    [dim]\\[saved] {path}[/dim]')

    # 5f. Piece (+ Player) Embedding PCA 散点图
    piece_colors = ['#55A868', '#4C72B0', '#DD8452'][:n_pieces]
    piece_markers = (['o', 's', 's'] if n_pieces == 3 else ['s', 's'])[:n_pieces]
    if has_player:
        all_emb = torch.cat([piece_w, player_w], dim=0)
        all_labels = PIECE_NAMES + PLAYER_NAMES
        all_colors = piece_colors + ['#8172B2', '#C44E52']
        all_markers = piece_markers + ['^', '^']
        title = 'Piece & Player Embedding PCA'
    else:
        all_emb = piece_w
        all_labels = PIECE_NAMES
        all_colors = piece_colors
        all_markers = piece_markers
        title = 'Piece Embedding PCA'

    pca2 = PCA(n_components=2)
    emb_2d = pca2.fit_transform(all_emb.numpy())
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(len(all_labels)):
        ax.scatter(emb_2d[i, 0], emb_2d[i, 1], s=200, c=all_colors[i],
                   marker=all_markers[i], edgecolors='black', linewidths=0.5, zorder=5)
        ax.annotate(all_labels[i], (emb_2d[i, 0], emb_2d[i, 1]),
                    textcoords='offset points', xytext=(8, 8), fontsize=10, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'emb_piece_player.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    console.print(f'    [dim]\\[saved] {path}[/dim]')

    # 5g. Phase Embedding interpolation trajectory
    if has_phase:
        t_vals = torch.linspace(0.0, 1.0, 11).unsqueeze(1)
        phase_traj = torch.lerp(phase_w[0].unsqueeze(0), phase_w[1].unsqueeze(0), t_vals)
        phase_np = phase_traj.numpy()
        if np.allclose(phase_np, phase_np[0]):
            phase_2d = np.column_stack([t_vals.squeeze(1).numpy(), np.zeros(len(t_vals))])
            phase_x_label = 't'
            phase_y_label = 'degenerate'
        else:
            phase_pca = PCA(n_components=2)
            phase_2d = phase_pca.fit_transform(phase_np)
            phase_x_label = f'PC1 ({phase_pca.explained_variance_ratio_[0]*100:.1f}%)'
            phase_y_label = f'PC2 ({phase_pca.explained_variance_ratio_[1]*100:.1f}%)'

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(phase_2d[:, 0], phase_2d[:, 1], color='#4C72B0', linewidth=2, alpha=0.9)
        ax.scatter(phase_2d[:, 0], phase_2d[:, 1], c=t_vals.squeeze(1).numpy(),
                   cmap='viridis', s=70, edgecolors='black', linewidths=0.4, zorder=5)
        ax.annotate('Opening (t=0.0)', (phase_2d[0, 0], phase_2d[0, 1]),
                    textcoords='offset points', xytext=(8, 8), fontsize=9, fontweight='bold')
        ax.annotate('Endgame (t=1.0)', (phase_2d[-1, 0], phase_2d[-1, 1]),
                    textcoords='offset points', xytext=(8, -14), fontsize=9, fontweight='bold')
        for idx, t in enumerate(t_vals.squeeze(1).tolist()):
            if idx in (0, len(phase_2d) - 1):
                continue
            ax.annotate(f'{t:.1f}', (phase_2d[idx, 0], phase_2d[idx, 1]),
                        textcoords='offset points', xytext=(4, 4), fontsize=7)
        ax.set_xlabel(phase_x_label)
        ax.set_ylabel(phase_y_label)
        ax.set_title('Phase Embedding Interpolation')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, 'emb_phase_interp.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        console.print(f'    [dim]\\[saved] {path}[/dim]')

        # 5h. Top phase dimensions over t
        phase_delta = (phase_w[1] - phase_w[0]).abs()
        top_dims = torch.argsort(phase_delta, descending=True)[:6].tolist()
        fig, ax = plt.subplots(figsize=(8, 5))
        for dim in top_dims:
            ax.plot(t_vals.squeeze(1).numpy(), phase_traj[:, dim].numpy(),
                    marker='o', linewidth=1.8, label=f'd{dim}')
        ax.set_xlabel('t')
        ax.set_ylabel('Embedding value')
        ax.set_title('Phase Embedding Top Dimensions vs t')
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        path = os.path.join(output_dir, 'emb_phase_dims.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        console.print(f'    [dim]\\[saved] {path}[/dim]')


# ────────────────────────── Attention Map ──────────────────────────

def _extract_attention_data(net, state_tensor):
    """Extract attention weights and gate scores from all Attention modules.

    Hooks q_norm/k_norm outputs to compute softmax(Q @ K^T / sqrt(d_k)).
    For GatedAttention, also hooks qkvg_proj to extract gate = sigmoid(g).

    Returns:
        attn_weights: list of (num_heads, seq_len, seq_len) tensors
        gate_scores:  list of (seq_len, num_heads) tensors, or empty if no gates
    """
    attn_weights = []
    gate_scores = []

    captured = {}  # id(module) -> {'q': tensor, 'k': tensor, 'gate': tensor}
    hooks = []
    modules_order = []

    for m in net.modules():
        if hasattr(m, 'q_norm') and hasattr(m, 'k_norm'):
            mid = id(m)
            captured[mid] = {}
            modules_order.append(m)

            def _make_hooks(mid, mod):
                def _hq(module, input, output):
                    captured[mid]['q'] = output.detach()
                def _hk(module, input, output):
                    captured[mid]['k'] = output.detach()
                def _hg(module, input, output):
                    # gate_proj output: (B, S, H)
                    captured[mid]['gate'] = torch.sigmoid(output.detach())
                return _hq, _hk, _hg

            hq, hk, hg = _make_hooks(mid, m)
            hooks.append(m.q_norm.register_forward_hook(hq))
            hooks.append(m.k_norm.register_forward_hook(hk))
            if hasattr(m, 'gate_proj'):
                hooks.append(m.gate_proj.register_forward_hook(hg))

    if not hooks:
        return [], []

    with torch.no_grad():
        net(state_tensor)

    for h in hooks:
        h.remove()

    for m in modules_order:
        mid = id(m)
        if mid in captured and 'q' in captured[mid] and 'k' in captured[mid]:
            q = captured[mid]['q'].permute(0, 2, 1, 3)  # (B, H, S, D)
            k = captured[mid]['k'].permute(0, 2, 1, 3)
            head_dim = q.shape[-1]
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            weights = torch.softmax(scores, dim=-1)
            attn_weights.append(weights[0].cpu())
            if 'gate' in captured[mid]:
                gate_scores.append(captured[mid]['gate'][0].cpu())  # (S, H)

    return attn_weights, gate_scores


def analyze_attention(net, output_dir, gcfg, device='cpu'):
    """Visualize attention maps for each pattern board position."""
    utils_mod = importlib.import_module(gcfg['utils_module'])
    board_to_state = utils_mod.board_to_state
    rows, cols = gcfg['board_shape']
    seq_len = rows * cols

    # Key query positions to visualize (board indices)
    if (rows, cols) == (8, 8):
        query_positions = {
            'corner(a1)': 0, 'X-sq(b2)': 9, 'edge(a4)': 3,
            'center(d4)': 27, 'center(e5)': 36,
        }
    else:
        # Generic: corners + center
        query_positions = {
            'top-left': 0, 'center': (rows // 2) * cols + cols // 2,
        }

    for desc, make_board, turn, fname in gcfg['patterns']:
        board = make_board()
        state = board_to_state(board, turn)
        t = torch.from_numpy(state).to(device)

        attn_list, gate_list = _extract_attention_data(net, t)
        if not attn_list:
            return

        for layer_idx, weights in enumerate(attn_list):
            num_heads = weights.shape[0]
            n_queries = len(query_positions)
            fig, axes = plt.subplots(n_queries, num_heads,
                                     figsize=(3.5 * num_heads, 3.5 * n_queries),
                                     squeeze=False)

            for qi, (q_name, q_pos) in enumerate(query_positions.items()):
                for hi in range(num_heads):
                    ax = axes[qi, hi]
                    attn_map = weights[hi, q_pos, :seq_len].reshape(rows, cols).numpy()
                    im = ax.imshow(attn_map, cmap='hot', interpolation='nearest',
                                   vmin=0, vmax=attn_map.max())
                    # Mark query position
                    qr, qc = q_pos // cols, q_pos % cols
                    ax.plot(qc, qr, 'c*', markersize=12, markeredgecolor='white',
                            markeredgewidth=0.5)
                    if qi == 0:
                        ax.set_title(f'Head {hi}', fontsize=11)
                    if hi == 0:
                        ax.set_ylabel(q_name, fontsize=11)
                    ax.set_xticks(range(cols))
                    ax.set_yticks(range(rows))
                    ax.tick_params(labelsize=7)
                    fig.colorbar(im, ax=ax, shrink=0.8)

            player = 'X' if turn == 1 else 'O'
            fig.suptitle(f'Attention Map — {desc} (turn={player})', fontsize=13, fontweight='bold')
            plt.tight_layout()
            path = os.path.join(output_dir, f'attn_{fname}_L{layer_idx}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            console.print(f'    [dim]\\[saved] {path}[/dim]')

        # Gate score visualization (GatedAttention only)
        for layer_idx, gate in enumerate(gate_list):
            # gate: (S, H) — sigmoid gate score per position per head
            num_heads = gate.shape[1]
            gate_np = gate.numpy()  # (64, H)

            fig, axes = plt.subplots(1, num_heads + 1,
                                     figsize=(3.5 * (num_heads + 1), 3.5),
                                     squeeze=False)

            for hi in range(num_heads):
                ax = axes[0, hi]
                g_map = gate_np[:seq_len, hi].reshape(rows, cols)
                im = ax.imshow(g_map, cmap='RdYlGn', interpolation='nearest',
                               vmin=0, vmax=1)
                ax.set_title(f'Head {hi}', fontsize=11)
                ax.set_xticks(range(cols))
                ax.set_yticks(range(rows))
                ax.tick_params(labelsize=7)
                fig.colorbar(im, ax=ax, shrink=0.8)

            # Mean across heads
            ax = axes[0, num_heads]
            g_mean = gate_np[:seq_len].mean(axis=1).reshape(rows, cols)
            im = ax.imshow(g_mean, cmap='RdYlGn', interpolation='nearest',
                           vmin=0, vmax=1)
            ax.set_title('Mean', fontsize=11)
            ax.set_xticks(range(cols))
            ax.set_yticks(range(rows))
            ax.tick_params(labelsize=7)
            fig.colorbar(im, ax=ax, shrink=0.8)

            player = 'X' if turn == 1 else 'O'
            fig.suptitle(f'Gate Score — {desc} (turn={player})', fontsize=13, fontweight='bold')
            plt.tight_layout()
            path = os.path.join(output_dir, f'gate_{fname}_L{layer_idx}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            console.print(f'    [dim]\\[saved] {path}[/dim]')


# ────────────────────────── NN ──────────────────────────

def analyze_nn(model_path, gcfg, device='cpu', output_dir=None):
    """加载 NN 并返回每个 pattern 的 raw policy，同时打印摘要。"""
    console.print()
    console.print(Panel(f'[bold]{model_path}[/bold]', title='[bold]NN Model[/bold]', border_style='magenta'))

    net_mod = importlib.import_module(gcfg['network_module'])
    utils_mod = importlib.import_module(gcfg['utils_module'])
    env_mod = importlib.import_module(gcfg['env_module'])
    CNN = net_mod.CNN
    Env = env_mod.Env
    board_to_state = utils_mod.board_to_state

    action_label = gcfg['action_label']
    action_size = gcfg['action_size']
    top_k = gcfg['plot_top_k']
    show_all = top_k >= action_size

    net = CNN(lr=0, device=device)
    net.load(model_path)
    net.eval()

    nn_policies = {}
    for desc, make_board, turn, fname in gcfg['patterns']:
        board = make_board()
        player = 'X' if turn == 1 else 'O'
        state = board_to_state(board, turn)
        t = torch.from_numpy(state).to(device)

        # Get valid mask from environment
        env = Env(board.astype(np.float32))
        mask = np.array(env.valid_mask(), dtype=bool)

        with torch.no_grad():
            log_p, log_v, log_s, *_ = net(t)
            prob = log_p.exp().cpu().numpy().flatten()
            vdist = log_v.exp().cpu().numpy().flatten()  # [draw, win(to-move), loss(to-move)]
            sdist = log_s.exp().cpu().numpy().flatten()

        # Mask and renormalize policy
        prob[~mask] = 0.0
        total = prob.sum()
        if total > 0:
            prob /= total

        nn_policies[fname] = prob
        scalar_v = vdist[1] - vdist[2]
        entropy = -np.sum(prob * np.log(prob + 1e-8))
        top5s = np.argsort(sdist)[::-1][:5]

        nn_table = Table(show_header=False, box=None, padding=(0, 2))
        nn_table.add_column('Key', style='bold')
        nn_table.add_column('Value')
        display_top_k = top_k if not show_all else None
        nn_table.add_row('Policy', fmt_prob_rich(prob, action_label, top_k=display_top_k))
        nn_table.add_row('Value (3-way)', Text.assemble(
            ('Draw:', 'green'),     (f'{vdist[0]*100:5.1f}%  ', ''),
            ('Win(to-move):', 'blue'), (f'{vdist[1]*100:5.1f}%  ', ''),
            ('Loss(to-move):', 'dark_orange'), (f'{vdist[2]*100:5.1f}%', ''),
        ))
        nn_table.add_row('Scalar value', f'{scalar_v: .4f}')
        nn_table.add_row('Entropy', f'{entropy:.4f}')
        nn_table.add_row('Steps top-5',
            ', '.join(f's={s}:{sdist[s]*100:.1f}%' for s in top5s))

        console.print(Panel(nn_table, title=f'[bold]{desc} (turn={player})[/bold]', border_style='magenta'))

    if hasattr(net, 'piece_emb'):
        analyze_embeddings(net, output_dir, gcfg)

    # Attention map visualization (only for models with Attention modules)
    has_attn = any(hasattr(m, 'q_norm') and hasattr(m, 'k_norm')
                   for m in net.modules())
    if has_attn:
        analyze_attention(net, output_dir, gcfg, device)

    return nn_policies


# ────────────────────────── main ──────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Inspect buffer & NN for key board states')
    parser.add_argument('--game', default='Connect4', choices=list(GAME_CONFIGS.keys()))
    parser.add_argument('--buffer', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--best', action='store_true', help='Use best model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--output', default=None, help='Output directory for plots (default: tools/figures/<timestamp>)')
    parser.add_argument('--font', default=None, help='Matplotlib font family name for Chinese')
    parser.add_argument('--font-path', default=None, help='Path to .ttf/.otf/.ttc font file')
    parser.add_argument('--no-buffer', action='store_true')
    parser.add_argument('--no-nn', action='store_true')
    args = parser.parse_args()

    gcfg = GAME_CONFIGS[args.game]
    if args.buffer is None:
        env_buffer = os.path.join('dataset', f'{args.game}_dataset.pt')
        legacy_buffer = os.path.join('dataset', 'dataset.pt')
        args.buffer = env_buffer if os.path.exists(env_buffer) else legacy_buffer
    if args.model is None:
        # Auto-detect latest experiment directory
        from src.pipeline import _latest_experiment_dir
        env_dir = f'./params/{args.game}'
        exp_dir = _latest_experiment_dir(env_dir)
        if exp_dir:
            variant = 'best' if args.best else 'current'
            args.model = os.path.join(exp_dir, variant)
        else:
            # Fallback to legacy paths
            args.model = gcfg['best_model'] if args.best else gcfg['default_model']
    elif args.best:
        from src.pipeline import _latest_experiment_dir
        env_dir = f'./params/{args.game}'
        exp_dir = _latest_experiment_dir(env_dir)
        args.model = os.path.join(exp_dir, 'best') if exp_dir else gcfg['best_model']

    if args.output is None:
        args.output = os.path.join('tools', 'figures', datetime.now().strftime('%Y%m%d_%H%M%S'))

    os.chdir(ROOT)
    setup_matplotlib_font(args.font, args.font_path)

    # 先跑 NN，拿到 raw policy 供 buffer 绘图叠加
    nn_policies = {}
    if not args.no_nn:
        nn_policies = analyze_nn(args.model, gcfg, args.device, args.output)

    if not args.no_buffer:
        analyze_buffer(args.buffer, args.top, args.output, nn_policies, gcfg)

    # 保存终端输出到文件
    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, 'report.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(console.export_text())
    console.print(f'[dim]\\[saved] {log_path}[/dim]')


if __name__ == '__main__':
    main()
