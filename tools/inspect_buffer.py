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

console = Console()

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
        plot_top_k=7,
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
        plot_top_k=10,
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


# ────────────────────────── NN ──────────────────────────

def analyze_nn(model_path, gcfg, device='cpu'):
    """加载 NN 并返回每个 pattern 的 raw policy，同时打印摘要。"""
    console.print()
    console.print(Panel(f'[bold]{model_path}[/bold]', title='[bold]NN Model[/bold]', border_style='magenta'))

    if not os.path.exists(model_path):
        console.print(f'[bold red]  [!] 模型不存在: {model_path}[/bold red]')
        return {}

    net_mod = importlib.import_module(gcfg['network_module'])
    utils_mod = importlib.import_module(gcfg['utils_module'])
    CNN = net_mod.CNN
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
        with torch.no_grad():
            log_p, log_v, log_s = net(t)
            prob = log_p.exp().cpu().numpy().flatten()
            vdist = log_v.exp().cpu().numpy().flatten()  # [draw, win(to-move), loss(to-move)]
            sdist = log_s.exp().cpu().numpy().flatten()

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

    return nn_policies


# ────────────────────────── main ──────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Inspect buffer & NN for key board states')
    parser.add_argument('--game', default='Connect4', choices=list(GAME_CONFIGS.keys()))
    parser.add_argument('--buffer', default='dataset/dataset.pt')
    parser.add_argument('--model', default=None)
    parser.add_argument('--best', action='store_true', help='Use best model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--output', default='tools/figures', help='Output directory for plots')
    parser.add_argument('--font', default=None, help='Matplotlib font family name for Chinese')
    parser.add_argument('--font-path', default=None, help='Path to .ttf/.otf/.ttc font file')
    parser.add_argument('--no-buffer', action='store_true')
    parser.add_argument('--no-nn', action='store_true')
    args = parser.parse_args()

    gcfg = GAME_CONFIGS[args.game]
    if args.model is None:
        args.model = gcfg['best_model'] if args.best else gcfg['default_model']
    elif args.best:
        args.model = gcfg['best_model']

    os.chdir(ROOT)
    setup_matplotlib_font(args.font, args.font_path)

    # 先跑 NN，拿到 raw policy 供 buffer 绘图叠加
    nn_policies = {}
    if not args.no_nn:
        nn_policies = analyze_nn(args.model, gcfg, args.device)

    if not args.no_buffer:
        analyze_buffer(args.buffer, args.top, args.output, nn_policies, gcfg)


if __name__ == '__main__':
    main()
