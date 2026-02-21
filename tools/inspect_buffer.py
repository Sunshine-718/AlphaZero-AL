"""
查看 dataset.pt 中空棋盘的 prob / winner 分布，以及 NN 对空棋盘的输出。
统计全量数据并生成可视化图片。

用法:
    python tools/inspect_buffer.py
    python tools/inspect_buffer.py --buffer dataset/dataset.pt
    python tools/inspect_buffer.py --model params/AZ_Connect4_CNN_best.pt
    python tools/inspect_buffer.py --font "Microsoft YaHei"
    python tools/inspect_buffer.py --font-path /path/to/NotoSansCJK-Regular.ttc
    python tools/inspect_buffer.py --no-buffer   # 只看 NN
    python tools/inspect_buffer.py --no-nn       # 只看 buffer
    python tools/inspect_buffer.py --output figs # 指定图片输出目录
"""
import sys, os, argparse, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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

BAR = '─' * 72
COL_LABELS = [f'col{i}' for i in range(7)]
WINNER_LABELS = {1: 'P1 wins', -1: 'P2 wins', 0: 'Draw'}
WINNER_COLORS = {1: '#4C72B0', -1: '#DD8452', 0: '#55A868'}

# 棋盘模式: (描述, 构造函数, turn, 文件名前缀)
def make_empty():
    return np.zeros((6, 7), dtype=np.float32)

def make_first_move():
    b = np.zeros((6, 7), dtype=np.float32)
    b[5, 3] = 1
    return b

PATTERNS = [
    ('空棋盘',           make_empty,      1,  'empty_board_X'),
    ('X 下中间列后 (O)', make_first_move, -1, 'after_center_O'),
]


def setup_matplotlib_font(font_name=None, font_path=None):
    """配置 matplotlib 中文字体。"""
    chosen = None

    if font_path:
        if not os.path.exists(font_path):
            print(f'[!] 指定字体文件不存在: {font_path}')
        else:
            try:
                fm.fontManager.addfont(font_path)
                chosen = fm.FontProperties(fname=font_path).get_name()
            except Exception as e:
                print(f'[!] 加载字体文件失败: {font_path} ({e})')

    available = {f.name for f in fm.fontManager.ttflist}
    if chosen is None and font_name:
        if font_name in available:
            chosen = font_name
        else:
            print(f'[!] 指定字体名未找到: {font_name}')

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
        print(f'[font] 使用中文字体: {chosen}')
        return

    # 环境里没有中文字体时，避免保存图片时大量重复 glyph 警告刷屏
    warnings.filterwarnings(
        'ignore',
        message=r'Glyph .* missing from font\(s\) DejaVu Sans',
        category=UserWarning,
    )
    print('[font] 未检测到可用中文字体，中文标题可能显示为方框。可用 --font-path 指定 .ttf/.otf。')


def board_matches(state_int8, board, turn):
    ch0 = state_int8[0]
    ch1 = state_int8[1]
    t = 1 if state_int8[2, 0, 0] >= 0 else -1
    if t != turn:
        return False
    p1_match = (ch0 == (board == 1).astype(np.int8)).all()
    p2_match = (ch1 == (board == -1).astype(np.int8)).all()
    return bool(p1_match and p2_match)


def fmt_prob(prob):
    parts = []
    best = prob.argmax()
    for i, p in enumerate(prob):
        marker = '*' if i == best else ' '
        parts.append(f'{marker}col{i}:{p*100:5.1f}%')
    return '  '.join(parts)


# ────────────────────────── 绘图 ──────────────────────────

def plot_pattern(desc, ep, ew, es, nn_prob, output_dir, fname):
    """为一个 pattern 生成 2x2 子图并保存。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{desc}  (n={len(ep)})', fontsize=15, fontweight='bold')

    # ── (a) Winner 饼图 ──
    ax = axes[0, 0]
    counts = []
    labels = []
    colors = []
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
    x = np.arange(7)
    bars = ax.bar(x, mean_p, yerr=std_p, capsize=4, color='#4C72B0',
                  alpha=0.8, label='Buffer MCTS mean')
    if nn_prob is not None:
        ax.scatter(x, nn_prob, color='#C44E52', s=80, zorder=5,
                   marker='D', label='NN raw policy')
    ax.set_xticks(x)
    ax.set_xticklabels(COL_LABELS)
    ax.set_ylabel('Probability')
    ax.set_title('Policy Distribution')
    ax.legend(fontsize=9)
    # 标注数值
    for i, v in enumerate(mean_p):
        ax.text(i, v + std_p[i] + 0.01, f'{v*100:.1f}%', ha='center', fontsize=8)

    # ── (c) Steps-to-end 直方图 (按 winner 堆叠) ──
    ax = axes[1, 0]
    if es is not None:
        s_np = es.numpy()
        bins = np.arange(s_np.min() - 0.5, s_np.max() + 1.5, 1)
        stack_data = []
        stack_labels = []
        stack_colors = []
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
    bp_data = [ep[:, i].numpy() for i in range(7)]
    boxplot_kwargs = dict(
        patch_artist=True,
        showfliers=True,
        flierprops=dict(markersize=2, alpha=0.3),
    )
    try:
        # Matplotlib >= 3.9
        bp = ax.boxplot(bp_data, tick_labels=COL_LABELS, **boxplot_kwargs)
    except TypeError:
        # 兼容旧版本 Matplotlib
        bp = ax.boxplot(bp_data, labels=COL_LABELS, **boxplot_kwargs)
    for patch in bp['boxes']:
        patch.set_facecolor('#A1C9F4')
    if nn_prob is not None:
        ax.scatter(range(1, 8), nn_prob, color='#C44E52', s=60, zorder=5,
                   marker='D', label='NN raw policy')
        ax.legend(fontsize=9)
    ax.set_ylabel('Probability')
    ax.set_title('Policy Spread (Box Plot)')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{fname}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    [saved] {path}')


# ────────────────────────── Buffer ──────────────────────────

def analyze_buffer(path, top_n, output_dir, nn_policies):
    if not os.path.exists(path):
        print(f'[!] Buffer 不存在: {path}')
        return

    print(f'\n{BAR}\n  Buffer: {path}\n{BAR}')
    data = torch.load(path, map_location='cpu', weights_only=True)
    states  = data['state']
    probs   = data['prob']
    winners = data['winner']
    steps   = data.get('steps_to_end')
    ptr     = data['_ptr']
    n = min(ptr, states.shape[0])
    print(f'  容量={states.shape[0]}, ptr={ptr}, 有效={n}\n')

    for desc, make_board, turn, fname in PATTERNS:
        board = make_board()
        board_np = board.astype(np.int8)
        idx = [i for i in range(n) if board_matches(states[i], board_np, turn)]

        player = 'X' if turn == 1 else 'O'
        print(f'  ══ {desc} (turn={player}) ══')

        if not idx:
            print(f'    未找到匹配样本。\n')
            continue

        idx_t = torch.tensor(idx)
        ep = probs[idx_t]
        ew = winners[idx_t].view(-1)
        es = steps[idx_t].view(-1) if steps is not None else None

        print(f'    样本数: {len(idx)}')

        # winner 分布
        print(f'    Winner 分布:')
        for label, val in [('P1 wins', 1), ('P2 wins', -1), ('Draw', 0)]:
            cnt = int((ew == val).sum())
            print(f'      {label:8s}: {cnt:5d}  ({cnt/len(idx)*100:5.1f}%)')

        if es is not None:
            print(f'    Steps-to-end: mean={es.float().mean():.1f}, min={es.min()}, max={es.max()}')

        mp = ep.mean(dim=0).numpy()
        print(f'    Policy 均值: {fmt_prob(mp)}')

        # 前 top_n 条
        show = min(top_n, len(idx))
        print(f'    前 {show} 条:')
        print(f'    {"idx":>6s} {"winner":>8s} {"steps":>5s}  policy')
        for k in range(show):
            i = idx[k]
            w = WINNER_LABELS[int(ew[k])]
            s = int(es[k]) if es is not None else '?'
            print(f'    {i:6d} {w:>8s} {s:>5}  {fmt_prob(ep[k].numpy())}')

        # 绘图
        nn_prob = nn_policies.get(fname) if nn_policies else None
        plot_pattern(desc, ep, ew, es, nn_prob, output_dir, fname)
        print()


# ────────────────────────── NN ──────────────────────────

def analyze_nn(model_path, device='cpu'):
    """加载 NN 并返回每个 pattern 的 raw policy，同时打印摘要。"""
    print(f'\n{BAR}\n  NN Model: {model_path}\n{BAR}')
    if not os.path.exists(model_path):
        print(f'  [!] 模型不存在: {model_path}')
        return {}

    from src.environments.Connect4.Network import CNN
    from src.environments.Connect4.utils import board_to_state

    net = CNN(lr=3e-3, in_dim=3, h_dim=128, out_dim=7,
              dropout=0.2, device=device, num_res_blocks=3)
    net.load(model_path)
    net.eval()

    nn_policies = {}
    for desc, make_board, turn, fname in PATTERNS:
        board = make_board()
        player = 'X' if turn == 1 else 'O'
        state = board_to_state(board, turn)
        t = torch.from_numpy(state).to(device)
        with torch.no_grad():
            log_p, log_v, log_s = net(t)
            prob = log_p.exp().cpu().numpy().flatten()
            vdist = log_v.exp().cpu().numpy().flatten()  # [draw, P1 wins, P2 wins]
            sdist = log_s.exp().cpu().numpy().flatten()

        nn_policies[fname] = prob
        scalar_v = turn * (vdist[1] - vdist[2])
        entropy = -np.sum(prob * np.log(prob + 1e-8))
        top5s = np.argsort(sdist)[::-1][:5]

        print(f'\n  ── {desc} (turn={player}) ──')
        print(f'    Policy:       {fmt_prob(prob)}')
        print(f'    Value (3way): Draw:{vdist[0]*100:5.1f}%  P1win:{vdist[1]*100:5.1f}%  P2win:{vdist[2]*100:5.1f}%')
        print(f'    Scalar value: {scalar_v: .4f}')
        print(f'    Entropy:      {entropy:.4f}')
        print(f'    Steps top-5:  {", ".join(f"s={s}:{sdist[s]*100:.1f}%" for s in top5s)}')
    print()
    return nn_policies


# ────────────────────────── main ──────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Inspect buffer & NN for key board states')
    parser.add_argument('--buffer', default='dataset/dataset.pt')
    parser.add_argument('--model', default='params/AZ_Connect4_CNN_current.pt')
    parser.add_argument('--best', action='store_true', help='Use best model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--output', default='tools/figures', help='Output directory for plots')
    parser.add_argument('--font', default=None, help='Matplotlib font family name for Chinese')
    parser.add_argument('--font-path', default=None, help='Path to .ttf/.otf/.ttc font file')
    parser.add_argument('--no-buffer', action='store_true')
    parser.add_argument('--no-nn', action='store_true')
    args = parser.parse_args()
    if args.best:
        args.model = 'params/AZ_Connect4_CNN_best.pt'
    os.chdir(ROOT)
    setup_matplotlib_font(args.font, args.font_path)

    # 先跑 NN，拿到 raw policy 供 buffer 绘图叠加
    nn_policies = {}
    if not args.no_nn:
        nn_policies = analyze_nn(args.model, args.device)

    if not args.no_buffer:
        analyze_buffer(args.buffer, args.top, args.output, nn_policies)


if __name__ == '__main__':
    main()
