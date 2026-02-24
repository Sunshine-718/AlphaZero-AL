"""
MCTS 根节点诊断脚本
对关键棋盘局面运行 Python MCTS，提取 Q/N/prior/noise 等根节点统计量，
帮助定位 O 策略不能收敛的根本原因。

用法:
    python tools/diagnose_mcts.py
    python tools/diagnose_mcts.py --model params/AZ_Connect4_CNN_current.pt
    python tools/diagnose_mcts.py --quick       # 快速模式，减少运行次数
"""
import sys, os, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, track

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.environments.Connect4.env import Env
from src.environments.Connect4.Network import CNN
from src.MCTS import MCTS, MCTS_AZ
from src.utils import policy_value_fn as rollout_pv_fn, evaluate_rollout

console = Console()

# ──────── 生产参数（精确匹配 C++）────────
C_INIT       = 1.0
C_BASE       = 100_000   # C++ = n * c_base_factor = 100 * 1000
N_PLAYOUT    = 100
DISCOUNT     = 1.0
ALPHA        = 0.3
NOISE_EPS    = 0.25
FPU_REDUCTION = 0.2
LAMBDA_S     = 0.1

# ──────── 诊断参数 ────────
N_RUNS       = 30
SIM_BUDGETS  = [100, 200, 400, 800, 1600]
N_RUNS_SWEEP = 20
PURE_MCTS_BUDGETS = [1000, 10000, 50000]
PURE_MCTS_C_INITS = [0.1, 0.3, 0.5, 1, 2, 4, 8, 16]
# Section 8: AZ MCTS 超参扫描
AZ_C_INITS   = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
AZ_ALPHAS    = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
AZ_FPUS      = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
N_RUNS_HP    = 20
OUTPUT_DIR   = 'tools/figures'

COL_COLORS = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']


# ═══════════════════════ helpers ═══════════════════════

def make_positions():
    """返回待诊断的局面列表: [(description, env, ply)]"""
    positions = []

    e1 = Env()
    positions.append(('Empty board (X, ply=1)', e1, 1))

    e2 = Env(); e2.step(3)
    positions.append(('X->col3 (O, ply=2)', e2, 2))

    e3 = Env(); e3.step(3); e3.step(3)
    positions.append(('X->3,O->3 (X, ply=3)', e3, 3))

    e4 = Env(); e4.step(3); e4.step(3); e4.step(2)
    positions.append(('X->3,O->3,X->2 (O, ply=4)', e4, 4))

    return positions


def load_model(path, device='cpu'):
    net = CNN(lr=3e-3, in_dim=3, h_dim=128, out_dim=7,
              dropout=0.2, device=device, num_res_blocks=3, lambda_s=LAMBDA_S)
    net.load(path)
    net.eval()
    return net


def make_mcts(net, n_playout=N_PLAYOUT, deterministic=False):
    mcts = MCTS_AZ(
        policy_value_fn=net,
        c_init=C_INIT,
        n_playout=n_playout,
        discount=DISCOUNT,
        alpha=None if deterministic else ALPHA,
        cache_size=0,
        eps=NOISE_EPS,
        fpu_reduction=FPU_REDUCTION,
        use_symmetry=False,
    )
    mcts.c_base = C_BASE          # Python 默认 500, C++ 用 100000
    if deterministic:
        mcts.eval()
    return mcts


def run_mcts_once(net, env, n_playout=N_PLAYOUT, deterministic=False):
    """运行一次 MCTS，返回 root 节点"""
    mcts = make_mcts(net, n_playout, deterministic)
    # get_action_visits 在 root 上做 n_playout 次 playout
    list(mcts.get_action_visits(env.copy()))   # consume the iterator
    return mcts.root


def nn_eval(net, env):
    """用 NN 评估局面，返回 (probs, scalar_value, vdist_3way, expected_steps)"""
    state = env.current_state()
    probs, value, _ = net.predict(state)
    probs = probs.flatten()
    value = value.flatten()[0]
    t = torch.from_numpy(state).float()
    with torch.no_grad():
        _, log_v, log_s = net(t)
        vdist = log_v.exp().cpu().numpy().flatten()   # [draw, p1win, p2win]
        sdist = log_s.exp().cpu().numpy().flatten()
    expected_steps = float(np.sum(sdist * np.arange(43)))
    return probs, value, vdist, expected_steps


# ═══════════════════ Section 1: NN child values ═══════════════════

def section1(net, env, desc):
    console.print(Panel(
        f'[bold]Section 1: NN Child Value Evaluation[/bold]\n{desc}',
        border_style='cyan'))

    probs, root_val, vdist, esteps = nn_eval(net, env)
    player = 'X' if env.turn == 1 else 'O'

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column('K', style='bold'); t.add_column('V')
    t.add_row('Player', player)
    t.add_row('Value', f'{root_val:+.4f}')
    t.add_row('3-way', f'Draw={vdist[0]*100:.1f}%  P1win={vdist[1]*100:.1f}%  P2win={vdist[2]*100:.1f}%')
    t.add_row('E[steps]', f'{esteps:.1f}')
    t.add_row('Policy', '  '.join(f'col{i}:{probs[i]*100:.1f}%' for i in range(7)))
    console.print(t)

    # Evaluate each child
    valid = env.valid_move()
    child_data = []

    ct = Table(title='Child NN Values (-V = from parent perspective)',
               show_header=True, header_style='bold')
    ct.add_column('col', justify='center')
    ct.add_column('-V_child', justify='right')
    ct.add_column('Draw%', justify='right')
    ct.add_column('P1w%', justify='right')
    ct.add_column('P2w%', justify='right')
    ct.add_column('E[steps]', justify='right')
    ct.add_column('Prior', justify='right')

    for col in valid:
        ce = env.copy(); ce.step(col)
        cp, cv, cvd, ces = nn_eval(net, ce)
        child_data.append(dict(col=col, neg_v=-cv, vdist=cvd, esteps=ces, prior=probs[col]))
        ct.add_row(f'col{col}', f'{-cv:+.4f}',
                   f'{cvd[0]*100:.1f}', f'{cvd[1]*100:.1f}', f'{cvd[2]*100:.1f}',
                   f'{ces:.1f}', f'{probs[col]*100:.1f}%')

    console.print(ct)

    # Delta between center columns
    v_map = {d['col']: d['neg_v'] for d in child_data}
    cc = [c for c in [2, 3, 4] if c in v_map]
    if len(cc) >= 2:
        console.print('[bold]Center column deltas (-V, higher = better for current player):[/bold]')
        for i, c1 in enumerate(cc):
            for c2 in cc[i+1:]:
                d = v_map[c1] - v_map[c2]
                console.print(f'  col{c1} - col{c2} = {d:+.4f}')

    return child_data


# ═══════════════════ Section 2: single MCTS run ═══════════════════

def section2(net, env, desc):
    console.print(Panel(
        f'[bold]Section 2: Single MCTS Run (n={N_PLAYOUT})[/bold]\n{desc}',
        border_style='green'))

    root = run_mcts_once(net, env)
    total_n = sum(c.n_visits for c in root.children.values())
    seen_pol = sum(c.prior for c in root.children.values() if c.n_visits > 0)
    fpu = max(-1.0, root.Q - FPU_REDUCTION * np.sqrt(seen_pol))

    t = Table(title='Root Children', show_header=True, header_style='bold')
    for h in ['col', 'N', 'N%', 'Q', '-Q', 'Prior', 'Noise', 'EffPrior']:
        t.add_column(h, justify='right' if h != 'col' else 'center')

    for a in sorted(root.children):
        n = root.children[a]
        ep = (1 - NOISE_EPS) * n.prior + NOISE_EPS * n.noise
        t.add_row(f'col{a}', str(n.n_visits),
                  f'{n.n_visits/total_n*100:.1f}%',
                  f'{n.Q:+.4f}', f'{-n.Q:+.4f}',
                  f'{n.prior:.4f}', f'{n.noise:.4f}', f'{ep:.4f}')

    console.print(t)
    console.print(f'  Root.Q={root.Q:+.4f}  FPU={fpu:+.4f}  seen_policy={seen_pol:.4f}  total_N={total_n}')
    return root


# ═══════════════════ Section 3: aggregate ═══════════════════

def section3(net, env, desc, n_runs):
    console.print(Panel(
        f'[bold]Section 3: Aggregate ({n_runs} runs, n={N_PLAYOUT})[/bold]\n{desc}',
        border_style='yellow'))

    all_N = {i: [] for i in range(7)}
    all_Q = {i: [] for i in range(7)}

    for _ in track(range(n_runs), description='  MCTS runs', console=console, transient=True):
        root = run_mcts_once(net, env)
        for a, nd in root.children.items():
            all_N[a].append(nd.n_visits)
            all_Q[a].append(nd.Q)

    t = Table(title=f'Aggregated ({n_runs} runs)', show_header=True, header_style='bold')
    for h in ['col', 'N_mean', 'N_std', 'N%', 'Q_mean', 'Q_std', '-Q_mean']:
        t.add_column(h, justify='right' if h != 'col' else 'center')

    agg = {}
    for col in sorted(all_N):
        if not all_N[col]: continue
        na, qa = np.array(all_N[col]), np.array(all_Q[col])
        agg[col] = dict(N_mean=na.mean(), N_std=na.std(),
                        Q_mean=qa.mean(), Q_std=qa.std())
        t.add_row(f'col{col}',
                  f'{na.mean():.1f}', f'{na.std():.1f}',
                  f'{na.mean()/N_PLAYOUT*100:.1f}%',
                  f'{qa.mean():+.4f}', f'{qa.std():.4f}',
                  f'{-qa.mean():+.4f}')

    console.print(t)

    # SNR
    cc = [c for c in [2, 3, 4] if c in agg]
    if len(cc) >= 2:
        console.print('[bold]Signal-to-noise (Q):[/bold]')
        for i, c1 in enumerate(cc):
            for c2 in cc[i+1:]:
                d = agg[c1]['Q_mean'] - agg[c2]['Q_mean']
                n = np.sqrt(agg[c1]['Q_std']**2 + agg[c2]['Q_std']**2)
                snr = abs(d) / n if n > 0 else float('inf')
                console.print(f'  Q(col{c1})-Q(col{c2})={d:+.4f}  noise={n:.4f}  SNR={snr:.3f}')

    return agg


# ═══════════════════ Section 4: sim budget sweep ═══════════════════

def section4(net, env, desc, budgets, n_runs):
    console.print(Panel(
        f'[bold]Section 4: Simulation Budget Sweep[/bold]\n{desc}',
        border_style='magenta'))

    sweep = {}
    for ns in budgets:
        pcts = {i: [] for i in range(7)}
        for _ in track(range(n_runs), description=f'  n={ns}', console=console, transient=True):
            root = run_mcts_once(net, env, n_playout=ns)
            total = sum(c.n_visits for c in root.children.values())
            for a, nd in root.children.items():
                pcts[a].append(nd.n_visits / total * 100)
        sweep[ns] = {}
        parts = [f'n={ns:5d}:']
        for col in sorted(pcts):
            if pcts[col]:
                m = np.mean(pcts[col])
                sweep[ns][col] = m
                parts.append(f'col{col}={m:.1f}%')
        console.print('  ' + '  '.join(parts))

    return sweep


# ═══════════════════ Section 5: noise ablation ═══════════════════

def section5(net, env, desc, n_runs):
    console.print(Panel(
        f'[bold]Section 5: Noise Ablation (n={N_PLAYOUT})[/bold]\n{desc}',
        border_style='red'))

    results = {}
    for label, det in [('With noise (eps=0.25)', False), ('No noise (deterministic)', True)]:
        pcts = {i: [] for i in range(7)}
        qs   = {i: [] for i in range(7)}
        for _ in track(range(n_runs), description=f'  {label}', console=console, transient=True):
            root = run_mcts_once(net, env, deterministic=det)
            total = sum(c.n_visits for c in root.children.values())
            for a, nd in root.children.items():
                pcts[a].append(nd.n_visits / total * 100)
                qs[a].append(nd.Q)
        results[label] = {}
        parts = [f'{label}:']
        for col in sorted(pcts):
            if pcts[col]:
                mp, mq = np.mean(pcts[col]), np.mean(qs[col])
                results[label][col] = dict(pct=mp, Q=mq)
                parts.append(f'col{col}={mp:.1f}%')
        console.print('  ' + '  '.join(parts))

    return results


# ═══════════════════ Section 7: pure MCTS ═══════════════════

def section7(env, desc, budgets, c_inits=None):
    """纯 MCTS (UCT + random rollout), 无 NN 参与"""
    if c_inits is None:
        c_inits = [4]
    console.print(Panel(
        f'[bold]Section 7: Pure MCTS (UCT + random rollout)[/bold]\n{desc}',
        border_style='blue'))

    results = {}   # {(c_init, n): {col: {pct, Q, N}}}
    for c in c_inits:
        console.print(f'  [bold]c_init={c}[/bold]')
        for ns in budgets:
            console.print(f'    [dim]Running pure MCTS c={c} n={ns} ...[/dim]')
            pure = MCTS(rollout_pv_fn, c_init=c, n_playout=ns, discount=1.0,
                        alpha=None, use_symmetry=False)
            pure.get_action(env.copy())
            root = pure.root
            total = sum(ch.n_visits for ch in root.children.values())

            key = (c, ns)
            results[key] = {}
            parts = [f'c={c:2d} n={ns:6d}:']
            for col in sorted(root.children):
                nd = root.children[col]
                pct = nd.n_visits / total * 100
                results[key][col] = dict(pct=pct, Q=nd.Q, N=nd.n_visits)
                parts.append(f'col{col}={pct:.1f}%(Q={-nd.Q:+.3f})')
            console.print('    ' + '  '.join(parts))

    # Summary table: c_init x col3 visit%
    console.print()
    t = Table(title='Pure MCTS: col3 visit% (c_init x n_playout)',
              show_header=True, header_style='bold')
    t.add_column('c_init', justify='center')
    for ns in budgets:
        t.add_column(f'n={ns}', justify='right')
    for c in c_inits:
        row = [str(c)]
        for ns in budgets:
            d = results.get((c, ns), {}).get(3, {})
            pct = d.get('pct', 0)
            q = d.get('Q', 0)
            row.append(f'{pct:.1f}% (Q={-q:+.3f})')
        t.add_row(*row)
    console.print(t)

    return results


# ═══════════════════ Section 8: pure MCTS hyperparam sweep ═══════════════════

class RolloutAdapter:
    """Random rollout adapter — 让 MCTS_AZ 可以用 random rollout 代替 NN"""
    n_actions = 7

    def predict(self, state):
        env = Env()
        env.board = (state[0, 0] - state[0, 1]).astype(np.float32)
        env.turn = 1 if state[0, 2, 0, 0] > 0 else -1
        probs = np.ones((1, self.n_actions), dtype=np.float32) / self.n_actions
        val = evaluate_rollout(env.copy())
        return probs, np.array([[val]], dtype=np.float32), np.array([[0.5]], dtype=np.float32)


_ROLLOUT_ADAPTER = RolloutAdapter()


def _run_pure_hp_sweep(env, param_name, param_values, n_runs, n_playout, **defaults):
    """对纯 MCTS 的某个超参进行扫描 (PUCT + uniform prior + random rollout) — 多进程"""
    _tools_dir = os.path.dirname(os.path.abspath(__file__))
    if _tools_dir not in sys.path:
        sys.path.insert(0, _tools_dir)
    from mcts_worker import run_single
    board = env.board.copy()
    turn = env.turn
    n_workers = min(os.cpu_count() or 4, 8, len(param_values) * n_runs)

    # 构建所有任务: (param_val, run_idx) -> future
    tasks = []
    for val in param_values:
        kwargs = dict(c_init=4, alpha=None, fpu_reduction=0.0, discount=1.0)
        kwargs.update(defaults)
        kwargs[param_name] = val
        for _ in range(n_runs):
            tasks.append((val, kwargs['c_init'], kwargs['alpha'],
                          kwargs['fpu_reduction'], n_playout, kwargs['discount']))

    # 收集结果
    raw = {val: [] for val in param_values}
    total_tasks = len(tasks)
    with Progress(console=console) as progress:
        bar = progress.add_task(f'  {param_name} sweep', total=total_tasks)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for val, c_init, alpha, fpu, npl, disc in tasks:
                f = pool.submit(run_single, board, turn, c_init, alpha, fpu, npl, disc)
                futures[f] = val
            for f in as_completed(futures):
                val = futures[f]
                raw[val].append(f.result())
                progress.update(bar, advance=1,
                                description=f'  {param_name} sweep ({val})')

    # 聚合
    results = {}
    for val in param_values:
        pcts = {i: [] for i in range(7)}
        qs   = {i: [] for i in range(7)}
        for col_pcts, col_qs in raw[val]:
            for a in col_pcts:
                pcts[a].append(col_pcts[a])
                qs[a].append(col_qs[a])
        results[val] = {}
        for col in sorted(pcts):
            if pcts[col]:
                results[val][col] = dict(pct_mean=np.mean(pcts[col]),
                                         pct_std=np.std(pcts[col]),
                                         Q_mean=np.mean(qs[col]))
    return results


def section8(env, desc, c_inits, alphas, fpus, n_runs, n_playout):
    """纯 MCTS 超参扫描: c_init / alpha / fpu_reduction (PUCT + uniform prior + random rollout)"""
    console.print(Panel(
        f'[bold]Section 8: Pure MCTS Hyperparam Sweep '
        f'(PUCT + rollout, n={n_playout}, {n_runs} runs)[/bold]\n{desc}',
        border_style='bright_magenta'))

    all_results = {}

    # ── c_init sweep at each discount (alpha=0.3, fpu=0.2) ──
    for disc in [1.0, 0.99, 0.975]:
        label = f'c_init (discount={disc})'
        console.print(f'  [bold]Sweep c_init[/bold]  (alpha=0.3, fpu=0.2, discount={disc})')
        res_c = _run_pure_hp_sweep(env, 'c_init', c_inits, n_runs, n_playout,
                                   alpha=0.3, fpu_reduction=0.2, discount=disc)
        all_results[label] = (c_inits, res_c)
        for val in c_inits:
            d3 = res_c[val].get(3, {})
            console.print(f'    c_init={val:<5}  col3={d3.get("pct_mean",0):.1f}%  '
                          f'-Q={-d3.get("Q_mean",0):+.3f}')

    return all_results


def plot_hp_sweep(hp_results_list, descs, out):
    """画纯 MCTS 超参扫描图: 每个 position 一行, 动态列"""
    n_pos = len(hp_results_list)
    # 收集所有出现过的 param key，保持顺序
    all_keys = []
    for hp_res in hp_results_list:
        for k in hp_res:
            if k not in all_keys:
                all_keys.append(k)
    n_cols = max(len(all_keys), 1)
    fig, axes = plt.subplots(n_pos, n_cols, figsize=(6*n_cols, 5*n_pos), squeeze=False)

    for row, (hp_res, desc) in enumerate(zip(hp_results_list, descs)):
        for col_idx, pname in enumerate(all_keys):
            ax = axes[row][col_idx]
            if pname not in hp_res:
                ax.set_visible(False)
                continue
            param_vals, sweep_data = hp_res[pname]
            all_cols = sorted(set().union(*(sweep_data[v].keys() for v in param_vals)))
            for c in all_cols:
                y = [sweep_data[v].get(c, {}).get('pct_mean', 0) for v in param_vals]
                style = '-o' if c == 3 else '--s'
                lw = 2.5 if c == 3 else 1
                alpha_v = 1.0 if c == 3 else 0.4
                ax.plot(range(len(param_vals)), y, style,
                        color=COL_COLORS[c % 7], label=f'col{c}',
                        lw=lw, ms=6, alpha=alpha_v)
            ax.set_xticks(range(len(param_vals)))
            ax.set_xticklabels([str(v) for v in param_vals], fontsize=8)
            ax.set_xlabel(pname)
            ax.set_ylabel('Visit %')
            ax.set_title(f'{desc}\n{pname} sweep', fontsize=9)
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.3)

    fig.suptitle('Pure MCTS Hyperparam Sweep (PUCT + rollout)', fontweight='bold', fontsize=13)
    plt.tight_layout()
    p = os.path.join(out, 'pure_mcts_hp_sweep.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    console.print(f'  [dim][saved] {p}[/dim]')


# ═══════════════════════ Plots ═══════════════════════

def _setup_font():
    """尝试设置中文字体，失败则忽略"""
    import matplotlib.font_manager as fm
    import warnings
    for name in ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei',
                 'Noto Sans CJK SC', 'WenQuanYi Micro Hei']:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            break
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore', message=r'Glyph .* missing from font')


def plot_child_values(child_data_list, descs, out):
    n = len(child_data_list)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), squeeze=False)
    axes = axes[0]
    for ax, cd, desc in zip(axes, child_data_list, descs):
        cols = [d['col'] for d in cd]
        vals = [d['neg_v'] for d in cd]
        colors = [COL_COLORS[c % 7] for c in cols]
        bars = ax.bar([f'col{c}' for c in cols], vals, color=colors, alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                    f'{v:+.3f}', ha='center', va='bottom', fontsize=7)
        ax.axhline(0, color='gray', ls='--', alpha=.4)
        ax.set_ylabel('-V (parent perspective)')
        ax.set_title(desc, fontsize=9)
    fig.suptitle('NN Child Value Evaluation', fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, 'mcts_child_values.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    console.print(f'  [dim][saved] {p}[/dim]')


def plot_q_vs_visits(agg_list, descs, out):
    n = len(agg_list)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), squeeze=False)
    axes = axes[0]
    for ax, agg, desc in zip(axes, agg_list, descs):
        for col in sorted(agg):
            d = agg[col]
            ax.errorbar(d['N_mean'], -d['Q_mean'],
                        xerr=d['N_std'], yerr=d['Q_std'],
                        fmt='o', color=COL_COLORS[col % 7],
                        label=f'col{col}', ms=8, capsize=4)
        ax.set_xlabel('Visit count (N)')
        ax.set_ylabel('-Q (parent perspective)')
        ax.set_title(desc, fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=.3)
    fig.suptitle('Q vs Visits (aggregated)', fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, 'mcts_q_vs_visits.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    console.print(f'  [dim][saved] {p}[/dim]')


def plot_sim_sweep(sweep_list, descs, budgets, out):
    n = len(sweep_list)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5), squeeze=False)
    axes = axes[0]
    for ax, sw, desc in zip(axes, sweep_list, descs):
        all_cols = sorted(set().union(*[s.keys() for s in sw.values()]))
        for col in all_cols:
            y = [sw[b].get(col, 0) for b in budgets]
            ax.plot(budgets, y, 'o-', color=COL_COLORS[col % 7],
                    label=f'col{col}', lw=2, ms=5)
        ax.set_xlabel('Simulations'); ax.set_ylabel('Visit %')
        ax.set_xscale('log', base=2); ax.set_title(desc, fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=.3)
    fig.suptitle('Simulation Budget Sweep', fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, 'sim_budget_sweep.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    console.print(f'  [dim][saved] {p}[/dim]')


def plot_noise_ablation(nr_list, descs, out):
    n = len(nr_list)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5), squeeze=False)
    axes = axes[0]
    for ax, nr, desc in zip(axes, nr_list, descs):
        labels = list(nr.keys())
        all_cols = sorted(set().union(*[r.keys() for r in nr.values()]))
        x = np.arange(len(all_cols)); w = 0.35
        for i, lab in enumerate(labels):
            vals = [nr[lab].get(c, {}).get('pct', 0) for c in all_cols]
            ax.bar(x + (i - 0.5)*w, vals, w, label=lab, alpha=.8)
        ax.set_xticks(x); ax.set_xticklabels([f'col{c}' for c in all_cols])
        ax.set_ylabel('Visit %'); ax.set_title(desc, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle('Noise Ablation', fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, 'noise_ablation.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    console.print(f'  [dim][saved] {p}[/dim]')


def plot_multi_ply(agg_list, descs, out):
    n = len(agg_list)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5*rows), squeeze=False)
    axes_flat = axes.flatten()
    for idx, (ax, agg, desc) in enumerate(zip(axes_flat, agg_list, descs)):
        cols = sorted(agg)
        nmeans = [agg[c]['N_mean'] for c in cols]
        total = sum(nmeans)
        pcts = [m / total * 100 for m in nmeans]
        qmeans = [-agg[c]['Q_mean'] for c in cols]
        colors = [COL_COLORS[c % 7] for c in cols]
        bars = ax.bar([f'col{c}' for c in cols], pcts, color=colors, alpha=.85)
        for b, q, p in zip(bars, qmeans, pcts):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.3,
                    f'Q={q:+.3f}', ha='center', va='bottom', fontsize=7)
        ax.set_ylabel('Visit %'); ax.set_title(desc, fontsize=9)
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.suptitle('Multi-Ply MCTS Analysis', fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, 'multi_ply.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    console.print(f'  [dim][saved] {p}[/dim]')


def plot_pure_mcts(pure_results_list, az_agg_list, descs, budgets, c_inits, out):
    """图1: Pure MCTS c_init sweep — col3 visit% heatmap"""
    n_pos = len(pure_results_list)
    fig, axes = plt.subplots(1, n_pos, figsize=(7*n_pos, max(4, len(c_inits)*0.6+2)), squeeze=False)
    axes = axes[0]
    for ax, pure_res, desc in zip(axes, pure_results_list, descs):
        mat = np.zeros((len(c_inits), len(budgets)))
        for i, c in enumerate(c_inits):
            for j, ns in enumerate(budgets):
                d = pure_res.get((c, ns), {}).get(3, {})
                mat[i, j] = d.get('pct', 0)
        im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=10, vmax=35)
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_yticks(range(len(c_inits)))
        ax.set_yticklabels([str(c) for c in c_inits])
        ax.set_xlabel('n_playout')
        ax.set_ylabel('c_init')
        ax.set_title(f'{desc}\ncol3 visit %', fontsize=9)
        for i in range(len(c_inits)):
            for j in range(len(budgets)):
                ax.text(j, i, f'{mat[i,j]:.1f}%', ha='center', va='center', fontsize=8,
                        color='white' if mat[i,j] > 25 else 'black')
    fig.suptitle('Pure MCTS: col3 Exploration (c_init x n_playout)', fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, 'pure_mcts_c_sweep.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    console.print(f'  [dim][saved] {p}[/dim]')

    # 图2: AlphaZero vs Pure MCTS (最高 budget, 默认 c_init=4) 对比
    if az_agg_list:
        default_c = 4 if 4 in c_inits else c_inits[0]
        fig2, axes2 = plt.subplots(1, n_pos, figsize=(8*n_pos, 6), squeeze=False)
        axes2 = axes2[0]
        for ax, pure_res, az_agg, desc in zip(axes2, pure_results_list, az_agg_list, descs):
            az_cols = sorted(az_agg)
            az_total = sum(az_agg[c]['N_mean'] for c in az_cols)
            az_pcts = {c: az_agg[c]['N_mean'] / az_total * 100 for c in az_cols}
            max_n = max(budgets)
            pure_data = pure_res.get((default_c, max_n), {})
            pure_cols = sorted(pure_data)
            all_cols = sorted(set(az_cols) | set(pure_cols))
            x = np.arange(len(all_cols))
            w = 0.35
            az_vals = [az_pcts.get(c, 0) for c in all_cols]
            pure_vals = [pure_data.get(c, {}).get('pct', 0) for c in all_cols]
            ax.bar(x - w/2, az_vals, w, label=f'AZ MCTS (n={N_PLAYOUT})', color='#4C72B0', alpha=.85)
            ax.bar(x + w/2, pure_vals, w, label=f'Pure MCTS (c={default_c}, n={max_n})', color='#DD8452', alpha=.85)
            ax.set_xticks(x)
            ax.set_xticklabels([f'col{c}' for c in all_cols])
            ax.set_ylabel('Visit %')
            ax.set_title(desc, fontsize=9)
            ax.legend(fontsize=8)
        fig2.suptitle('AlphaZero MCTS vs Pure MCTS (no NN)', fontweight='bold')
        plt.tight_layout()
        p = os.path.join(out, 'pure_mcts_comparison.png')
        fig2.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig2)
        console.print(f'  [dim][saved] {p}[/dim]')


# ═══════════════════════ main ═══════════════════════

def main():
    parser = argparse.ArgumentParser(description='MCTS Root Node Diagnostics')
    parser.add_argument('--model', default='params/AZ_Connect4_CNN_current.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default=OUTPUT_DIR)
    parser.add_argument('--n-runs', type=int, default=N_RUNS)
    parser.add_argument('--quick', action='store_true', help='Fewer runs for faster results')
    parser.add_argument('--pure-only', action='store_true', help='Only run Section 7 (pure MCTS)')
    parser.add_argument('--hp-only', action='store_true', help='Only run Section 8 (AZ hyperparam sweep)')
    args = parser.parse_args()

    selective = args.pure_only or args.hp_only
    n_runs = args.n_runs
    sim_budgets = SIM_BUDGETS
    n_runs_sweep = N_RUNS_SWEEP
    n_runs_hp = N_RUNS_HP
    pure_hp_n = 50000
    if args.quick:
        n_runs = min(n_runs, 10)
        sim_budgets = [100, 400, 1600]
        n_runs_sweep = 5
        n_runs_hp = 5
        pure_hp_n = 1000

    os.chdir(ROOT)
    os.makedirs(args.output, exist_ok=True)
    _setup_font()

    console.print(Panel('[bold]MCTS Root Node Diagnostics[/bold]', border_style='blue'))
    net = load_model(args.model, args.device)
    console.print(f'  Model: {args.model}')
    console.print(f'  Params: c_init={C_INIT} c_base={C_BASE} fpu={FPU_REDUCTION} '
                  f'alpha={ALPHA} eps={NOISE_EPS} lambda_s={LAMBDA_S}')

    positions = make_positions()

    pure_mcts_budgets = PURE_MCTS_BUDGETS
    pure_mcts_c_inits = PURE_MCTS_C_INITS
    az_c_inits = AZ_C_INITS
    az_alphas  = AZ_ALPHAS
    az_fpus    = AZ_FPUS
    if args.quick:
        pure_mcts_budgets = [1000, 5000]
        pure_mcts_c_inits = [0.3, 1, 4, 16]
        az_c_inits = [0.3, 0.7, 1.5, 3.0, 5.0]
        az_alphas  = [0.03, 0.1, 0.3, 0.7]
        az_fpus    = [0.0, 0.2, 0.5, 0.9]

    all_child, all_agg, all_sweep, all_noise = [], [], [], []
    all_pure, all_hp = [], []
    descs, sweep_descs, noise_descs, pure_descs, hp_descs = [], [], [], [], []

    for desc, env, ply in positions:
        console.rule(f'[bold]{desc}[/bold]')
        env.show()
        descs.append(desc)

        if not selective:
            all_child.append(section1(net, env, desc))
            section2(net, env, desc)
            all_agg.append(section3(net, env, desc, n_runs))

            # Sweep & noise ablation only for the first 2 key positions
            if ply <= 2:
                all_sweep.append(section4(net, env, desc, sim_budgets, n_runs_sweep))
                sweep_descs.append(desc)
                all_noise.append(section5(net, env, desc, n_runs))
                noise_descs.append(desc)

        # Pure MCTS for the first 2 key positions
        if ply <= 2 and not args.hp_only:
            all_pure.append(section7(env, desc, pure_mcts_budgets, pure_mcts_c_inits))
            pure_descs.append(desc)

        # AZ MCTS hyperparam sweep for the first 2 key positions
        if ply <= 2 and not args.pure_only:
            all_hp.append(section8(env, desc, az_c_inits, az_alphas, az_fpus, n_runs_hp, pure_hp_n))
            hp_descs.append(desc)

        console.print()

    # ── Plots ──
    console.rule('[bold]Generating plots[/bold]')
    if not selective:
        plot_child_values(all_child, descs, args.output)
        plot_q_vs_visits(all_agg, descs, args.output)
        if all_sweep:
            plot_sim_sweep(all_sweep, sweep_descs, sim_budgets, args.output)
        if all_noise:
            plot_noise_ablation(all_noise, noise_descs, args.output)
        plot_multi_ply(all_agg, descs, args.output)
    if all_pure:
        plot_pure_mcts(all_pure, all_agg[:len(all_pure)] if all_agg else [],
                       pure_descs, pure_mcts_budgets, pure_mcts_c_inits, args.output)
    if all_hp:
        plot_hp_sweep(all_hp, hp_descs, args.output)

    console.print(Panel(
        f'[bold green]Done! Figures saved to: {args.output}/[/bold green]',
        border_style='green'))


if __name__ == '__main__':
    main()
