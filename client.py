import torch
import requests
import pickle
from src.game import Game
from src.environments import load
from src.player import AlphaZeroPlayer
import argparse
import signal
import time
import warnings
import traceback

warnings.filterwarnings('error', category=RuntimeWarning)


running = True


def _stop(*_):
    global running
    running = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

parser = argparse.ArgumentParser(
    description="AlphaZero Actor. Parameters default to server values; "
                "explicitly set args override server and won't be synced.")

parser.add_argument('--host', '-H', type=str, default='127.0.0.1', help='Server host IP')
parser.add_argument('--port', '-P', '-p', type=int, default=7718, help='Server port')
parser.add_argument('-d', '--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
parser.add_argument('-B', '--batch_size', type=int, default=100,
                    help='Number of parallel self-play games')
parser.add_argument('--cache_size', type=int, default=0,
                    help='Transposition table size (0=disabled)')
parser.add_argument('--retry', type=int, default=3, help='Max connection retries')
parser.add_argument('--actor', type=str, default='current',
                    help='Which weight to load (best/current)')

# ── Overridable params (default=None → use server value) ─────────────────────
parser.add_argument('-n', type=int, default=None, help='MCTS simulations per move')
parser.add_argument('-c', '--c_init', type=float, default=None, help='PUCT exploration constant')
parser.add_argument('-a', '--alpha', type=float, default=None, help='Dirichlet noise alpha')
parser.add_argument('--eps', type=float, default=None, help='Noise mixing epsilon')
parser.add_argument('--noise_steps', type=int, default=None, help='Steps to decay noise eps')
parser.add_argument('--noise_eps_min', type=float, default=None, help='Minimum noise epsilon')
parser.add_argument('--fpu_reduction', type=float, default=None, help='First-play urgency reduction')
parser.add_argument('--vl_batch', type=int, default=None, help='Virtual loss batch size')
parser.add_argument('-t', '--temp', type=float, default=None, help='Self-play temperature')
parser.add_argument('--temp_decay_moves', type=int, default=None, help='Temp decay moves')
parser.add_argument('--temp_endgame', type=float, default=None, help='Temp floor')
parser.add_argument('--mlh_slope', type=float, default=None, help='MLH slope')
parser.add_argument('--mlh_cap', type=float, default=None, help='MLH cap')
parser.add_argument('--score_utility_factor', type=float, default=None, help='Score utility weight')
parser.add_argument('--score_scale', type=float, default=None, help='Score atan scale')
parser.add_argument('--value_decay', type=float, default=None, help='Value decay γ')
parser.add_argument('--td_steps', type=int, default=None, help='Future-root-WDL consistency: steps k for S_{t+k}')
parser.add_argument('--no_symmetry', action='store_true', help='Disable symmetry augmentation')
parser.add_argument('--compile', action='store_true',
                    help='Enable torch.compile for inference acceleration (requires PyTorch 2.0+)')
parser.add_argument('--config', action='store_true', help='Display current config and exit')

args = parser.parse_args()

# 映射: CLI arg name → server config key
_ARG_TO_CFG = {
    'n': 'n_playout', 'c_init': 'c_init', 'alpha': 'dirichlet_alpha',
    'eps': 'noise_eps', 'noise_steps': 'noise_steps', 'noise_eps_min': 'noise_eps_min',
    'fpu_reduction': 'fpu_reduction', 'vl_batch': 'vl_batch',
    'temp': 'temp', 'temp_decay_moves': 'temp_decay_moves', 'temp_endgame': 'temp_endgame',
    'mlh_slope': 'mlh_slope', 'mlh_cap': 'mlh_cap',
    'score_utility_factor': 'score_utility_factor', 'score_scale': 'score_scale',
    'value_decay': 'value_decay',
    'td_steps': 'td_steps',
}

# 记录用户手动设置的参数（不会被 server 同步覆盖）
_USER_OVERRIDES = set()
for arg_name, cfg_key in _ARG_TO_CFG.items():
    if getattr(args, arg_name) is not None:
        _USER_OVERRIDES.add(cfg_key)
if args.no_symmetry:
    _USER_OVERRIDES.add('use_symmetry')


def print_config(cfg):
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    groups = [
        ("Connection", {
            "host": args.host,
            "port": args.port,
            "device": args.device,
            "batch_size": args.batch_size,
            "cache_size": args.cache_size,
            "actor": args.actor,
        }),
        ("MCTS Search", {
            "n_playout": cfg['n_playout'],
            "c_init": cfg['c_init'],
            "c_base": cfg['c_base'],
            "fpu_reduction": cfg['fpu_reduction'],
            "vl_batch": cfg.get('vl_batch', 1),
            "use_symmetry": cfg['use_symmetry'],
            "value_decay": cfg.get('value_decay', 1.0),
        }),
        ("Exploration Noise", {
            "dirichlet_alpha": cfg['dirichlet_alpha'],
            "noise_eps": cfg['noise_eps'],
            "noise_steps": cfg['noise_steps'],
            "noise_eps_min": cfg['noise_eps_min'],
        }),
        ("Auxiliary Utility", {
            "mlh_slope": cfg['mlh_slope'],
            "mlh_cap": cfg['mlh_cap'],
            "score_utility_factor": cfg.get('score_utility_factor', 0.0),
            "score_scale": cfg.get('score_scale', 8.0),
        }),
        ("Self-play", {
            "temp": cfg['temp'],
            "temp_decay_moves": cfg['temp_decay_moves'],
            "temp_endgame": cfg['temp_endgame'],
            "td_steps": cfg.get('td_steps', 0),
        }),
    ]

    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("Group", style="bold yellow", min_width=12)
    table.add_column("Parameter", min_width=20)
    table.add_column("Value", justify="right", style="bold green", min_width=10)
    table.add_column("Source", justify="center", min_width=8)

    for i, (group_name, params) in enumerate(groups):
        if i > 0:
            table.add_section()
        first = True
        for key, val in params.items():
            source = "[bold magenta]CLI[/bold magenta]" if key in _USER_OVERRIDES else "server"
            table.add_row(group_name if first else "", key, str(val), source)
            first = False

    console.print()
    console.print(Panel(table,
                        title=f"[bold]AlphaZero Actor Config[/bold]",
                        subtitle=f"[dim]{cfg['env']} / {cfg['model']}[/dim]",
                        border_style="blue"))
    console.print()


headers = {'Content-Type': 'application/octet-stream'}


class Actor:
    def __init__(self):
        self.server_url = f'http://{args.host}:{args.port}'
        self.cfg = self._fetch_server_config()
        self._apply_user_overrides()

        print_config(self.cfg)
        if args.config:
            import sys
            sys.exit(0)

        env_name = self.cfg['env']
        model_name = self.cfg['model']
        self.module = load(env_name)
        self.env = self.module.Env()
        self.game = Game(self.env)
        self.batch_size = args.batch_size

        if model_name == 'CNN':
            self.net = self.module.CNN(lr=0, device=args.device)
        elif model_name == 'ViT':
            self.net = self.module.ViT(lr=0, device=args.device)
        else:
            raise ValueError(f'Unknown model type: {model_name}')
        if args.compile:
            print('torch.compile enabled — compiling model (first iteration will be slow)...')
            self.net = torch.compile(self.net, mode='reduce-overhead')
        self.net.eval()

        self.az_player = AlphaZeroPlayer(
            self.net,
            n_envs=self.batch_size,
            c_init=self.cfg['c_init'],
            c_base=self.cfg['c_base'],
            n_playout=self.cfg['n_playout'],
            alpha=self.cfg['dirichlet_alpha'],
            is_selfplay=1,
            cache_size=args.cache_size,
            noise_epsilon=self.cfg['noise_eps'],
            fpu_reduction=self.cfg['fpu_reduction'],
            use_symmetry=self.cfg['use_symmetry'],
            game_name=env_name,
            noise_steps=self.cfg['noise_steps'],
            noise_eps_min=self.cfg['noise_eps_min'],
            mlh_slope=self.cfg['mlh_slope'],
            mlh_cap=self.cfg['mlh_cap'],
            score_utility_factor=self.cfg.get('score_utility_factor', 0.0),
            score_scale=self.cfg.get('score_scale', 8.0),
            vl_batch=self.cfg.get('vl_batch', 1),
            value_decay=self.cfg.get('value_decay', 1.0))
        self.mtime = 0

    def _fetch_server_config(self):
        """从 server 获取 MCTS / self-play 参数。"""
        while True:
            try:
                r = requests.get(f'{self.server_url}/config', timeout=5)
                if r.status_code == 200:
                    return r.json()
            except requests.exceptions.ConnectionError:
                pass
            print('Waiting for server...')
            time.sleep(2)

    def _apply_user_overrides(self):
        """用户手动设置的 CLI 参数覆盖 server 值。"""
        for arg_name, cfg_key in _ARG_TO_CFG.items():
            val = getattr(args, arg_name)
            if val is not None:
                self.cfg[cfg_key] = val
        if args.no_symmetry:
            self.cfg['use_symmetry'] = False
        if _USER_OVERRIDES:
            print(f'User overrides: { {k: self.cfg[k] for k in _USER_OVERRIDES} }')

    def _sync_config(self):
        """从 server 拉取最新参数，同步用户未手动设置的参数到 MCTS。"""
        try:
            r = requests.get(f'{self.server_url}/config', timeout=5)
            if r.status_code != 200:
                return
        except requests.exceptions.RequestException:
            return

        server_cfg = r.json()
        updated = {}

        # 逐个检查可同步的参数
        _SYNC_MAP = {
            'c_init':         lambda v: self.az_player.mcts.set_c_init(v),
            'c_base':         lambda v: self.az_player.mcts.set_c_base(v),
            'n_playout':      lambda v: setattr(self.az_player.mcts, 'n_playout', v),
            'dirichlet_alpha': lambda v: self.az_player.mcts.set_alpha(v),
            'noise_eps':      lambda v: self.az_player.mcts.set_noise_epsilon(v),
            'noise_steps':    lambda v: setattr(self.az_player, 'noise_steps', v),
            'noise_eps_min':  lambda v: setattr(self.az_player, 'noise_eps_min', v),
            'fpu_reduction':  lambda v: self.az_player.mcts.set_fpu_reduction(v),
            'vl_batch':       lambda v: setattr(self.az_player, '_vl_batch', v),
            'use_symmetry':   lambda v: self.az_player.mcts.set_use_symmetry(v),
            'value_decay':    lambda v: self.az_player.mcts.set_value_decay(v),
            'td_steps':       None,  # 每轮 data_collector 从 self.cfg 读
            'temp':           None,  # 每轮 data_collector 从 self.cfg 读
            'temp_decay_moves': None,
            'temp_endgame':   None,
        }

        for cfg_key, apply_fn in _SYNC_MAP.items():
            if cfg_key in _USER_OVERRIDES:
                continue
            new_val = server_cfg.get(cfg_key)
            if new_val is not None and new_val != self.cfg.get(cfg_key):
                self.cfg[cfg_key] = new_val
                if apply_fn is not None:
                    apply_fn(new_val)
                updated[cfg_key] = new_val

        # MLH 参数一起更新
        if not ({'mlh_slope', 'mlh_cap'} & _USER_OVERRIDES):
            new_slope = server_cfg.get('mlh_slope')
            new_cap = server_cfg.get('mlh_cap')
            if (new_slope is not None and
                    (new_slope != self.cfg.get('mlh_slope') or
                     new_cap != self.cfg.get('mlh_cap'))):
                self.cfg['mlh_slope'] = new_slope
                self.cfg['mlh_cap'] = new_cap
                self.az_player.mcts.set_mlh_params(new_slope, new_cap)
                updated['mlh'] = (new_slope, new_cap)

        # Score utility 参数一起更新
        if not ({'score_utility_factor', 'score_scale'} & _USER_OVERRIDES):
            new_factor = server_cfg.get('score_utility_factor')
            new_scale = server_cfg.get('score_scale')
            if (new_factor is not None and
                    (new_factor != self.cfg.get('score_utility_factor') or
                     new_scale != self.cfg.get('score_scale'))):
                self.cfg['score_utility_factor'] = new_factor
                self.cfg['score_scale'] = new_scale
                self.az_player.mcts.set_score_utility_params(new_factor, new_scale)
                updated['score_utility'] = (new_factor, new_scale)

        if updated:
            print(f'[SYNC] Updated from server: {updated}')

    def load_weights(self):
        r = requests.get(f'{self.server_url}/weights?ts={self.mtime}&actor={args.actor}')
        if r.status_code == 200:
            if self.mtime == 0:
                print('Server Connected, start collecting.')

            download_size = len(r.content)
            print(f"[[TRAFFIC_LOG::DOWNLOAD::+::{download_size}]]")

            weights = pickle.loads(r.content)
            self.net.to('cpu')
            if args.compile and not any(k.startswith('_orig_mod.') for k in weights):
                weights = {f'_orig_mod.{k}': v for k, v in weights.items()}
            self.net.load_state_dict(weights, strict=True)
            self.net.to(args.device)
            self.net.eval()
            self.mtime = float(r.headers['X-Timestamp'])
            self.az_player.mcts.refresh_cache(self.net)
            self._sync_config()

    def push_data(self, data):
        payload = pickle.dumps({'__az__': True, 'data': data}, protocol=pickle.HIGHEST_PROTOCOL)
        resp = requests.post(f'{self.server_url}/upload', headers=headers, data=payload)

        if resp.status_code == 200:
            upload_size = len(payload)
            print(f"[[TRAFFIC_LOG::UPLOAD::+::{upload_size}]]")

    def data_collector(self):
        self.load_weights()
        data = []
        try:
            start_time = time.time()
            temp = self.cfg['temp']
            temp_decay_moves = self.cfg['temp_decay_moves']
            temp_endgame = self.cfg['temp_endgame']
            td_steps = self.cfg.get('td_steps', 0)
            results = self.game.batch_self_play(self.az_player, self.batch_size,
                                                temp, temp_decay_moves, temp_endgame,
                                                td_steps=td_steps)
            for _, play_data in results:
                data.append(play_data)
            duration = time.time() - start_time
            print(f"Collected {len(data)} games in {duration:.2f}s (FPS: {len(data) / duration:.2f})")
            self.push_data(data)
        except RuntimeWarning as e:
            print(f"RuntimeWarning during self-play: {e}")


if __name__ == '__main__':
    pipeline = Actor()
    print(f"Client started on device: {args.device}")
    retry_count = 0
    while running:
        try:
            pipeline.data_collector()
            retry_count = 0
        except requests.exceptions.ConnectionError:
            retry_count += 1
            print(f'Server connection lost, retry {retry_count}/{args.retry}')
            if retry_count >= args.retry:
                print("Max retries reached. Waiting 10s...")
                time.sleep(10)
                retry_count = 0
            else:
                time.sleep(1)
            continue
        except Exception as e:
            traceback.print_exc()
            break

    print('Client quit')
