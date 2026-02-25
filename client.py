import torch
import requests
import pickle
from src.game import Game
from src.environments import load
from src.player import BatchedAlphaZeroPlayer
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
    description="AlphaZero Actor. MCTS/self-play parameters are fetched from server.")

parser.add_argument('--host', '-H', type=str, default='127.0.0.1', help='Server host IP')
parser.add_argument('--port', '-P', '-p', type=int, default=7718, help='Server port')
parser.add_argument('-d', '--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
parser.add_argument('-B', '--batch_size', type=int, default=100,
                    help='Number of parallel self-play games')
parser.add_argument('--cache_size', type=int, default=0,
                    help='Transposition table size (0=disabled)')
parser.add_argument('--retry', type=int, default=3, help='Max connection retries')
parser.add_argument('--actor', type=str, default='best',
                    help='Which weight to load (best/current)')

args = parser.parse_args()

headers = {'Content-Type': 'application/octet-stream'}


class Actor:
    def __init__(self):
        self.server_url = f'http://{args.host}:{args.port}'
        self.cfg = self.fetch_config()

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

        self.az_player = BatchedAlphaZeroPlayer(
            self.net,
            n_envs=self.batch_size,
            c_init=self.cfg['c_init'],
            c_base=self.cfg['c_base'],
            n_playout=self.cfg['n_playout'],
            alpha=self.cfg['dirichlet_alpha'],
            cache_size=args.cache_size,
            noise_epsilon=self.cfg['noise_eps'],
            fpu_reduction=self.cfg['fpu_reduction'],
            use_symmetry=self.cfg['use_symmetry'],
            noise_steps=self.cfg['noise_steps'],
            noise_eps_min=self.cfg['noise_eps_min'],
            mlh_slope=self.cfg['mlh_slope'],
            mlh_cap=self.cfg['mlh_cap'])
        self.mtime = 0

    def fetch_config(self):
        """从 server 获取 MCTS / self-play 参数。"""
        while True:
            try:
                r = requests.get(f'{self.server_url}/config', timeout=5)
                if r.status_code == 200:
                    cfg = r.json()
                    print('Config from server:')
                    for k, v in cfg.items():
                        print(f'  {k}: {v}')
                    return cfg
            except requests.exceptions.ConnectionError:
                pass
            print('Waiting for server...')
            time.sleep(2)

    def load_weights(self):
        r = requests.get(f'{self.server_url}/weights?ts={self.mtime}&actor={args.actor}')
        if r.status_code == 200:
            if self.mtime == 0:
                print('Server Connected, start collecting.')

            download_size = len(r.content)
            print(f"[[TRAFFIC_LOG::DOWNLOAD::+::{download_size}]]")

            weights = pickle.loads(r.content)
            self.net.to('cpu')
            self.net.load_state_dict(weights)
            self.net.to(args.device)
            self.mtime = float(r.headers['X-Timestamp'])
            self.az_player.mcts.refresh_cache(self.net)

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
            temp_thres = self.cfg['temp_thres']
            results = self.game.batch_self_play(self.az_player, self.batch_size, temp, temp_thres)
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
