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

parser = argparse.ArgumentParser(description="AlphaZero Actor.")
parser.add_argument('-n', type=int, default=100,
                    help='Number of simulations before AlphaZero make an action')
parser.add_argument('--host', '-H', type=str, default='127.0.0.1', help='Host IP')
parser.add_argument('--port', '-P', '-p', type=int, default=7718, help='Port number')
parser.add_argument('-c', '--c_init', type=float, default=1, help='C_puct init')
parser.add_argument('--c_base_factor', type=float, default=1000, help='C_puct base factor')
parser.add_argument('--fpu_reduction', type=float, default=0.2, help='FPU reduction factor')
parser.add_argument('-a', '--alpha', type=float, default=1.55, help='Dirichlet alpha')
parser.add_argument('--noise_eps', type=float, default=0.25, help='Noise epsilon')
parser.add_argument('--discount', type=float, default=1, help='Discount factor')
parser.add_argument('-t', '--temp', type=float, default=1, help='Softmax temperature')
parser.add_argument('--temp_thres', type=float, default=12, help='Step threshold to change temperature to -> 0')
parser.add_argument('-m', '--model', type=str, default='CNN', help='Model type (CNN)')
parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available()
                    else 'cpu', help='Device type')
parser.add_argument('-e', '--env', '--environment', type=str, default='Connect4', help='Environment name')
parser.add_argument('--retry', type=int, default=3, help='Retry times')

parser.add_argument('-B', '--batch_size', type=int, default=100, help='Batch size for self-play')
parser.add_argument('--cache_size', type=int, default=0,
                    help='Transposition table size (0 = disabled, >0 = LRU cache capacity)')

args = parser.parse_args()

headers = {'Content-Type': 'application/octet-stream'}


class Actor:
    def __init__(self, env_name=args.env):
        collection = ('Connect4', )
        if env_name not in collection:
            raise ValueError(f'Environment does not exist, available env: {collection}')
        self.env_name = env_name
        self.module = load(env_name)
        self.env = self.module.Env()
        self.game = Game(self.env)
        self.batch_size = args.batch_size
        if args.model == 'CNN':
            self.net = self.module.CNN(lr=0, device=args.device)
        elif args.model == 'ViT':
            self.net = self.module.ViT(lr=0, device=args.device)
        else:
            raise ValueError(f'Unknown model type: {args.model}')
        self.az_player = BatchedAlphaZeroPlayer(self.net,
                                                n_envs=self.batch_size,
                                                c_init=args.c_init,
                                                c_base=args.n * args.c_base_factor,
                                                n_playout=args.n,
                                                discount=args.discount,
                                                alpha=args.alpha,
                                                cache_size=args.cache_size,
                                                noise_epsilon=args.noise_eps,
                                                fpu_reduction=args.fpu_reduction)
        self.mtime = 0

    def load_weights(self):
        r = requests.get(f'http://{args.host}:{args.port}/weights?ts={self.mtime}')
        if r.status_code == 200:
            if self.mtime == 0:
                print('Server Connected, start collecting.')

            # --- 新增: 打印特殊日志供 GUI 统计下载流量 ---
            download_size = len(r.content)
            print(f"[[TRAFFIC_LOG::DOWNLOAD::+::{download_size}]]")
            # ----------------------------------------

            weights = pickle.loads(r.content)
            self.net.to('cpu')
            self.net.load_state_dict(weights)
            self.net.to(args.device)
            self.mtime = float(r.headers['X-Timestamp'])
            self.az_player.mcts.refresh_cache(self.net)

    # 将 @staticmethod 改为实例方法，以便访问 self
    def push_data(self, data):
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        resp = requests.post(f'http://{args.host}:{args.port}/upload', headers=headers, data=payload)

        # --- 新增: 打印特殊日志供 GUI 统计上传流量 ---
        if resp.status_code == 200:
            upload_size = len(payload)
            print(f"[[TRAFFIC_LOG::UPLOAD::+::{upload_size}]]")
        # ----------------------------------------

    def data_collector(self):
        self.load_weights()
        data = []
        try:
            start_time = time.time()
            results = self.game.streaming_self_play(self.az_player, self.batch_size, args.temp, args.temp_thres)
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
            retry_count = 0  # Reset retry on success
        except requests.exceptions.ConnectionError:
            retry_count += 1
            print(f'Server connection lost, retry {retry_count}/{args.retry}')
            if retry_count >= args.retry:
                print("Max retries reached. Waiting 10s...")
                time.sleep(10)
                retry_count = 0  # Loop indefinitely but slow down
            else:
                time.sleep(1)
            continue
        except Exception as e:
            traceback.print_exc()
            break

    print('Client quit')
