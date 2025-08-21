import torch
import requests
import pickle
from src.game import Game
from src.environments import load
from src.player import AlphaZeroPlayer
import argparse
import signal
import time


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
parser.add_argument('-c', '--c_init', type=float, default=1.25, help='C_puct init')
parser.add_argument('-a', '--alpha', type=float, default=0.7, help='Dirichlet alpha')
parser.add_argument('--n_play', type=int, default=1, help='n_playout')
parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
parser.add_argument('-t', '--temp', '--temperature', type=float, default=1, help='Softmax temperature')
parser.add_argument('--n_step', type=int, default=10, help='N steps to decay temperature')
parser.add_argument('-m', '--model', type=str, default='CNN', help='Model type (CNN)')
parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device type')
parser.add_argument('-e', '--env', '--environment', type=str, default='Connect4', help='Environment name')
parser.add_argument('--retry', type=int, default=3, help='Retry times')
parser.add_argument('--no-cache', action='store_false', dest='cache', help='Disable cache')
parser.add_argument('--cache_size', type=int, default=5000, help='LRU cache max size')

args = parser.parse_args()

headers = {'Content-Type': 'application/octet-stream'}


class Actor:
    def __init__(self, env_name=args.env):
        collection = ('Connect4', )  # NBTTT implementation not yet finished.
        if env_name not in collection:
            raise ValueError(f'Environment does not exist, available env: {collection}')
        self.env_name = env_name
        self.module = load(env_name)
        self.env = self.module.Env()
        self.game = Game(self.env)
        if args.model == 'CNN':
            self.net = self.module.CNN(lr=0, device=args.device)
        elif args.model == 'ViT':
            self.net = self.module.ViT(lr=0, device=args.device)
        else:
            raise ValueError(f'Unknown model type: {args.model}')
        self.az_player = AlphaZeroPlayer(self.net, c_init=args.c_init, n_playout=args.n, 
                                         discount=args.discount, alpha=args.alpha, is_selfplay=1, 
                                         use_cache=args.cache, cache_size=args.cache_size)
        self.mtime = 0

    def load_weights(self):
        r = requests.get(f'http://{args.host}:{args.port}/weights?ts={self.mtime}')
        if r.status_code == 200:
            if self.mtime == 0:
                print('Server Connected, start collecting.')
            weights = pickle.loads(r.content)
            self.net.to('cpu')
            self.net.load_state_dict(weights)
            self.net.to(args.device)
            self.az_player.reload(self.net)
            self.mtime = float(r.headers['X-Timestamp'])

    @staticmethod
    def push_data(data):
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        resp = requests.post(f'http://{args.host}:{args.port}/upload', headers=headers, data=payload)

    def data_collector(self, n_games=args.n_play):
        self.load_weights()
        data = []
        for _ in range(n_games):
            _, play_data = self.game.start_self_play(self.az_player, temp=args.temp, first_n_steps=args.n_step)
            play_data = list(play_data)
            assert(len(play_data) <= 42)    # Only for Connect4
            data.append(play_data)
        self.push_data(data)


if __name__ == '__main__':
    pipeline = Actor()
    for i in range(args.retry):
        try:
            while running:
                pipeline.data_collector()
        except requests.exceptions.ConnectionError:
            print(f'Server connection lost, retry{i + 1}')
            time.sleep(1)
            continue
    print('quit')
