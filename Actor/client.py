import torch
import requests
import pickle
from game import Game
from copy import deepcopy
from environments import load
from player import AlphaZeroPlayer
from tqdm.auto import trange
import argparse
import signal


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

args = parser.parse_args()


host = args.host
port = args.port
headers = {'Content-Type': 'application/octet-stream'}


class TrainPipeline:
    def __init__(self, env_name='Connect4', model='CNN', name='AZ', play_batch_size=1):
        collection = ('Connect4', )  # NBTTT implementation not yet finished.
        if env_name not in collection:
            raise ValueError(f'Environment does not exist, available env: {collection}')
        self.env_name = env_name
        self.module = load(env_name)
        self.env = self.module.Env()
        self.game = Game(self.env)
        self.name = f'{name}_{env_name}'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.global_step = 0
        self.play_batch_size = play_batch_size
        for key, value in self.module.training_config.items():
            setattr(self, key, value)
        self.buffer = None
        if model == 'CNN':
            self.net = self.module.CNN(lr=0, device=self.device)
        elif model == 'ViT':
            self.net = self.module.ViT(lr=0, device=self.device)
        else:
            raise ValueError(f'Unknown model type: {model}')
        self.az_player = AlphaZeroPlayer(self.net, c_puct=self.c_puct,
                                         n_playout=args.n, alpha=self.dirichlet_alpha, is_selfplay=1)
        self.mtime = 0
        
    def load_weights(self):
        r = requests.get(f'http://{host}:{port}/weights?ts={self.mtime}')
        if r.status_code == 200:
            weights = pickle.loads(r.content)
            self.net.to('cpu')
            self.net.load_state_dict(weights)
            self.net.to(self.device)
            self.mtime = float(r.headers['X-Timestamp'])

    @staticmethod
    def push_data(data):
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        resp = requests.post(f'http://{host}:{port}/upload', headers=headers, data=payload)
        
    def data_collector(self, n_games=1):
        self.load_weights()
        data = []
        for _ in range(n_games):
            _, play_data = self.game.start_self_play(self.az_player, temp=self.temp, first_n_steps=self.first_n_steps)
            play_data = list(play_data)
            data.append(play_data)
        self.push_data(data)


if __name__ == '__main__':
    print('Running...')
    pipeline = TrainPipeline()
    try:
        while running:
            pipeline.data_collector()
    except Exception as e:
        print(e)
    print('quit')
            