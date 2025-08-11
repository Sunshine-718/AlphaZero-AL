import os
import torch
import numpy as np
import requests
import pickle
from game import Game
from copy import deepcopy
from environments import load
from player import AlphaZeroPlayer
from tqdm.auto import trange


host = '127.0.0.1'
port = 9999
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
            self.net = self.module.CNN(lr=self.lr, device=self.device)
        elif model == 'ViT':
            self.net = self.module.ViT(lr=self.lr, device=self.device)
        else:
            raise ValueError(f'Unknown model type: {model}')
        self.az_player = AlphaZeroPlayer(self.net, c_puct=self.c_puct,
                                         n_playout=self.n_playout, alpha=self.dirichlet_alpha, is_selfplay=1)
        self.mtime = 0
    
    def data_collector(self, n_games=1):
        r = requests.get(f'http://{host}:{port}/weights?ts={self.mtime}')
        if r.status_code == 200:
            weights = pickle.loads(r.content)
            self.net.to('cpu')
            self.net.load(weights)
            self.net.to(self.device)
            self.mtime = float(r.headers['X-Timestamp'])
            print(f"权重已更新")
        elif r.status_code == 304:
            print("权重未更新")
        data = []
        for _ in trange(n_games):
            _, play_data = self.game.start_self_play(self.az_player, temp=self.temp, first_n_steps=self.first_n_steps)
            play_data = list(play_data)
            data.append(play_data)
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        resp = requests.post(f'http://{host}:{port}/upload', headers=headers, data=payload)
        print(resp.json())


if __name__ == '__main__':
    pipeline = TrainPipeline()
    while True:
        pipeline.data_collector()
            