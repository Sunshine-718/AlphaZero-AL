import os
import numpy as np
import torch
from abc import ABC, abstractmethod
from .utils import Elo
from .game import Game
from copy import deepcopy
from .environments import load
from .player import MCTSPlayer, AlphaZeroPlayer, NetworkPlayer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader
import swanlab


class TrainPipeline(ABC):
    def __init__(self, env_name='Connect4', model='CNN', name='AZ', config=None, rank=0, world_size=1):
        collection = ('Connect4', )
        if env_name not in collection:
            raise ValueError(f'Environment does not exist, available env: {collection}')
        self.env_name = env_name
        self.rank = rank
        self.world_size = world_size
        self.module = load(env_name)
        self.env = self.module.Env()
        self.game = Game(self.env)
        self.name = f'{name}_{env_name}'
        self.params = './params'
        self.global_step = 0
        self.raw_config = config if config else {}
        for key, value in config.items():
            setattr(self, key, value)
        self.buffer = None
        if model == 'CNN':
            self.net = self.module.CNN(lr=self.lr, device=self.device)
        elif model == 'ViT':
            self.net = self.module.ViT(lr=self.lr, device=self.device)
        else:
            raise ValueError(f'Unknown model type: {model}')
        self.current = f'{self.params}/{self.name}_{self.net.name()}_current.pt'
        self.best = f'{self.params}/{self.name}_{self.net.name()}_best.pt'
        self.net.load(self.current)
        self.ddp_net = DDP(self.net, device_ids=[rank], find_unused_parameters=False) if world_size > 1 else self.net
        self.az_player = AlphaZeroPlayer(self.net, c_init=self.c_puct, n_playout=self.n_playout,
                                         discount=self.discount, alpha=self.dirichlet_alpha, is_selfplay=1,
                                         cache_size=self.cache_size)
        self.update_best_net()
        self.elo = Elo(self.init_elo, 1500)
        os.makedirs('params', exist_ok=True)

    def init_buffer(self, buffer):
        self.buffer = buffer

    def _on_weights_saved(self):
        pass

    @abstractmethod
    def data_collector(self):
        ...

    def policy_update(self):
        if self.world_size > 1:
            # rank 0 采样 batch_size * world_size 条，每张卡各得 batch_size 条（标准数据切分并行）
            total_samples = self.batch_size * self.world_size
            if self.rank == 0:
                dataloader = self.buffer.sample(total_samples)
                tensors = [torch.cat([batch[i] for batch in dataloader], dim=0).to(self.device).float()
                           for i in range(7)]
                n = torch.tensor([tensors[0].shape[0]], dtype=torch.long, device=self.device)
            else:
                n = torch.tensor([0], dtype=torch.long, device=self.device)

            dist.broadcast(n, src=0)
            n_samples = n.item()

            rows, cols = self.env.board.shape
            shapes = [(n_samples, self.net.in_dim, rows, cols), (n_samples, 1),
                      (n_samples, self.net.n_actions), (n_samples, 1), (n_samples, 1),
                      (n_samples, self.net.in_dim, rows, cols), (n_samples, 1)]
            orig_dtypes = [torch.float32, torch.int16, torch.float32, torch.float32,
                           torch.int8, torch.float32, torch.bool]

            broadcast_tensors = []
            for i, (shape, orig_dtype) in enumerate(zip(shapes, orig_dtypes)):
                buf = tensors[i] if self.rank == 0 else torch.empty(shape, dtype=torch.float32, device=self.device)
                dist.broadcast(buf, src=0)
                # 每张卡切取属于自己的那份 batch_size 条数据
                per_rank = n_samples // self.world_size
                shard = buf[self.rank * per_rank: (self.rank + 1) * per_rank]
                broadcast_tensors.append(shard.to(dtype=orig_dtype))

            dataloader = DataLoader(TensorDataset(*broadcast_tensors), self.batch_size, shuffle=True)
        else:
            dataloader = self.buffer.sample(self.batch_size)

        p_l, v_l, const_loss, ent, g_n, f1 = self.net.train_step(dataloader, self.module.augment,
                                                                   ddp_model=self.ddp_net)
        print(f'F1 score (new): {f1: .3f}')
        return p_l, v_l, const_loss, ent, g_n, f1

    def update_elo(self):
        print('Updating elo score...')
        self.net.eval()
        az = deepcopy(self.az_player)
        az.is_selfplay = False
        az.eval()
        mcts = MCTSPlayer(1, self.pure_mcts_n_playout, self.discount)

        w1 = self.game.play(player1=az, player2=mcts, show=0)
        self.elo.update(1 if w1 == 1 else 0.5 if w1 == 0 else 0)
        w2 = self.game.play(player1=mcts, player2=az, show=0)
        print('Complete.')
        return self.elo.update(1 if w2 == -1 else 0.5 if w2 == 0 else 0)

    def select_best_player(self, n_games=10):
        print('Evaluating best player...')
        self.net.eval()
        self.best_net.eval()
        current = NetworkPlayer(self.net, False)
        best = NetworkPlayer(self.best_net, False)
        current.eval()
        best.eval()

        win_rate = 0
        # Play half the games as each color to remove first-move bias
        for p1, p2, win_val in [(current, best, 1), (best, current, -1)]:
            for _ in range(n_games // 2):
                winner = self.game.play(player1=p1, player2=p2, show=0)
                if winner == win_val:
                    win_rate += 1 / n_games
                elif winner == 0:
                    win_rate += 0.5 / n_games

        flag = win_rate >= self.win_rate_threshold
        if flag:
            self.update_best_net()
        print('Complete.')
        return flag, win_rate

    def update_best_net(self):
        self.best_net = deepcopy(self.net)

    def _log_train_step(self, p_loss, v_loss, const_loss, entropy, grad_norm, f1):
        if self.episode_len is not None:
            swanlab.log({'Metric/Episode length': self.episode_len}, step=self.global_step)
        swanlab.log({
            'Metric/lr': self.net.opt.param_groups[0]['lr'],
            'Metric/Gradient Norm': grad_norm,
            'Metric/F1 score': f1,
            'Metric/Loss/Action Loss': p_loss,
            'Metric/Loss/Value loss': v_loss,
            'Metric/Loss/Consistency loss': const_loss,
            'Metric/Entropy': entropy,
        }, step=self.global_step)

    def _log_eval(self, r_a, r_b, win_rate, best_counter):
        swanlab.log({
            f'Metric/Elo/AlphaZero_{self.n_playout}': r_a,
            f'Metric/Elo/MCTS_{self.pure_mcts_n_playout}': r_b,
            'Metric/win rate': win_rate,
        }, step=self.global_step)
        if best_counter is not None:
            swanlab.log({'Metric/Best policy': best_counter}, step=self.global_step)

        if self.env_name == 'Connect4':
            p0, v0, p1, v1 = self.module.inspect(self.net)
            log_dict = {"Metric/initial value/X": v0, "Metric/initial value/O": v1}
            for idx, prob in enumerate(p0):
                log_dict[f'Action probability/X/{idx}'] = prob
            for idx, prob in enumerate(p1):
                log_dict[f'Action probability/O/{idx}'] = prob
            for idx, prob in enumerate(np.cumsum(p0)):
                log_dict[f'Action probability/X_cummulative/{idx}'] = prob
            for idx, prob in enumerate(np.cumsum(p1)):
                log_dict[f'Action probability/O_cummulative/{idx}'] = prob
            swanlab.log(log_dict, step=self.global_step)

    def run(self):
        if self.rank == 0:
            print('=' * 50)
            print(f'Hyperparameters:\n'
                  f'\tC_puct: {self.c_puct}\n'
                  f'\tSimulation (AlphaZero): {self.n_playout}\n'
                  f'\tSimulation (Benchmark): {self.pure_mcts_n_playout}\n'
                  f'\tDiscount: {self.discount}\n'
                  f'\tDirichlet alpha: {self.dirichlet_alpha}\n'
                  f'\tBuffer size: {self.buffer_size}\n'
                  f'\tBatch size: {self.batch_size}')
            print('=' * 50)

            run_config = {'env_name': self.env_name, 'model': self.net.name()}
            run_config.update(self.raw_config)
            swanlab.init(project="AlphaZero-AL", experiment_name=self.name, config=run_config)
            self.buffer.load('./dataset/dataset.pt')

        best_counter = 0

        while True:
            if self.rank == 0:
                self.data_collector()

            if self.world_size > 1:
                dist.barrier()

            self.global_step += 1
            p_loss, v_loss, const_loss, entropy, grad_norm, f1 = self.policy_update()

            if self.world_size > 1:
                dist.barrier()

            if self.rank == 0:
                self.net.save(self.current)
                self._on_weights_saved()
                print(f'batch i: {self.global_step}, episode_len: {self.episode_len}, '
                      f'loss: {p_loss + v_loss: .8f}, entropy: {entropy: .8f}')
                self._log_train_step(p_loss, v_loss, const_loss, entropy, grad_norm, f1)

            if self.global_step % self.interval != 0:
                continue

            if self.world_size > 1:
                dist.barrier()

            if self.rank == 0:
                print(f'current self-play batch: {self.global_step + 1}')
                r_a, r_b = self.update_elo()
                print(f'Elo score: AlphaZero: {r_a: .2f}, Benchmark: {r_b: .2f}')

                flag, win_rate = self.select_best_player(self.num_eval)
                new_best = best_counter + 1 if flag else None
                self._log_eval(r_a, r_b, win_rate, new_best)

                if flag:
                    print('New best policy!!')
                    best_counter += 1
                    self.net.save(self.best)
                    os.makedirs('dataset', exist_ok=True)
                    self.buffer.save('./dataset/dataset.pt')

    def __call__(self):
        self.run()
