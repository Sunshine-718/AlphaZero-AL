import os
import numpy as np
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from .utils import Elo
from .game import Game
from copy import deepcopy
from .environments import load
from .player import MCTSPlayer, AlphaZeroPlayer
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import swanlab


class TrainPipeline(ABC):
    def __init__(self, env_name='Connect4', model='CNN', name='AZ', config=None,
                 rank=0, world_size=1, local_rank=0):
        collection = ('Connect4', )
        if env_name not in collection:
            raise ValueError(f'Environment does not exist, available env: {collection}')
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_ddp = world_size > 1
        self.env_name = env_name
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
            self.net = self.module.CNN(lr=self.lr, device=self.device,
                                      lambda_s=getattr(self, 'lambda_s', 0.1),
                                      policy_lr_scale=getattr(self, 'policy_lr_scale', 0.3),
                                      dropout=getattr(self, 'dropout', 0.2))
        elif model == 'ViT':
            self.net = self.module.ViT(lr=self.lr, device=self.device,
                                       lambda_s=getattr(self, 'lambda_s', 0.1))
        else:
            raise ValueError(f'Unknown model type: {model}')
        self.current = f'{self.params}/{self.name}_{self.net.name()}_current.pt'
        self.best = f'{self.params}/{self.name}_{self.net.name()}_best.pt'
        self.net.load(self.current)

        if self.is_ddp:
            self.ddp_net = DDP(self.net, device_ids=[self.local_rank], find_unused_parameters=True)
        else:
            self.ddp_net = None

        self.az_player = AlphaZeroPlayer(self.net, c_init=self.c_puct, n_playout=self.n_playout,
                                         discount=self.discount, alpha=self.dirichlet_alpha, is_selfplay=1,
                                         cache_size=self.cache_size, eps=self.eps,
                                         use_symmetry=getattr(self, 'use_symmetry', True))
        self.update_best_net()
        self.elo = Elo(self.init_elo, 1500)
        self.r_a = 0
        self.r_b = 0
        os.makedirs('params', exist_ok=True)

    def init_buffer(self, buffer):
        self.buffer = buffer

    @abstractmethod
    def data_collector(self):
        ...

    def _broadcast_dataloader(self):
        """Rank 0 从 buffer 采样完整 dataset 并广播给所有 rank。
        各 rank 用各自的随机种子 shuffle，配合 per-GPU batch_size 实现数据并行。"""
        if self.rank == 0:
            dataloader = self.buffer.sample(self.batch_size * self.world_size)
            dataset = dataloader.dataset
            state = dataset.tensors[0].to(self.device).contiguous()
            prob = dataset.tensors[1].to(self.device).contiguous()
            winner = dataset.tensors[2].to(self.device).contiguous().float()
            steps_to_end = dataset.tensors[3].to(self.device).contiguous().float()
            meta = torch.tensor([state.shape[0], state.shape[1], state.shape[2],
                                 state.shape[3], prob.shape[1]],
                                dtype=torch.long, device=self.device)
        else:
            meta = torch.zeros(5, dtype=torch.long, device=self.device)

        dist.broadcast(meta, src=0)
        N, C, H, W, A = meta.tolist()
        N, C, H, W, A = int(N), int(C), int(H), int(W), int(A)

        if self.rank != 0:
            state = torch.empty((N, C, H, W), dtype=torch.float32, device=self.device)
            prob = torch.empty((N, A), dtype=torch.float32, device=self.device)
            winner = torch.empty((N, 1), dtype=torch.float32, device=self.device)
            steps_to_end = torch.empty((N, 1), dtype=torch.float32, device=self.device)

        dist.broadcast(state, src=0)
        dist.broadcast(prob, src=0)
        dist.broadcast(winner, src=0)
        dist.broadcast(steps_to_end, src=0)

        winner = winner.to(torch.int8)
        steps_to_end = steps_to_end.to(torch.int8)
        dataset = TensorDataset(state, prob, winner, steps_to_end)
        return DataLoader(dataset, self.batch_size, shuffle=True)

    def policy_update(self):
        if self.is_ddp:
            dist.barrier()
            dataloader = self._broadcast_dataloader()
        else:
            dataloader = self.buffer.sample(self.batch_size)

        model_for_training = self.ddp_net if self.is_ddp else None
        p_l, v_l, s_l, ent, g_n, f1 = self.net.train_step(
            dataloader, self.module.augment, ddp_model=model_for_training,
            n_epochs=getattr(self, 'n_epochs', 10),
            balance_class_weight=getattr(self, 'balance_sampling', False))

        if self.is_ddp:
            dist.barrier()

        if self.rank == 0:
            print(f'F1 score (new): {f1: .3f}')
        return p_l, v_l, s_l, ent, g_n, f1

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

        n_half = n_games // 2

        # 前半场: current=P1, best=P2
        results1 = self._batched_eval_games(self.net, self.best_net, n_half, self.n_playout)
        # 后半场: best=P1, current=P2
        results2 = self._batched_eval_games(self.best_net, self.net, n_half, self.n_playout)

        # current wins: P1赢(=1) in results1, P2赢(=-1) in results2
        current_wins = np.sum(results1 == 1) + np.sum(results2 == -1)
        draws = np.sum(results1 == 0) + np.sum(results2 == 0)
        win_rate = (current_wins + 0.5 * draws) / n_games

        flag = win_rate >= self.win_rate_threshold
        if flag:
            self.update_best_net()
        print('Complete.')
        return flag, win_rate

    def _batched_eval_games(self, net_p1, net_p2, n_envs, n_playout,
                            eval_noise_eps=0.05, eval_temp=0.2):
        """用 BatchedMCTS 批量对弈 n_envs 局，net_p1 执先(turn=1), net_p2 执后(turn=-1)。

        评估时使用低噪声 + 低温度采样（参考 AlphaZero.jl arena 设置），
        确保不同对局走出不同棋路，同时仍以近似最优策略对弈。

        返回 np.ndarray shape=(n_envs,)，值为 1(P1赢)/-1(P2赢)/0(平局)。"""
        from .MCTS_cpp import BatchedMCTS as PyBatchedMCTS

        board_shape = self.env.board.shape
        action_size = self.net.n_actions

        # 评估用低 Dirichlet 噪声：alpha 与训练一致，epsilon 降为 0.05
        use_sym = getattr(self, 'use_symmetry', True)
        mcts_p1 = PyBatchedMCTS(
            n_envs, c_init=self.c_puct, c_base=500, discount=self.discount,
            alpha=self.dirichlet_alpha, n_playout=n_playout, game_name=self.env_name,
            noise_epsilon=eval_noise_eps, use_symmetry=use_sym)
        mcts_p2 = PyBatchedMCTS(
            n_envs, c_init=self.c_puct, c_base=500, discount=self.discount,
            alpha=self.dirichlet_alpha, n_playout=n_playout, game_name=self.env_name,
            noise_epsilon=eval_noise_eps, use_symmetry=use_sym)

        envs = [self.env.copy() for _ in range(n_envs)]
        for e in envs:
            e.reset()

        boards = np.zeros((n_envs, *board_shape), dtype=np.int8)
        turns = np.ones(n_envs, dtype=np.int32)
        done = np.zeros(n_envs, dtype=bool)
        results = np.zeros(n_envs, dtype=np.int32)

        while not done.all():
            active_idx = np.where(~done)[0]
            current_turn = int(envs[active_idx[0]].turn)

            for i in range(n_envs):
                boards[i] = envs[i].board.astype(np.int8)
                turns[i] = int(envs[i].turn)

            if current_turn == 1:
                mcts_p1.batch_playout(net_p1, boards, turns)
                visits = mcts_p1.get_visits_count()
            else:
                mcts_p2.batch_playout(net_p2, boards, turns)
                visits = mcts_p2.get_visits_count()

            # 低温度采样：temp>0 保证不同对局走不同棋路
            if eval_temp > 0:
                actions = self._sample_actions(visits, eval_temp, active_idx, done)
            else:
                actions = np.argmax(visits, axis=1).astype(np.int32)

            mcts_p1.prune_roots(actions)
            mcts_p2.prune_roots(actions)

            for i in active_idx:
                envs[i].step(int(actions[i]))
                if envs[i].done():
                    done[i] = True
                    results[i] = envs[i].winPlayer()

        return results

    @staticmethod
    def _sample_actions(visits, temp, active_idx, done):
        """根据 visit counts 做温度采样，已结束的游戏用 argmax。"""
        n_envs = visits.shape[0]
        actions = np.argmax(visits, axis=1).astype(np.int32)
        for i in active_idx:
            v = visits[i]
            valid_mask = v > 0
            valid_actions = np.where(valid_mask)[0]
            log_v = np.log(v[valid_mask].astype(np.float64))
            log_v -= log_v.max()
            probs = np.exp(log_v / temp)
            probs /= probs.sum()
            actions[i] = np.random.choice(valid_actions, p=probs)
        return actions

    def update_best_net(self):
        self.best_net = deepcopy(self.net)

    def _log_train_step(self, p_loss, v_loss, s_loss, entropy, grad_norm, f1):
        if self.episode_len is not None:
            swanlab.log({'Metric/Episode length': self.episode_len}, step=self.global_step)
        swanlab.log({
            'Metric/lr': self.net.opt.param_groups[0]['lr'],
            'Metric/Gradient Norm': grad_norm,
            'Metric/F1 score': f1,
            'Metric/Loss/Action Loss': p_loss,
            'Metric/Loss/Value loss': v_loss,
            'Metric/Loss/Steps loss': s_loss,
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
                  f'\tBatch size: {self.batch_size}\n'
                  f'\tWorld size: {self.world_size}')
            print('=' * 50)

            run_config = {'env_name': self.env_name, 'model': self.net.name()}
            run_config.update(self.raw_config)
            swanlab.init(project="AlphaZero-AL", experiment_name=self.name, config=run_config)
            self.buffer.load('./dataset/dataset.pt')

        best_counter = 0

        while True:
            if self.rank == 0:
                self.data_collector()
                self.global_step += 1

            p_loss, v_loss, s_loss, entropy, grad_norm, f1 = self.policy_update()

            if self.rank == 0:
                self.net.save(self.current)
                print(f'batch i: {self.global_step}, episode_len: {self.episode_len}, '
                      f'loss: {p_loss + v_loss + s_loss: .8f}, entropy: {entropy: .8f}')
                self._log_train_step(p_loss, v_loss, s_loss, entropy, grad_norm, f1)

                if self.global_step % self.interval == 0:
                    print(f'current self-play batch: {self.global_step + 1}')
                    self.r_a, self.r_b = self.update_elo()
                    print(f'Elo score: AlphaZero: {self.r_a: .2f}, Benchmark: {self.r_b: .2f}')

                    flag, win_rate = self.select_best_player(self.num_eval)
                    new_best = best_counter + 1 if flag else None
                    self._log_eval(self.r_a, self.r_b, win_rate, new_best)

                    if flag:
                        print('New best policy!!')
                        best_counter += 1
                        self.net.save(self.best)
                        os.makedirs('dataset', exist_ok=True)
                        self.buffer.save('./dataset/dataset.pt')

            if self.is_ddp:
                dist.barrier()

    def __call__(self):
        self.run()
