import torch
import requests
import pickle
import argparse
import signal
import time
import warnings
import numpy as np

# 引入 Cython 编译的高性能模块
from src.environments import load
from src.mcts_cython import BatchedMCTS

warnings.filterwarnings('error', category=RuntimeWarning)

# --- 全局控制 ---
running = True


def _stop(*_):
    global running
    running = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

# --- 参数解析 (保持与原版兼容，增加并行相关参数) ---
parser = argparse.ArgumentParser(description="AlphaZero Batched Actor.")
parser.add_argument('-n', type=int, default=100, help='Number of simulations (MCTS counts)')
parser.add_argument('--host', '-H', type=str, default='127.0.0.1', help='Host IP')
parser.add_argument('--port', '-P', '-p', type=int, default=7718, help='Port number')
parser.add_argument('-c', '--c_init', type=float, default=1.25, help='C_puct init')
parser.add_argument('--c_base', type=float, default=500, help='C_puct init')
# 新增: 批处理大小
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Parallel game batch size')

parser.add_argument('-a', '--alpha', type=float, default=0.3, help='Dirichlet alpha')
parser.add_argument('--n_play', type=int, default=1, help='Number of games to collect per upload cycle')
parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
parser.add_argument('-t', '--temp', '--temperature', type=float, default=1, help='Softmax temperature')
parser.add_argument('--tempD', type=float, default=0.99, help='Temperature discount factor')
parser.add_argument('-m', '--model', type=str, default='CNN', help='Model type (CNN)')
parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available()
                    else 'cpu', help='Device type')
parser.add_argument('-e', '--env', '--environment', type=str, default='Connect4', help='Environment name')
parser.add_argument('--retry', type=int, default=3, help='Retry connection times')

args = parser.parse_args()
Env = load(args.env).Env
headers = {'Content-Type': 'application/octet-stream'}


class BatchedActor:
    def __init__(self, env_name=args.env):
        if env_name != 'Connect4':
            raise ValueError(f'Only Connect4 is supported for BatchedMCTS currently.')

        self.env_name = env_name
        self.module = load(env_name)

        # 加载模型
        if args.model == 'CNN':
            self.net = self.module.CNN(lr=0, device=args.device)
        else:
            raise ValueError(f'Unknown model type: {args.model}')

        self.net.eval()
        self.mtime = 0
        self.batch_size = args.batch_size

        # 初始化并行 MCTS
        # [修改] max_nodes 大幅增加以支持树持久化 (Tree Persistence)
        # 预估：游戏长度(40) * 每次搜索扩展节点数(n=100) * 安全系数(2)
        # 这里的 5000 足够 Connect4 使用，内存消耗可控
        self.mcts = BatchedMCTS(n_envs=self.batch_size,
                                max_nodes=args.n * 50, 
                                n_actions=7,
                                c_puct_init=args.c_init,
                                c_puct_base=args.c_base)

    def load_weights(self):
        """从服务器拉取最新权重"""
        try:
            r = requests.get(f'http://{args.host}:{args.port}/weights?ts={self.mtime}', timeout=5)
            if r.status_code == 200:
                if self.mtime == 0:
                    print(f'Server Connected. Starting Batched Collection (Batch={self.batch_size})...')

                # 流量日志
                download_size = len(r.content)
                print(f"[[TRAFFIC_LOG::DOWNLOAD::+::{download_size}]]")

                weights = pickle.loads(r.content)
                self.net.to('cpu')
                self.net.load_state_dict(weights)
                self.net.to(args.device)
                self.net.eval()
                self.mtime = float(r.headers['X-Timestamp'])
                return True
        except Exception as e:
            print(f"[Warn] Load weights failed: {e}")
        return False

    def push_data(self, data):
        """上传对局数据"""
        try:
            payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            resp = requests.post(f'http://{args.host}:{args.port}/upload', headers=headers, data=payload)

            if resp.status_code == 200:
                upload_size = len(payload)
                print(f"[[TRAFFIC_LOG::UPLOAD::+::{upload_size}]]")
        except Exception as e:
            print(f"[Warn] Upload data failed: {e}")

    def prepare_input(self, boards, turns):
        """
        将原始棋盘转换为网络输入张量 (Batch, 3, 6, 7)
        """
        B = boards.shape[0]
        x = torch.zeros(B, 3, 6, 7, device=args.device)

        boards_t = torch.from_numpy(boards).to(args.device)
        turns_t = torch.from_numpy(turns).to(args.device).view(B, 1, 1)

        x[:, 0] = (boards_t == 1).float()
        x[:, 1] = (boards_t == -1).float()
        x[:, 2] = turns_t.float()

        return x

    def run_batched_selfplay(self):
        """核心并行循环，支持 Tree Persistence"""
        # 1. 更新权重
        self.load_weights()

        # 2. 初始化 batch_size 个环境
        envs = [Env() for _ in range(self.batch_size)]

        # Buffer 用于存储进行中的游戏数据
        data_buffer = [
            {'states': [], 'actions': [], 'probs': [], 'players': [], 'steps': 0}
            for _ in range(self.batch_size)
        ]

        finished_games = []
        active_indices = list(range(self.batch_size))
        
        # [修改] 初始状态下全部重置 MCTS (使用 reset_indices)
        self.mcts.reset()

        # 3. 主循环：直到所有环境都完成一局
        while active_indices and running:
            
            current_batch_size = len(active_indices)
            
            # [关键] 构造 int32 类型的 indices 数组，供 Cython 使用
            active_indices_arr = np.array(active_indices, dtype=np.int32)

            # --- MCTS 搜索 ---
            current_boards = np.array([envs[i].board for i in active_indices])
            current_turns = np.array([envs[i].turn for i in active_indices], dtype=np.int32)

            # [修改] 不再调用 self.mcts.reset()，以保留搜索树

            # =================================================================
            # [Step 1] 根节点初始化与噪声注入
            # =================================================================
            net_input = self.prepare_input(current_boards, current_turns)

            with torch.no_grad():
                log_probs, _ = self.net(net_input)
                root_policies = log_probs.exp().cpu().numpy()

            # 生成并注入 Dirichlet 噪声
            noise = np.random.dirichlet([args.alpha] * 7, size=current_batch_size).astype(np.float32)

            # 注意：set_root_priors 需要真实的 env_idx
            for k in range(current_batch_size):
                env_idx = active_indices[k]
                self.mcts.set_root_priors(env_idx, root_policies[k], noise[k])

            # =================================================================
            # [Step 2] 执行 N 次模拟
            # =================================================================
            for _ in range(args.n):
                # Phase 1: Selection (传入 active_indices 以正确映射内存)
                ret = self.mcts.search_batch(current_boards, current_turns, active_indices_arr)
                leaf_parents, leaf_actions, leaf_boards, leaf_turns, leaf_dones, leaf_winners = ret

                # Phase 2: Inference
                net_input = self.prepare_input(leaf_boards, leaf_turns)

                with torch.no_grad():
                    log_probs, value_out = self.net(net_input)
                    probs = log_probs.exp().cpu().numpy()

                    # CNN Value Head: [Draw, Win, Lose]
                    value_probs = value_out.exp().cpu().numpy()
                    values = value_probs[:, 1] - value_probs[:, 2]

                # Phase 3: Terminal Value Fix
                for i in range(current_batch_size):
                    if leaf_dones[i]:
                        if leaf_winners[i] == 0:
                            values[i] = 0.0
                        else:
                            if leaf_winners[i] == leaf_turns[i]:
                                values[i] = 1.0
                            else:
                                values[i] = -1.0

                # Phase 4: Backprop (传入 active_indices 以正确映射内存)
                leaf_valid_masks = (leaf_boards[:, 0, :] == 0).astype(np.float32)
                self.mcts.backprop_batch(leaf_parents, leaf_actions, probs, values, 
                                       leaf_valid_masks, active_indices_arr)

            # --- 选步与执行 ---
            next_active = []
            step_actions = [] # 记录本轮所有 active env 采取的动作
            finished_indices = []

            for k, env_idx in enumerate(active_indices):
                env = envs[env_idx]
                data = data_buffer[env_idx]
                steps = data['steps']

                temperature = args.temp * pow(args.tempD, steps)
                
                # 获取 Visit Count
                counts = self.mcts.get_action_visits(env_idx)

                if counts.sum() == 0:
                    action = -1
                    policy = np.ones(7) / 7
                else:
                    if temperature < 1e-3:
                        policy = np.zeros(7, dtype=np.float32)
                        best_action = np.argmax(counts)
                        policy[best_action] = 1.0
                    else:
                        log_counts = np.log(counts + 1e-10)
                        policy = np.exp(log_counts / temperature)
                        policy /= policy.sum()

                    action = np.random.choice(7, p=policy)

                    # 收集数据
                    state_to_save = env.current_state()[0].astype(np.int8)
                    data['states'].append(state_to_save)
                    data['actions'].append(action)
                    data['probs'].append(policy)
                    data['players'].append(env.turn)

                    env.step(action)
                    data['steps'] += 1
                
                step_actions.append(action) # 无论是否结束，都要记录动作以更新树（或重置）

                if env.done() or action == -1:
                    winner = env.winPlayer()
                    finished_indices.append(env_idx) # 标记结束

                    # 结算游戏
                    states = data['states']
                    actions_list = data['actions']
                    probs_list = data['probs']
                    players = data['players']

                    final_state = env.current_state()[0].astype(np.int8)
                    next_states = states[1:] + [final_state]

                    winner_z = np.zeros(len(players), dtype=np.int32)
                    if winner != 0:
                        winner_z[np.array(players) == winner] = 1
                        winner_z[np.array(players) != winner] = -1

                    disc_factors = list(reversed([pow(args.discount, k) for k in range(len(winner_z))]))

                    dones = [False] * len(players)
                    dones[-1] = True

                    game_result = list(zip(
                        states,
                        actions_list,
                        probs_list,
                        disc_factors,
                        winner_z,
                        next_states,
                        dones
                    ))

                    finished_games.append(game_result)
                    data_buffer[env_idx] = {'states': [], 'actions': [], 'probs': [], 'players': [], 'steps': 0}
                else:
                    next_active.append(env_idx)

            # =================================================================
            # [Step 3] 树维护 (Tree Maintenance) - 核心修复
            # =================================================================
            
            # 1. 对所有刚才进行了一步的 Active Envs，移动根节点 (Prune)
            if active_indices:
                self.mcts.prune_roots(active_indices_arr, np.array(step_actions, dtype=np.int32))
            
            # 2. 对所有已经结束的 Envs，重置内存 (Reset)
            if finished_indices:
                self.mcts.reset_indices(np.array(finished_indices, dtype=np.int32))

            active_indices = next_active

        # 上传数据
        if finished_games:
            print(f"Collected {len(finished_games)} games. Uploading...")
            self.push_data(finished_games)


if __name__ == '__main__':
    actor = BatchedActor()
    print(f"Batched Actor started on device: {args.device}")

    for i in range(args.retry):
        try:
            while running:
                actor.run_batched_selfplay()
        except KeyboardInterrupt:
            running = False
        except Exception as e:
            print(f'Error: {e}, retry {i + 1}')
            import traceback
            traceback.print_exc()
            time.sleep(1)
            continue
    print('quit')