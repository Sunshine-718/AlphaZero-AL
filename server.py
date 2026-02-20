import os
import time
import random
import torch
import torch.distributed as dist
import queue
import pickle
import logging
import threading
import numpy as np
from src.pipeline import TrainPipeline
from src.ReplayBuffer import ReplayBuffer
from flask import Flask, request, jsonify
import argparse
import sys

# --- 流量统计全局变量 ---
TOTAL_RECEIVED_BYTES = 0
TOTAL_SENT_BYTES = 0
# -----------------------


parser = argparse.ArgumentParser(description='AlphaZero Training Server')
parser.add_argument('--host', '-H', type=str, default='0.0.0.0', help='Host IP')
parser.add_argument('--port', '-P', '-p', type=int, default=7718, help='Port number')
parser.add_argument('-n', type=int, default=100,
                    help='Number of simulations before AlphaZero make an action')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('-c', '--c_init', type=float, default=4, help='C_puct init')
parser.add_argument('--c_base_factor', type=float, default=1000, help='C_puct base factor')
parser.add_argument('--fpu_reduction', type=float, default=0.2, help='FPU reduction factor')
parser.add_argument('--eps', type=float, default=0.25, help='PUCT epsilon for noise mixing')
parser.add_argument('-a', '--alpha', type=float, default=1.55, help='Dirichlet alpha')
parser.add_argument('-b', '--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--q_size', type=int, default=100, help='Minimum buffer size before training starts')
parser.add_argument('--buf', '--buffer_size', type=int, default=100000, help='Buffer size')
parser.add_argument('--mcts_n', type=int, default=1000, help='MCTS n_playout')
parser.add_argument('--discount', type=float, default=1, help='Discount factor')
parser.add_argument('--thres', type=float, default=0.65, help='Win rate threshold')
parser.add_argument('--num_eval', type=int, default=50, help='Number of evaluation.')
parser.add_argument('-m', '--model', type=str, default='CNN', help='Model type (CNN)')
parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available()
                    else 'cpu', help='Device type')
parser.add_argument('-e', '--env', '--environment', type=str, default='Connect4', help='Environment name')
parser.add_argument('--interval', type=int, default=10, help='Eval interval')
parser.add_argument('--name', type=str, default='AZ', help='Name of AlphaZero')
parser.add_argument('--cache_size', type=int, default=10000, help='LRU transposition table max size')
parser.add_argument("--actor", type=str, default="best", help="Which weight are actors using?")

args, _ = parser.parse_known_args()

config = {"lr": args.lr,
          "c_puct": args.c_init,
          "c_base": args.n * args.c_base_factor,
          "fpu_reduction": args.fpu_reduction,
          "eps": args.eps,
          "n_playout": args.n,
          "discount": args.discount,
          "buffer_size": args.buf,
          "batch_size": args.batch_size,
          "pure_mcts_n_playout": args.mcts_n,
          "dirichlet_alpha": args.alpha,
          "init_elo": 1500,
          "num_eval": args.num_eval,
          "win_rate_threshold": args.thres,
          "interval": args.interval,
          "device": args.device,
          "cache_size": args.cache_size}


class ServerPipeline(TrainPipeline):
    def __init__(self, *args, min_buffer_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_buffer_size = min_buffer_size
        self._warmed_up = False
        self._episode_len_list: list = []
        self._episode_len_lock = threading.Lock()
        self._new_data_event = threading.Event()
        self._inbox: queue.Queue = queue.Queue()
        self.episode_len = None

    def data_collector(self):
        """冷启动阶段等待 buffer 积累到 min_buffer_size，之后直接返回让训练持续跑。"""
        if self._warmed_up:
            with self._episode_len_lock:
                self.episode_len = float(np.mean(self._episode_len_list)) if self._episode_len_list else None
                self._episode_len_list.clear()
            return

        while len(self.buffer) < self.min_buffer_size:
            self._new_data_event.wait(timeout=1)
            self._new_data_event.clear()
        self._warmed_up = True

    def inbox_worker(self, buffer):
        """后台线程：持续从 inbox 取数据存入 buffer，每存完一局就通知训练线程。"""
        while True:
            play_data = self._inbox.get()  # 阻塞等待
            for data in play_data:
                buffer.store(*data)
            with self._episode_len_lock:
                self._episode_len_list.append(len(play_data))
            self._new_data_event.set()  # 通知训练线程


app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload():
    global TOTAL_RECEIVED_BYTES
    if request.data:
        data_len = len(request.data)
        TOTAL_RECEIVED_BYTES += data_len
        print(f"[[TRAFFIC_LOG::RECEIVED::+::{data_len}]]", file=sys.stdout)
    data = pickle.loads(request.data)
    for d in data:
        pipeline._inbox.put(d)
    return jsonify({'status': 'success'})


@app.route('/weights', methods=['GET'])
def weights():
    global TOTAL_SENT_BYTES
    path = pipeline.current if args.actor == "current" else pipeline.best
    if pipeline.r_a < pipeline.r_b:
        path = pipeline.current
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return '', 304
    try:
        client_ts = float(request.args.get('ts', 0))
    except ValueError:
        client_ts = 0
    if client_ts == 0:
        print(f'Client {request.remote_addr}:{request.environ.get("REMOTE_PORT")} connected.')
    if mtime > client_ts:
        payload = pickle.dumps(
            torch.load(path, map_location='cpu', weights_only=True)["model_state_dict"],
            protocol=pickle.HIGHEST_PROTOCOL
        )
        TOTAL_SENT_BYTES += len(payload)
        print(f"[[TRAFFIC_LOG::SENT::+::{len(payload)}]]", file=sys.stdout)
        return payload, 200, {
            'Content-Type': 'application/octet-stream',
            'X-Timestamp': str(mtime)
        }
    else:
        return '', 304


@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'total_received_bytes': TOTAL_RECEIVED_BYTES,
        'total_sent_bytes': TOTAL_SENT_BYTES
    })


@app.route('/reset_traffic', methods=['POST'])
def reset_traffic():
    global TOTAL_RECEIVED_BYTES
    global TOTAL_SENT_BYTES
    TOTAL_RECEIVED_BYTES = 0
    TOTAL_SENT_BYTES = 0
    print('Network traffic statistics reset.')
    return jsonify({'status': 'success', 'message': 'Traffic stats reset'})


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # DDP 环境检测
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_ddp = world_size > 1

    if is_ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        args.device = f'cuda:{local_rank}'
        config['device'] = args.device

    # 统一种子初始化模型（保证各 rank 权重一致）
    set_seed(0)

    pipeline = ServerPipeline(args.env, args.model, args.name, config,
                              min_buffer_size=args.q_size,
                              rank=rank, world_size=world_size,
                              local_rank=local_rank)

    # 模型初始化后按 rank 重设种子（训练多样性）
    set_seed(rank)

    if rank == 0:
        log_file = 'flask_access.log'
        with open(log_file, 'w'):
            pass

        rows, cols = pipeline.env.board.shape
        buffer = ReplayBuffer(pipeline.net.in_dim, pipeline.buffer_size,
                              pipeline.net.n_actions, rows, cols,
                              device=pipeline.device)
        pipeline.init_buffer(buffer)

        # 启动后台数据搬运线程：inbox → buffer（持续运行）
        worker = threading.Thread(target=pipeline.inbox_worker, args=(buffer,), daemon=True)
        worker.start()

        # Flask 日志配置
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setLevel(logging.INFO)
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.INFO)
        log.handlers = [handler]
        app.logger.handlers = [handler]

        # Start training loop in background; Flask serves in main thread
        t = threading.Thread(target=pipeline, daemon=True)
        t.start()

        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    else:
        # 非零 rank：直接进入训练循环，在 barrier 处同步
        pipeline()

    if is_ddp:
        dist.destroy_process_group()
