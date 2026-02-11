import os
import time
import random
import torch
import queue
import pickle
import logging
import threading
import numpy as np
from src.pipeline import TrainPipeline
from src.ReplayBuffer import ReplayBuffer
from flask import Flask, request, jsonify
import argparse
import sys  # ADDED: For explicit stdout printing

# --- 新增: 流量统计全局变量 ---
TOTAL_RECEIVED_BYTES = 0
TOTAL_SENT_BYTES = 0
# ---------------------------

# --- 数据搬运线程的共享状态 ---
new_data_event = threading.Event()       # 通知训练线程有新数据可用
episode_len_lock = threading.Lock()      # 保护 episode_len_list 的并发访问
episode_len_list = []                    # 累积的 episode 长度
# -----------------------------

parser = argparse.ArgumentParser(description='AlphaZero Training Server')
parser.add_argument('--host', '-H', type=str, default='0.0.0.0', help='Host IP')
parser.add_argument('--port', '-P', '-p', type=int, default=7718, help='Port number')
parser.add_argument('-n', type=int, default=100,
                    help='Number of simulations before AlphaZero make an action')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('-c', '--c_init', type=float, default=1.25, help='C_puct init')
parser.add_argument('-a', '--alpha', type=float, default=0.3, help='Dirichlet alpha')
parser.add_argument('-b', '--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--q_size', type=int, default=100, help='Queue size')
parser.add_argument('--buf', '--buffer_size', type=int, default=100000, help='Buffer size')
parser.add_argument('--mcts_n', type=int, default=1000, help='MCTS n_playout')
parser.add_argument('--n_play', type=int, default=1, help='n_playout')
parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
parser.add_argument('-t', '--temp', type=float, default=1, help='Softmax temperature')
parser.add_argument('--thres', type=float, default=0.65, help='Win rate threshold')
parser.add_argument('--num_eval', type=int, default=50, help='Number of evaluation.')
parser.add_argument('-m', '--model', type=str, default='CNN', help='Model type (CNN)')
parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available()
                    else 'cpu', help='Device type')
parser.add_argument('-e', '--env', '--environment', type=str, default='Connect4', help='Environment name')
parser.add_argument('--interval', type=int, default=10, help='Eval interval')
parser.add_argument('--name', type=str, default='AZ', help='Name of AlphaZero')
parser.add_argument('--no-cache', action='store_false', dest='cache', help='Disable transposition table')
parser.add_argument('--cache_size', type=int, default=5000, help='LRU transposition table max size')
parser.add_argument('--pause', action='store_true', help='Pause')
args = parser.parse_args()

config = {"lr": args.lr,
          "temp": args.temp,
          "c_puct": args.c_init,
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
          "use_cache": args.cache,
          "cache_size": args.cache_size}


inbox = queue.Queue()

app = Flask(__name__)


def _inbox_worker(buffer):
    """后台线程：持续从 inbox 取数据存入 buffer，每存完一局就通知训练线程。"""
    global episode_len_list
    while True:
        play_data = inbox.get()  # 阻塞等待
        for data in play_data:
            buffer.store(*data)
        with episode_len_lock:
            episode_len_list.append(len(play_data))
        new_data_event.set()  # 通知训练线程


def data_collector(self):
    """等待 buffer 中有足够新数据后返回（非阻塞式消费）。"""
    global episode_len_list
    flag = 0
    while True:
        with episode_len_lock:
            n_episodes = len(episode_len_list)
        if n_episodes >= args.q_size:
            break
        if flag != n_episodes:
            print(f'[Pending] {n_episodes}/{args.q_size}')
            flag = n_episodes
        new_data_event.wait(timeout=1)
        new_data_event.clear()

    with episode_len_lock:
        self.episode_len = int(np.mean(episode_len_list)) if episode_len_list else 0
        episode_len_list.clear()


TrainPipeline.data_collector = data_collector


@app.route('/upload', methods=['POST'])
def upload():
    # --- 流量统计: 接收流量 ---
    global TOTAL_RECEIVED_BYTES
    if request.data:
        data_len = len(request.data)
        TOTAL_RECEIVED_BYTES += data_len
        # ADDED: Log traffic for GUI to capture (Server RECEIVED = Client UPLOAD)
        print(f"[[TRAFFIC_LOG::RECEIVED::+::{data_len}]]", file=sys.stdout)
    # -----------------------
    global inbox
    data = pickle.loads(request.data)
    for d in data:
        # print(f'Received data from {request.remote_addr}:{request.environ.get("REMOTE_PORT")}')
        inbox.put(d)
    return jsonify({'status': 'success'})


@app.route('/weights', methods=['GET'])
def weights():
    try:
        mtime = os.path.getmtime(pipeline.current)
    except FileNotFoundError:
        mtime = time.time()
    try:
        client_ts = float(request.args.get('ts', 0))
    except ValueError:
        client_ts = 0
    if client_ts == 0:
        print(f'Client {request.remote_addr}:{request.environ.get("REMOTE_PORT")} connected.')
    if mtime > client_ts and os.path.exists(pipeline.current):
        while True:
            try:
                params = torch.load(pipeline.current, map_location='cpu')['model_state_dict']
                break
            except RuntimeError:
                time.sleep(1)

        # --- 流量统计: 发送流量 ---
        payload = pickle.dumps(params, protocol=pickle.HIGHEST_PROTOCOL)
        payload_len = len(payload)
        global TOTAL_SENT_BYTES
        TOTAL_SENT_BYTES += payload_len

        # ADDED: Log traffic for GUI to capture (Server SENT = Client DOWNLOAD)
        print(f"[[TRAFFIC_LOG::SENT::+::{payload_len}]]", file=sys.stdout)
        # ------------------------

        return payload, 200, {
            'Content-Type': 'application/octet-stream',
            'X-Timestamp': str(mtime)
        }
    else:
        return '', 304

# --- 接口: 状态查询 (供 GUI 使用) ---


@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'total_received_bytes': TOTAL_RECEIVED_BYTES,
        'total_sent_bytes': TOTAL_SENT_BYTES
    })
# -------------------------------------

# --- 接口: 重置流量统计 (NEW) ---


@app.route('/reset_traffic', methods=['POST'])
def reset_traffic():
    global TOTAL_RECEIVED_BYTES
    global TOTAL_SENT_BYTES
    TOTAL_RECEIVED_BYTES = 0
    TOTAL_SENT_BYTES = 0
    print('Network traffic statistics reset.')
    return jsonify({'status': 'success', 'message': 'Traffic stats reset'})
# -------------------------------------


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    setup_seed(0)
    log_file = 'flask_access.log'
    with open(log_file, 'w'):
        pass
    pipeline = TrainPipeline(args.env, args.model, args.name, args.n_play, config)
    buffer = ReplayBuffer(3, pipeline.buffer_size, 7, 6, 7, device=pipeline.device)
    pipeline.init_buffer(buffer)

    # 启动后台数据搬运线程：inbox → buffer（持续运行）
    worker = threading.Thread(target=_inbox_worker, args=(buffer,), daemon=True)
    worker.start()

    if not args.pause:
        t = threading.Thread(target=pipeline, daemon=True)
        t.start()

    # Flask 日志配置
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setLevel(logging.INFO)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.INFO)
    log.handlers = [handler]
    app.logger.handlers = [handler]

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
