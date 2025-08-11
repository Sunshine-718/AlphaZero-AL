import os
import time
import torch
import queue
import pickle
import logging
import threading
import numpy as np
from pipeline import TrainPipeline
from flask import Flask, request, jsonify


from ReplayBuffer import ReplayBuffer

inbox = queue.Queue()

app = Flask(__name__)


def data_collector(self):
    episode_len = []
    while inbox.empty():
        print('Waiting data')
        time.sleep(1)
    while not inbox.empty():
        play_data = inbox.get()
        for data in play_data:
            self.buffer.store(*data)
        episode_len.append(len(play_data))
    self.episode_len = int(np.mean(episode_len))


TrainPipeline.data_collector = data_collector


@app.route('/upload', methods=['POST'])
def upload():
    raw_data = request.data
    data = pickle.loads(raw_data)
    for d in data:
        print(f'Receive from {request.remote_addr}:{request.environ.get('REMOTE_PORT')}, length: {len(data)}')
        inbox.put(d)
    print(inbox.qsize())
    return jsonify({'status': 'success'})


@app.route('/weights', methods=['GET'])
def weights():
    mtime = os.path.getmtime(pipeline.current)
    try:
        client_ts = float(request.args.get('ts', 0))
    except ValueError:
        client_ts = 0
    if mtime > client_ts and os.path.exists(pipeline.current):
        params = torch.load(pipeline.current, map_location='cpu')
        return pickle.dumps(params, protocol=pickle.HIGHEST_PROTOCOL), 200, {
            'Content-Type': 'application/octet-stream',
            'X-Timestamp': str(mtime)
        }
    else:
        return '', 304


if __name__ == '__main__':
    pipeline = TrainPipeline()
    buffer = ReplayBuffer(3, pipeline.buffer_size, 7, 6, 7, device=pipeline.device)
    pipeline.init_buffer(buffer)
    t = threading.Thread(target=pipeline, daemon=True)
    t.start()
    
    handler = logging.FileHandler('flask_access.log', encoding='utf-8')
    handler.setLevel(logging.INFO)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.INFO)
    log.handlers = [handler]
    app.logger.handlers = [handler]
    app.run(host='0.0.0.0', port=9999, debug=False, use_reloader=False)
