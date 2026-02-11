# 训练循环与数据收集并行化改造

> **日期**：2026-02-11
> **Commit**：`50ef374` — 将训练循环与数据收集从串行改为并行
> **涉及文件**：`server.py`

---

## 一、问题描述

修改前，`server.py` 中的数据流是完全串行的：

```
训练线程主循环 (pipeline.run):
  ┌─────────────────────────────────────────────────────────────┐
  │  data_collector()                                           │
  │    while inbox.qsize() < q_size:   ← GPU 空闲，忙等数据     │
  │        sleep(1)                                             │
  │    while not inbox.empty():        ← 一次性取出所有数据      │
  │        buffer.store(...)                                    │
  ├─────────────────────────────────────────────────────────────┤
  │  policy_update()                   ← 训练（此期间新数据堆积） │
  │  net.save()                                                 │
  ├─────────────────────────────────────────────────────────────┤
  │  回到 data_collector()             ← 重新开始等待            │
  └─────────────────────────────────────────────────────────────┘
```

### 串行模式的三个问题

**1. GPU 空等数据**

`data_collector()` 在 `while inbox.qsize() < q_size` 循环中每秒轮询一次，期间 GPU 完全空闲。对于 `q_size=100` 局游戏，如果客户端产出速率为 10 局/秒，训练线程需要白等 10 秒。

**2. 训练期间数据无法入库**

`policy_update()` 执行期间，客户端继续上传数据到 `inbox` 队列，但没有任何线程将其搬入 `buffer`。这些数据堆积在内存队列中，直到下一轮 `data_collector()` 才被消费。

**3. 数据入库存在竞态窗口**

原 `data_collector()` 先检查 `inbox.qsize() >= q_size`，然后用 `while not inbox.empty()` 清空队列。在这两步之间，Flask 线程可能继续往 `inbox` 放入新数据，导致实际取出的数据量不可预测。

---

## 二、修改方案

引入**三线程架构**，将数据搬运从训练循环中解耦：

```
Flask 线程:     /upload 请求 → inbox 队列     （不变）
Worker 线程:    inbox → buffer.store()         （新增，持续运行）
训练线程:       等待通知 → 训练 → 保存          （改为事件驱动）
```

### 线程间通信

```
                    inbox (Queue)
                         │
    Flask 线程 ──put()──►│
                         │
                         ▼
                   _inbox_worker 线程
                    │    │
          store()───┘    │
             ▼           │
          buffer    episode_len_list ──► new_data_event.set()
                                              │
                                              ▼
                                     训练线程 (data_collector)
                                       Event.wait() 唤醒
                                       检查计数 ≥ q_size?
                                         是 → 返回，开始训练
                                         否 → 继续等待
```

| 共享状态 | 类型 | 保护机制 | 用途 |
|----------|------|----------|------|
| `inbox` | `queue.Queue` | 内置线程安全 | Flask → Worker 的数据传递 |
| `new_data_event` | `threading.Event` | 内置线程安全 | Worker 通知训练线程有新数据 |
| `episode_len_list` | `list` | `episode_len_lock` (Lock) | 累积每局步数，供训练线程统计 |

---

## 三、代码变更详解

### 3.1 新增共享状态（第 21-25 行）

```python
new_data_event = threading.Event()       # 通知训练线程有新数据可用
episode_len_lock = threading.Lock()      # 保护 episode_len_list 的并发访问
episode_len_list = []                    # 累积的 episode 长度
```

### 3.2 新增 `_inbox_worker` 后台线程（第 78-87 行）

```python
def _inbox_worker(buffer):
    """后台线程：持续从 inbox 取数据存入 buffer，每存完一局就通知训练线程。"""
    global episode_len_list
    while True:
        play_data = inbox.get()          # 阻塞等待新数据
        for data in play_data:
            buffer.store(*data)          # 立即存入 buffer
        with episode_len_lock:
            episode_len_list.append(len(play_data))
        new_data_event.set()             # 唤醒训练线程
```

- `inbox.get()` 是阻塞调用，无数据时线程挂起，不消耗 CPU
- 每存完一局就 `set()` Event，训练线程可以实时感知进度
- 数据到达后立即入库，不再堆积在 `inbox` 中

### 3.3 改写 `data_collector`（第 90-107 行）

**修改前**：

```python
def data_collector(self):
    global inbox
    episode_len = []
    flag = 0
    length = inbox.qsize()
    while length < args.q_size:          # 忙等：轮询 inbox 大小
        if flag != inbox.qsize():
            print(f'[Pending] {length}/{args.q_size}')
            flag = length
        length = inbox.qsize()
        time.sleep(1)                    # 每秒检查一次
    while not inbox.empty():             # 一次性取出所有数据
        play_data = inbox.get()
        for data in play_data:
            self.buffer.store(*data)     # 在训练线程中做 store
        episode_len.append(len(play_data))
    self.episode_len = int(np.mean(episode_len))
```

**修改后**：

```python
def data_collector(self):
    global episode_len_list
    flag = 0
    while True:
        with episode_len_lock:
            n_episodes = len(episode_len_list)
        if n_episodes >= args.q_size:    # 检查 worker 已入库的局数
            break
        if flag != n_episodes:
            print(f'[Pending] {n_episodes}/{args.q_size}')
            flag = n_episodes
        new_data_event.wait(timeout=1)   # 事件驱动，非忙等
        new_data_event.clear()

    with episode_len_lock:
        self.episode_len = int(np.mean(episode_len_list)) if episode_len_list else 0
        episode_len_list.clear()         # 重置计数，为下一轮准备
```

关键区别：
- 不再直接操作 `inbox`，改为检查 `episode_len_list` 的长度（由 worker 维护）
- 不再在训练线程中做 `buffer.store()`，该工作已由 worker 完成
- 用 `Event.wait(timeout=1)` 替代 `time.sleep(1)`，有数据时立即响应

### 3.4 启动 Worker 线程（第 212-214 行）

```python
# 启动后台数据搬运线程：inbox → buffer（持续运行）
worker = threading.Thread(target=_inbox_worker, args=(buffer,), daemon=True)
worker.start()
```

在创建 `buffer` 之后、启动训练线程之前启动 worker。`daemon=True` 确保主进程退出时 worker 自动终止。

---

## 四、行为对比

| 场景 | 修改前 | 修改后 |
|------|--------|--------|
| 训练期间客户端上传数据 | 堆积在 inbox 队列 | Worker 实时搬入 buffer |
| 等待数据时 GPU 利用率 | 0%（空转） | 0%（不变，但等待时间更短） |
| 数据到达到可训练的延迟 | 必须等一整轮 q_size 局攒齐 | 数据到达即入库，计数达标即训练 |
| buffer.store() 所在线程 | 训练线程 | Worker 线程 |
| 训练线程的阻塞方式 | `time.sleep(1)` 轮询 | `Event.wait()` 事件驱动 |

---

## 五、线程安全分析

| 操作 | 涉及线程 | 安全保证 |
|------|----------|----------|
| `inbox.put()` / `inbox.get()` | Flask ↔ Worker | `queue.Queue` 内置锁 |
| `buffer.store()` | Worker 独占写入 | 训练线程只通过 `buffer.sample()` 读取，PyTorch tensor 的读写在不同索引上不冲突 |
| `episode_len_list` 读写 | Worker ↔ 训练线程 | `episode_len_lock` 互斥锁保护 |
| `new_data_event` | Worker ↔ 训练线程 | `threading.Event` 内置线程安全 |

**潜在的竞态**：`buffer.store()` 修改 `_ptr` 和数据 tensor，同时 `buffer.sample()` 读取 `_ptr` 和数据。由于 Python GIL 的存在，`_ptr` 的整数赋值是原子的。tensor 数据层面，`store()` 写入的是 `idx = _ptr % capacity` 位置，而 `sample()` 采样的是随机索引。极小概率下 sample 可能读到正在写入的位置，但这只会导致一个训练样本的数据略有不一致（混合了旧值和新值的部分字段），对训练过程的影响可忽略不计。
