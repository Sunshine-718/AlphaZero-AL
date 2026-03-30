"""
使用离线 dataset 预训练网络。

用法示例:
    python tools/pretrain_dataset.py
    python tools/pretrain_dataset.py --env Othello --device cuda
    python tools/pretrain_dataset.py --buffer dataset/Connect4_dataset.pt --epochs 20
    python tools/pretrain_dataset.py --exp 002 --init params/Connect4/001/current
"""
import argparse
import json
import os
import random
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.environments import load


def _next_experiment_id(env_dir):
    os.makedirs(env_dir, exist_ok=True)
    max_id = 0
    for name in os.listdir(env_dir):
        path = os.path.join(env_dir, name)
        if os.path.isdir(path) and name.isdigit():
            max_id = max(max_id, int(name))
    return f'{max_id + 1:03d}'


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_dataset_path(env_name, explicit_path=None):
    if explicit_path is not None:
        return explicit_path
    return os.path.join('dataset', f'{env_name}_dataset.pt')


def _slice_or_default(data, key, shape, dtype, valid_count, fallback_key=None, fill_value=0):
    tensor = data.get(key)
    if tensor is None and fallback_key is not None:
        tensor = data.get(fallback_key)
    if tensor is None:
        tensor = torch.full(shape, fill_value, dtype=dtype)
    return tensor[:valid_count].to(dtype=dtype)


def _time_order_indices(capacity, ptr, valid_count):
    if ptr <= capacity:
        return torch.arange(valid_count, dtype=torch.long)
    start = ptr % capacity
    return torch.cat([
        torch.arange(start, capacity, dtype=torch.long),
        torch.arange(0, start, dtype=torch.long),
    ])


def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'dataset 不存在: {path}')

    data = torch.load(path, map_location='cpu', weights_only=True)
    state = data['state']
    prob = data['prob']
    winner = data['winner']
    ptr = int(data.get('_ptr', state.shape[0]))
    capacity = int(state.shape[0])
    valid_count = min(ptr, capacity)

    if valid_count <= 0:
        raise ValueError(f'dataset 为空: {path}')

    steps_to_end = _slice_or_default(
        data, 'steps_to_end', (capacity, 1), torch.int16, valid_count, fill_value=0)
    aux_target = _slice_or_default(
        data, 'aux_target', (capacity, 1), torch.int16, valid_count,
        fallback_key='steps_to_end', fill_value=0)
    root_wdl = _slice_or_default(
        data, 'root_wdl', (capacity, 3), torch.float32, valid_count, fill_value=0)
    valid_mask = _slice_or_default(
        data, 'valid_mask', (capacity, prob.shape[1]), torch.bool, valid_count, fill_value=True)
    future_root_wdl = _slice_or_default(
        data, 'future_root_wdl', (capacity, 3), torch.float32, valid_count, fill_value=0)

    order = _time_order_indices(capacity, ptr, valid_count)
    dataset = TensorDataset(
        state[order].float(),
        prob[order].float(),
        winner[order].to(dtype=torch.int8),
        steps_to_end[order],
        aux_target[order],
        root_wdl[order],
        valid_mask[order],
        future_root_wdl[order],
    )
    info = {
        'capacity': capacity,
        'ptr': ptr,
        'num_samples': valid_count,
        'wrapped': ptr > capacity,
        'state_shape': list(state.shape),
        'prob_shape': list(prob.shape),
        'winner_shape': list(winner.shape),
    }
    return dataset, info


def split_dataset_by_time(dataset, val_ratio):
    if val_ratio <= 0:
        return dataset, None
    total = len(dataset)
    if total < 2:
        raise ValueError('样本数不足，无法切分训练/验证集。')

    val_count = int(round(total * val_ratio))
    val_count = max(1, min(val_count, total - 1))
    train_count = total - val_count
    train_tensors = tuple(t[:train_count] for t in dataset.tensors)
    val_tensors = tuple(t[train_count:] for t in dataset.tensors)
    return TensorDataset(*train_tensors), TensorDataset(*val_tensors)


def make_dataloader(dataset, batch_size, device, num_workers=0, compile_enabled=False, shuffle=True):
    effective_compile = compile_enabled
    if compile_enabled and len(dataset) < batch_size:
        print(f'[warn] dataset 样本数 {len(dataset)} 小于 batch_size {batch_size}，自动关闭 compile 对齐模式。')
        effective_compile = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=effective_compile,
        num_workers=num_workers,
        pin_memory=device != 'cpu',
    )


def move_batch_to_device(batch, device):
    moved = []
    for tensor in batch:
        if isinstance(tensor, torch.Tensor):
            moved.append(tensor.to(device=device, non_blocking=True))
        else:
            moved.append(tensor)
    return tuple(moved)


def make_device_aware_augment(augment, device):
    def wrapped(batch):
        augmented = augment(batch)
        return move_batch_to_device(augmented, device)

    return wrapped


def evaluate_dataloader(net, dataloader, augment, args):
    device = next(net.parameters()).device
    p_sum = torch.zeros(1, device=device)
    v_sum = torch.zeros(1, device=device)
    aux_sum = torch.zeros(1, device=device)
    n_batches = 0
    use_soft = args.value_decay < 1.0 or args.distill_alpha > 0
    y_true = []
    y_pred = []
    entropy_sum = 0.0

    net.eval()
    with torch.no_grad():
        progress = tqdm(
            dataloader,
            total=len(dataloader),
            desc='val',
            leave=False,
            dynamic_ncols=True,
        )
        for batch in progress:
            batch_data = net._prepare_training_batch(batch, augment)
            log_p_pred, value_pred, steps_pred = net(
                batch_data['state'],
                action_mask=batch_data.get('valid_mask'),
            )
            p_loss = net._policy_loss(
                log_p_pred,
                batch_data['prob'],
                batch_data['policy_mask'],
                args.psw_beta,
                args.entropy_lambda,
            )
            v_loss = net._value_loss(
                value_pred,
                batch_data,
                use_soft,
                args.value_decay,
                args.distill_alpha,
                args.distill_temp,
            )
            if args.td_alpha > 0:
                td_loss = net._td_consistency_loss(
                    value_pred,
                    batch_data,
                    args.td_steps,
                    args.value_decay,
                )
                if td_loss is not None:
                    v_loss = (1 - args.td_alpha) * v_loss + args.td_alpha * td_loss

            aux_loss = F.smooth_l1_loss(steps_pred, batch_data['aux_target'])
            p_sum += p_loss
            v_sum += v_loss
            aux_sum += aux_loss
            n_batches += 1

            entropy_sum += float(
                -torch.mean(torch.sum((log_p_pred.exp() * log_p_pred).nan_to_num(0.0), dim=-1))
            )
            y_true.append(batch_data['value_class'].cpu())
            y_pred.append(torch.argmax(value_pred, dim=-1).cpu())

            progress.set_postfix({
                'loss': f'{(p_loss + v_loss + aux_loss).item():.4f}',
                'p': f'{p_loss.item():.4f}',
                'v': f'{v_loss.item():.4f}',
                'aux': f'{aux_loss.item():.4f}',
            })

    n_batches = max(n_batches, 1)
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return {
        'policy_loss': (p_sum / n_batches).item(),
        'value_loss': (v_sum / n_batches).item(),
        'aux_loss': (aux_sum / n_batches).item(),
        'total_loss': ((p_sum + v_sum + aux_sum) / n_batches).item(),
        'entropy': entropy_sum / n_batches,
        'f1': f1_score(y_true, y_pred, average='macro'),
    }


def train_one_epoch(net, dataloader, augment, args, epoch):
    device = next(net.parameters()).device
    p_sum = torch.zeros(1, device=device)
    v_sum = torch.zeros(1, device=device)
    aux_sum = torch.zeros(1, device=device)
    grad_sum = 0.0
    entropy_sum = 0.0
    n_batches = 0
    use_soft = args.value_decay < 1.0 or args.distill_alpha > 0
    y_true = []
    y_pred = []

    net.train()
    progress = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f'train epoch {epoch:03d}',
        leave=True,
        dynamic_ncols=True,
    )
    for batch_idx, batch in enumerate(progress, start=1):
        batch_data = net._prepare_training_batch(batch, augment)
        net.opt.zero_grad(set_to_none=True)
        log_p_pred, value_pred, steps_pred = net(
            batch_data['state'],
            action_mask=batch_data.get('valid_mask'),
        )
        p_loss = net._policy_loss(
            log_p_pred,
            batch_data['prob'],
            batch_data['policy_mask'],
            args.psw_beta,
            args.entropy_lambda,
        )
        v_loss = net._value_loss(
            value_pred,
            batch_data,
            use_soft,
            args.value_decay,
            args.distill_alpha,
            args.distill_temp,
        )
        if args.td_alpha > 0:
            td_loss = net._td_consistency_loss(
                value_pred,
                batch_data,
                args.td_steps,
                args.value_decay,
            )
            if td_loss is not None:
                v_loss = (1 - args.td_alpha) * v_loss + args.td_alpha * td_loss

        aux_loss = F.smooth_l1_loss(steps_pred, batch_data['aux_target'])
        loss = p_loss + v_loss + aux_loss
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 5)
        net.opt.step()

        with torch.no_grad():
            entropy = -torch.mean(torch.sum((log_p_pred.exp() * log_p_pred).nan_to_num(0.0), dim=-1))
            batch_f1 = f1_score(
                batch_data['value_class'].cpu().numpy(),
                torch.argmax(value_pred, dim=-1).cpu().numpy(),
                average='macro',
            )

        p_sum += p_loss.detach()
        v_sum += v_loss.detach()
        aux_sum += aux_loss.detach()
        grad_sum += float(grad_norm)
        entropy_sum += float(entropy)
        n_batches += 1
        y_true.append(batch_data['value_class'].cpu())
        y_pred.append(torch.argmax(value_pred, dim=-1).cpu())

        progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'p': f'{p_loss.item():.4f}',
            'v': f'{v_loss.item():.4f}',
            'aux': f'{aux_loss.item():.4f}',
            'grad': f'{float(grad_norm):.3f}',
            'f1': f'{batch_f1:.3f}',
        })
        tqdm.write(
            f'[train epoch {epoch:03d} batch {batch_idx:04d}/{len(dataloader):04d}] '
            f'loss={loss.item():.6f} '
            f'(p={p_loss.item():.6f}, v={v_loss.item():.6f}, aux={aux_loss.item():.6f}) '
            f'ent={float(entropy):.6f} grad={float(grad_norm):.4f} f1={batch_f1:.4f}'
        )

    net.eval()
    net.scheduler.step()
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    n_batches = max(n_batches, 1)
    return {
        'policy_loss': (p_sum / n_batches).item(),
        'value_loss': (v_sum / n_batches).item(),
        'aux_loss': (aux_sum / n_batches).item(),
        'total_loss': ((p_sum + v_sum + aux_sum) / n_batches).item(),
        'entropy': entropy_sum / n_batches,
        'grad_norm': grad_sum / n_batches,
        'f1': f1_score(y_true, y_pred, average='macro'),
    }


def build_parser():
    parser = argparse.ArgumentParser(description='Pretrain network from offline replay dataset')
    parser.add_argument('--env', default='Connect4', choices=['Connect4', 'Othello'])
    parser.add_argument('--model', default='CNN', choices=['CNN'])
    parser.add_argument('--buffer', default=None, help='Dataset path, default uses dataset/<env>_dataset.pt')
    parser.add_argument('--exp', default=None, help='Experiment ID to save into, default creates next numeric id')
    parser.add_argument('--init', default=None, help='Optional checkpoint directory to initialize from')
    parser.add_argument('--weights-only', action='store_true', help='Only load model weights from --init')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation ratio split from the newest samples, e.g. 0.1')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--policy-lr-scale', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile for training')
    parser.add_argument('--distill-alpha', type=float, default=0.75)
    parser.add_argument('--distill-temp', type=float, default=2.0)
    parser.add_argument('--value-decay', type=float, default=1.0)
    parser.add_argument('--psw-beta', type=float, default=0.5)
    parser.add_argument('--entropy-lambda', type=float, default=0.05)
    parser.add_argument('--td-alpha', type=float, default=0.5)
    parser.add_argument('--td-steps', type=int, default=10)
    parser.add_argument('--save-best', action='store_true', default=True,
                        help='Also save the best offline checkpoint to best/')
    parser.add_argument('--no-save-best', action='store_false', dest='save_best')
    return parser


def main():
    os.chdir(ROOT)
    args = build_parser().parse_args()
    set_seed(args.seed)

    dataset_path = resolve_dataset_path(args.env, args.buffer)
    if not os.path.exists(dataset_path):
        print(f'[skip] 未找到可用 dataset，尝试过: {dataset_path}')
        return 0

    dataset, dataset_info = load_dataset(dataset_path)
    train_dataset, val_dataset = split_dataset_by_time(dataset, args.val_ratio)
    module = load(args.env)

    env_dir = os.path.join('params', args.env)
    exp_id = args.exp or _next_experiment_id(env_dir)
    experiment_dir = os.path.join(env_dir, exp_id)
    current_dir = os.path.join(experiment_dir, 'current')
    best_dir = os.path.join(experiment_dir, 'best')

    if args.model != 'CNN':
        raise ValueError(f'Unsupported model: {args.model}')

    net = module.CNN(
        lr=args.lr,
        device=args.device,
        policy_lr_scale=args.policy_lr_scale,
        dropout=args.dropout,
    )

    if args.init:
        if args.weights_only:
            net.load_weights_only(args.init, strict=True)
        else:
            net.load(args.init)
        print(f'[init] 从 {args.init} 初始化参数')

    if args.compile:
        print('[compile] torch.compile enabled')
        net = torch.compile(net, mode='reduce-overhead', dynamic=True)

    train_augment = make_device_aware_augment(module.augment, args.device)

    print(f'[env]      {args.env}')
    print(f'[dataset]  {dataset_path}')
    print(f'[samples]  {dataset_info["num_samples"]} / {dataset_info["capacity"]}  (ptr={dataset_info["ptr"]})')
    print(f'[wrapped]  {dataset_info["wrapped"]}')
    print(f'[output]   {experiment_dir}')
    print(f'[device]   {args.device}')
    print(f'[epochs]   {args.epochs}')
    print(f'[batch]    {args.batch_size}')
    if val_dataset is None:
        print(f'[split]    train={len(train_dataset)} val=0 mode=none')
    else:
        print(f'[split]    train={len(train_dataset)} val={len(val_dataset)} mode=time')

    train_loader = make_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        compile_enabled=args.compile,
        shuffle=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = make_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            compile_enabled=False,
            shuffle=False,
        )

    best_loss = float('inf')
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(net, train_loader, train_augment, args, epoch)
        train_total_loss = train_metrics['total_loss']
        epoch_record = {
            'epoch': epoch,
            'train': train_metrics,
        }
        message = (
            f'[epoch {epoch:03d}] '
            f'train_loss={train_total_loss:.6f} '
            f'(p={train_metrics["policy_loss"]:.6f}, '
            f'v={train_metrics["value_loss"]:.6f}, '
            f'aux={train_metrics["aux_loss"]:.6f}) '
            f'ent={train_metrics["entropy"]:.6f} '
            f'grad={train_metrics["grad_norm"]:.4f} '
            f'f1={train_metrics["f1"]:.4f}'
        )
        selection_loss = train_total_loss
        if val_loader is not None:
            val_metrics = evaluate_dataloader(net, val_loader, train_augment, args)
            epoch_record['val'] = val_metrics
            selection_loss = val_metrics['total_loss']
            message += (
                f' | val_loss={val_metrics["total_loss"]:.6f} '
                f'(p={val_metrics["policy_loss"]:.6f}, '
                f'v={val_metrics["value_loss"]:.6f}, '
                f'aux={val_metrics["aux_loss"]:.6f}) '
                f'ent={val_metrics["entropy"]:.6f} f1={val_metrics["f1"]:.4f}'
            )
        epoch_record['selection_loss'] = selection_loss
        history.append(epoch_record)
        print(message)
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_state = deepcopy({
                'model': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                'optimizer': deepcopy(net.opt.state_dict()),
                'scheduler': deepcopy(net.scheduler.state_dict()),
            })

    os.makedirs(current_dir, exist_ok=True)
    net.save(current_dir)
    print(f'[save]     current -> {current_dir}')

    if args.save_best:
        if best_state is not None:
            net.load_state_dict(best_state['model'])
            net.opt.load_state_dict(best_state['optimizer'])
            net.scheduler.load_state_dict(best_state['scheduler'])
        os.makedirs(best_dir, exist_ok=True)
        net.save(best_dir)
        print(f'[save]     best    -> {best_dir}')

    summary = {
        'env': args.env,
        'model': args.model,
        'dataset': dataset_path,
        'experiment_dir': experiment_dir,
        'current_dir': current_dir,
        'best_dir': best_dir if args.save_best else None,
        'num_samples': dataset_info['num_samples'],
        'capacity': dataset_info['capacity'],
        'ptr': dataset_info['ptr'],
        'args': vars(args),
        'history': history,
    }
    summary_path = os.path.join(experiment_dir, 'pretrain_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'[save]     summary -> {summary_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
