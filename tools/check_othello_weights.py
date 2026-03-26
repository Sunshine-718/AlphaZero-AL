"""Verify that the current Othello network can strictly load a checkpoint."""

import argparse
import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def latest_experiment_dir(env_dir: str) -> str | None:
    if not os.path.isdir(env_dir):
        return None
    max_id = -1
    max_path = None
    for name in os.listdir(env_dir):
        path = os.path.join(env_dir, name)
        if os.path.isdir(path) and name.isdigit():
            exp_id = int(name)
            if exp_id > max_id:
                max_id = exp_id
                max_path = path
    return max_path


def resolve_model_path(path_arg: str | None, variant: str) -> str:
    if path_arg:
        if os.path.isdir(path_arg):
            return os.path.join(path_arg, "model.pt") if os.path.isdir(os.path.join(path_arg, variant)) else (
                os.path.join(path_arg, "model.pt") if os.path.basename(path_arg) in {"current", "best"} else os.path.join(path_arg, variant, "model.pt")
            )
        return path_arg

    env_dir = os.path.join(ROOT, "params", "Othello")
    exp_dir = latest_experiment_dir(env_dir)
    if exp_dir is None:
        raise FileNotFoundError(f"No Othello experiment found under {env_dir}")
    return os.path.join(exp_dir, variant, "model.pt")


def load_checkpoint(path: str, device: str) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")
    return checkpoint


def collect_mismatches(model_state: dict, checkpoint_state: dict):
    model_keys = set(model_state.keys())
    checkpoint_keys = set(checkpoint_state.keys())

    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)

    shape_mismatch = []
    for key in sorted(model_keys & checkpoint_keys):
        model_tensor = model_state[key]
        checkpoint_tensor = checkpoint_state[key]
        model_shape = tuple(model_tensor.shape) if hasattr(model_tensor, "shape") else None
        checkpoint_shape = tuple(checkpoint_tensor.shape) if hasattr(checkpoint_tensor, "shape") else None
        if model_shape != checkpoint_shape:
            shape_mismatch.append((key, checkpoint_shape, model_shape))

    return missing, unexpected, shape_mismatch


def print_list(title: str, items: list[str], limit: int):
    print(f"{title}: {len(items)}")
    for item in items[:limit]:
        print(f"  - {item}")
    if len(items) > limit:
        print(f"  ... ({len(items) - limit} more)")


def print_shape_mismatch(items: list[tuple[str, tuple | None, tuple | None]], limit: int):
    print(f"Shape mismatches: {len(items)}")
    for key, ckpt_shape, model_shape in items[:limit]:
        print(f"  - {key}: checkpoint={ckpt_shape}, model={model_shape}")
    if len(items) > limit:
        print(f"  ... ({len(items) - limit} more)")


def main():
    parser = argparse.ArgumentParser(description="Strictly verify Othello checkpoint compatibility.")
    parser.add_argument("--path", default=None,
                        help="Checkpoint path, variant dir, or experiment dir. Default: latest Othello/current.")
    parser.add_argument("--variant", choices=("current", "best"), default="current",
                        help="Used only when --path is omitted or points to an experiment dir.")
    parser.add_argument("--device", default="cpu", help="Device for loading checkpoint and model.")
    parser.add_argument("--forward-check", action="store_true",
                        help="Run one dummy forward pass after successful load.")
    parser.add_argument("--limit", type=int, default=20,
                        help="Maximum number of mismatched keys to print per category.")
    args = parser.parse_args()

    model_path = resolve_model_path(args.path, args.variant)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    from src.environments.Othello.Network import CNN

    print(f"Model path: {model_path}")
    print(f"Device: {args.device}")

    net = CNN(lr=0, device=args.device)
    net.eval()

    checkpoint_state = load_checkpoint(model_path, args.device)
    model_state = net.state_dict()

    print(f"Checkpoint tensors: {len(checkpoint_state)}")
    print(f"Model tensors: {len(model_state)}")

    missing, unexpected, shape_mismatch = collect_mismatches(model_state, checkpoint_state)
    if missing or unexpected or shape_mismatch:
        print()
        print("Compatibility check failed before strict load.")
        if missing:
            print_list("Missing keys", missing, args.limit)
        if unexpected:
            print_list("Unexpected keys", unexpected, args.limit)
        if shape_mismatch:
            print_shape_mismatch(shape_mismatch, args.limit)
        print()

    try:
        net.load_state_dict(checkpoint_state, strict=True)
    except RuntimeError as exc:
        print("Strict load: FAILED")
        print(exc)
        return 1

    print("Strict load: OK")

    if args.forward_check:
        dummy = torch.zeros((1, 3, 8, 8), dtype=torch.float32, device=args.device)
        with torch.no_grad():
            log_prob, value, aux = net(dummy)
        print(f"Forward check: OK policy={tuple(log_prob.shape)} value={tuple(value.shape)} aux={tuple(aux.shape)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
