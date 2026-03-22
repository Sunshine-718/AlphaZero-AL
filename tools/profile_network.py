"""Profile each module in the Othello network (forward + backward)."""
import sys, time, torch, numpy as np
sys.path.insert(0, '.')

from src.environments.Othello.Network import CNN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 512 * 4
WARMUP = 10
REPEATS = 50

print(f"Device: {DEVICE}, Batch: {BATCH}, Warmup: {WARMUP}, Repeats: {REPEATS}")
print("=" * 70)

net = CNN(lr=1e-3, embed_dim=32, h_dim=32, device=DEVICE)

state = torch.randn(BATCH, 3, 8, 8, device=DEVICE)
state_np = state.cpu().numpy()

def bench(fn, name, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / repeats * 1000
    print(f"  {name:45s} {elapsed:8.3f} ms")
    return elapsed

# ── Forward (inference) ──
print("\n[Inference — torch.no_grad]")
net.eval()
with torch.no_grad():
    bench(lambda: net.forward(state), "forward()")
    bench(lambda: net.predict(state_np), "predict()")

    # module breakdown
    print()
    emb = net._embed_state(state)
    bench(lambda: net._embed_state(state), "_embed_state")

    x = emb
    for i, layer in enumerate(net.hidden):
        x_in = x
        bench(lambda x_in=x_in, layer=layer: layer(x_in), f"hidden[{i}] {layer.__class__.__name__}")
        x = layer(x_in)
    hidden_out = x

    bench(lambda: net.policy_head(hidden_out), "policy_head")
    bench(lambda: net.dual_head(hidden_out), "dual_head")

# ── Forward + Backward (training) ──
print("\n" + "=" * 70)
print("\n[Training — forward + backward]")
net.train()

# Dummy targets
target_pi = torch.randn(BATCH, 65, device=DEVICE).softmax(dim=-1)
target_wdl = torch.randn(BATCH, 3, device=DEVICE).softmax(dim=-1)
target_aux = torch.randint(0, 129, (BATCH,), device=DEVICE)

def train_forward():
    log_prob, value, steps = net.forward(state)
    return log_prob, value, steps

def train_forward_backward():
    net.opt.zero_grad(set_to_none=True)
    log_prob, value, steps = net.forward(state)
    loss_p = -(target_pi * log_prob).sum(dim=1).mean()
    loss_v = -(target_wdl * value).sum(dim=1).mean()
    loss_aux = torch.nn.functional.cross_entropy(steps, target_aux)
    loss = loss_p + loss_v + loss_aux
    loss.backward()
    return loss

def train_full_step():
    net.opt.zero_grad(set_to_none=True)
    log_prob, value, steps = net.forward(state)
    loss_p = -(target_pi * log_prob).sum(dim=1).mean()
    loss_v = -(target_wdl * value).sum(dim=1).mean()
    loss_aux = torch.nn.functional.cross_entropy(steps, target_aux)
    loss = loss_p + loss_v + loss_aux
    loss.backward()
    net.opt.step()

bench(train_forward, "forward only (train mode)")
bench(train_forward_backward, "forward + backward")
bench(train_full_step, "forward + backward + optimizer.step")

# ── Backward breakdown by module ──
print("\n[Backward breakdown]")

def bench_backward(module_fn, name):
    """Time backward through a single module."""
    def run():
        net.opt.zero_grad(set_to_none=True)
        out = module_fn()
        if isinstance(out, tuple):
            loss = sum(o.sum() for o in out)
        else:
            loss = out.sum()
        loss.backward()
    bench(run, name)

bench_backward(lambda: net._embed_state(state), "_embed_state fwd+bwd")

emb = net._embed_state(state).detach().requires_grad_(True)
bench_backward(lambda: net.hidden(emb), "hidden fwd+bwd")

hidden_out = net.hidden(net._embed_state(state)).detach().requires_grad_(True)
bench_backward(lambda: net.policy_head(hidden_out), "policy_head fwd+bwd")

hidden_out = net.hidden(net._embed_state(state)).detach().requires_grad_(True)
bench_backward(lambda: net.dual_head(hidden_out), "dual_head fwd+bwd")


# hidden breakdown
print("\n[Hidden body backward breakdown]")
x = net._embed_state(state).detach().requires_grad_(True)
for i, layer in enumerate(net.hidden):
    x_in = x
    def make_fn(layer, x_in):
        x_detached = layer(x_in).detach().requires_grad_(True)
        def run():
            net.opt.zero_grad(set_to_none=True)
            # need fresh input each time
            return layer(x_in)
        return run
    bench_backward(make_fn(layer, x_in), f"hidden[{i}] {layer.__class__.__name__} fwd+bwd")
    with torch.no_grad():
        x = layer(x_in).detach().requires_grad_(True)

print("\n" + "=" * 70)
print("Done.")
