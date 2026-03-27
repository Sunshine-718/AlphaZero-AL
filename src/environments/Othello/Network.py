import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from ..NetworkBase import Base


# D4 symmetry orbit ids for the 8x8 board.
_ORBIT_MAP = [
    0, 1, 2, 3, 3, 2, 1, 0,
    1, 4, 5, 6, 6, 5, 4, 1,
    2, 5, 7, 8, 8, 7, 5, 2,
    3, 6, 8, 9, 9, 8, 6, 3,
    3, 6, 8, 9, 9, 8, 6, 3,
    2, 5, 7, 8, 8, 7, 5, 2,
    1, 4, 5, 6, 6, 5, 4, 1,
    0, 1, 2, 3, 3, 2, 1, 0,
]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.dropout = nn.Dropout2d(dropout)
        self.resid = in_channels == out_channels

    def forward(self, x):
        residual = x if self.resid else 0
        x = self.norm1(x)
        x = self.conv1(x)
        x = nn.functional.silu(x)
        x = self.norm2(x)
        self.conv2(x)
        x = nn.functional.silu(x)
        return self.dropout(x) + residual


class GatedAttention(nn.Module):
    def __init__(self, h_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert h_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = h_dim // num_heads
        self.prenorm = nn.RMSNorm(h_dim, eps=1e-5)
        self.qkv_proj = nn.Linear(h_dim, 3 * h_dim, bias=False)
        self.gate_proj = nn.Linear(h_dim, num_heads, bias=False)
        self.o_proj = nn.Linear(h_dim, h_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.dropout_p = dropout
        self._init_attention()

    def _init_attention(self):
        with torch.no_grad():
            nn.init.orthogonal_(self.qkv_proj.weight)
            nn.init.orthogonal_(self.gate_proj.weight)
            nn.init.orthogonal_(self.o_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        residual = x
        x = self.prenorm(x)

        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        gate = self.gate_proj(x)
        q, k, v = qkv.unbind(2)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        out = nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out * gate.unsqueeze(-1).transpose(1, 2).sigmoid()
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.o_proj(out) + residual


class DualHead(nn.Module):
    def __init__(self, h_dim, n_classes=3, dropout=0.0):
        super().__init__()
        self.cls_norm = nn.RMSNorm(h_dim, eps=1e-5)
        self.cls_fc = nn.Linear(h_dim, h_dim)
        self.cls_drop = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(h_dim, eps=1e-5)
        self.fc = nn.Linear(h_dim, h_dim)
        self.out_norm = nn.RMSNorm(h_dim, eps=1e-5)
        self.value_out = nn.Linear(h_dim, n_classes)
        self.aux_out = nn.Linear(h_dim, 1)

    def forward(self, cls_token):
        x = self.cls_drop(nn.functional.silu(self.cls_fc(self.cls_norm(cls_token))))
        h = self.out_norm(nn.functional.silu(self.fc(self.norm(x))) + cls_token)
        value = nn.functional.log_softmax(self.value_out(h), dim=-1)
        aux = torch.tanh(self.aux_out(h).squeeze(-1))
        return value, aux


class PolicyHead(nn.Module):
    def __init__(self, h_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.RMSNorm(h_dim, eps=1e-5)
        self.fc = nn.Linear(h_dim, h_dim)
        self.out = nn.Linear(h_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, action_mask=None):
        x = self.norm(x)
        x = self.dropout(nn.functional.silu(self.fc(x)))
        logits = self.out(x).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return nn.functional.log_softmax(logits, dim=-1)


class AttentionBlock(nn.Module):
    def __init__(self, h_dim, num_head, dropout=0.0):
        super().__init__()
        self.attn = GatedAttention(h_dim, num_head, dropout)
        self.pass_token = nn.Parameter(torch.zeros((1, 1, h_dim)))
        self.cls_token = nn.Parameter(torch.zeros((1, 1, h_dim)))

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        x = x.reshape(batch_size, num_channels, height * width).transpose(1, 2)
        x = torch.cat(
            [
                x,
                self.pass_token.expand(batch_size, -1, -1),
                self.cls_token.expand(batch_size, -1, -1),
            ],
            dim=1,
        )
        return self.attn(x)


class CNN(Base):
    aux_target_offset = 64
    score_scale = 8.0

    def __init__(
        self,
        lr,
        embed_dim=32,
        h_dim=64,
        out_dim=65,
        dropout=0.1,
        device='cpu',
        num_res_blocks=2,
        policy_lr_scale=0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_dim = 3
        self.device = device
        self.n_actions = out_dim
        self.attn_dim = 4 * h_dim

        self.piece_emb = nn.Embedding(3, embed_dim)
        self.pos_emb = nn.Embedding(10, embed_dim)
        self.register_buffer('orbit_map', torch.tensor(_ORBIT_MAP, dtype=torch.long))

        self.hidden = nn.Sequential(
            nn.Conv2d(embed_dim, h_dim, kernel_size=3, padding=1),
            nn.SiLU(True),
            *[ResidualBlock(h_dim, h_dim, dropout=dropout) for _ in range(num_res_blocks)],
            nn.Conv2d(h_dim, self.attn_dim, kernel_size=1),
            nn.BatchNorm2d(self.attn_dim),
            nn.SiLU(True),
            AttentionBlock(self.attn_dim, 8, dropout),
        )

        self.policy_head = PolicyHead(self.attn_dim, dropout)
        self.dual_head = DualHead(self.attn_dim, 3, dropout)

        self.apply(self.init_weights)
        self._reset_attention_init()
        nn.init.constant_(self.policy_head.out.weight, 0)
        nn.init.constant_(self.dual_head.value_out.weight, 0)
        nn.init.constant_(self.dual_head.aux_out.weight, 0)

        self.opt = torch.optim.AdamW(
            [
                {'params': self.hidden.parameters()},
                {'params': self.dual_head.parameters()},
                {'params': self.piece_emb.parameters(), 'weight_decay': 0},
                {'params': self.pos_emb.parameters(), 'weight_decay': 0},
                {'params': self.policy_head.parameters(), 'lr': lr * policy_lr_scale},
            ],
            lr=lr,
            weight_decay=1e-2,
        )

        scheduler_warmup = LinearLR(self.opt, start_factor=0.001, total_iters=100)
        scheduler_train = LinearLR(self.opt, start_factor=1, end_factor=0.1, total_iters=1000)
        self.scheduler = SequentialLR(
            self.opt,
            schedulers=[scheduler_warmup, scheduler_train],
            milestones=[100],
        )
        self.to(self.device)

    def init_weights(self, m):
        if isinstance(m, nn.Embedding):
            nn.init.orthogonal_(m.weight)
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def name(self):
        return 'CNN'

    def _reset_attention_init(self):
        for module in self.modules():
            if isinstance(module, GatedAttention):
                module._init_attention()

    def _embed_state(self, state):
        """Convert `(B, 3, 8, 8)` states into `(B, embed_dim, 8, 8)` embeddings."""
        batch_size = state.size(0)
        own = state[:, 0].view(batch_size, 64) > 0.5
        opp = state[:, 1].view(batch_size, 64) > 0.5

        piece_id = own.long() + (opp.long() << 1)
        x = self.piece_emb(piece_id) + self.pos_emb(self.orbit_map).unsqueeze(0)
        return x.transpose(1, 2).reshape(batch_size, self.embed_dim, 8, 8)

    def forward(self, x, action_mask=None):
        x = self._embed_state(x)
        hidden = self.hidden(x)
        action_tokens = hidden[:, :self.n_actions]
        cls_token = hidden[:, -1]
        log_prob = self.policy_head(action_tokens, action_mask)
        value, aux = self.dual_head(cls_token)
        return log_prob, value, aux

    @torch.no_grad()
    def policy(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device != 'cpu'):
            return self(state)[0].exp().cpu().numpy()

    @torch.no_grad()
    def value(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device != 'cpu'):
            return self(state)[1].exp().cpu().numpy()

    @torch.no_grad()
    def predict(self, state):
        tensor = torch.from_numpy(state)
        if self.device != 'cpu':
            tensor = tensor.pin_memory().to(self.device, dtype=torch.float32, non_blocking=True)
        else:
            tensor = tensor.float()
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device != 'cpu'):
            log_prob, value_log_prob, aux_scalar = self(tensor)
        wdl = value_log_prob.exp()

        disc_diff = aux_scalar * float(self.aux_target_offset)
        score_scale = getattr(self, 'score_scale', 8.0)
        expected_utility = torch.atan(disc_diff / score_scale) * (2.0 / math.pi)

        if self.device != 'cpu':
            policy = log_prob.float().exp().to('cpu', non_blocking=True)
            value = wdl.float().to('cpu', non_blocking=True)
            utility = expected_utility.float().view(-1, 1).to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            return policy.numpy(), value.numpy(), utility.numpy()
        return (
            log_prob.exp().cpu().numpy(),
            wdl.cpu().numpy(),
            expected_utility.cpu().view(-1, 1).numpy(),
        )
