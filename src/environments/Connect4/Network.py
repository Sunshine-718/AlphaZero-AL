import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from ..NetworkBase import Base


# Connect4 6×7 棋盘的左右镜像对称轨道映射
# 对称变换：恒等 + 水平翻转 (col → 6-col)
# 24 个轨道 (6 rows × 4 unique columns)
_ORBIT_MAP = [
    0,  1,  2,  3,  2,  1,  0,
    4,  5,  6,  7,  6,  5,  4,
    8,  9,  10, 11, 10, 9,  8,
    12, 13, 14, 15, 14, 13, 12,
    16, 17, 18, 19, 18, 17, 16,
    20, 21, 22, 23, 22, 21, 20,
]

_ROWS = 6
_COLS = 7


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.norm = nn.GroupNorm(1, in_channels)
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        self.resid = in_channels == out_channels

    def forward(self, x):
        residual = x if self.resid else 0
        x = self.norm(x)
        x = self.conv(x)
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
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out * gate.unsqueeze(-1).transpose(1, 2).sigmoid()
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.o_proj(out) + residual


class AttentionBlock(nn.Module):
    def __init__(self, h_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = GatedAttention(h_dim, num_heads, dropout)

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        x = x.reshape(batch_size, num_channels, height * width).transpose(1, 2)
        return self.attn(x)


class ColumnPolicyHead(nn.Module):
    def __init__(self, h_dim, dropout=0.0, rows=_ROWS, cols=_COLS):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.norm = nn.RMSNorm(h_dim, eps=1e-5)
        self.row_gate = nn.Linear(h_dim, 1)
        self.fc = nn.Linear(h_dim, h_dim)
        self.out = nn.Linear(h_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, action_mask=None):
        batch_size, _, dim = x.shape
        x = self.norm(x).reshape(batch_size, self.rows, self.cols, dim).transpose(1, 2)
        row_scores = self.row_gate(x).squeeze(-1)
        row_weights = torch.softmax(row_scores, dim=-1)
        col_feat = (row_weights.unsqueeze(-1) * x).sum(dim=2)
        col_feat = self.dropout(nn.functional.silu(self.fc(col_feat)))
        logits = self.out(col_feat).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return nn.functional.log_softmax(logits, dim=-1)


class DualHead(nn.Module):
    def __init__(self, h_dim, n_classes=3, dropout=0.0):
        super().__init__()
        self.pool_norm = nn.RMSNorm(h_dim, eps=1e-5)
        self.pool_fc = nn.Linear(h_dim, h_dim)
        self.pool_drop = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(h_dim, eps=1e-5)
        self.fc = nn.Linear(h_dim, h_dim)
        self.out_norm = nn.RMSNorm(h_dim, eps=1e-5)
        self.value_out = nn.Linear(h_dim, n_classes)
        self.aux_out = nn.Linear(h_dim, 1)

    def forward(self, x):
        x = x.mean(dim=1)
        x = x + self.pool_drop(nn.functional.silu(self.pool_fc(self.pool_norm(x))))
        h = self.out_norm(nn.functional.silu(self.fc(self.norm(x))))
        value = nn.functional.log_softmax(self.value_out(h), dim=-1)
        moves_left = torch.sigmoid(self.aux_out(h).squeeze(-1))
        return value, moves_left


class CNN(Base):
    aux_target_offset = 42

    def __init__(
        self,
        lr,
        embed_dim=32,
        h_dim=64,
        out_dim=7,
        dropout=0.2,
        device='cpu',
        num_res_blocks=3,
        policy_lr_scale=0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_dim = 3  # 保持与 server.py ReplayBuffer 的兼容性
        self.device = device
        self.n_actions = out_dim

        # Embedding 层
        self.piece_emb = nn.Embedding(2, embed_dim)    # 0=己方, 1=对方
        self.pos_emb = nn.Embedding(24, embed_dim)     # 24 个轨道 (左右镜像对称)
        self.register_buffer('orbit_map', torch.tensor(_ORBIT_MAP, dtype=torch.long))

        # 6x7 board with padding=2 → 8x9 feature maps
        # Body: Stem + Residual Blocks
        self.hidden = nn.Sequential(
            nn.Conv2d(embed_dim, h_dim, kernel_size=3, padding=1),
            nn.SiLU(True),
            *[ResidualBlock(h_dim, h_dim, dropout=dropout) for _ in range(num_res_blocks)],
            AttentionBlock(h_dim, 4, dropout),
        )

        # Policy head: conv → flatten → linear → log_softmax
        self.policy_head = ColumnPolicyHead(h_dim, dropout)

        # Value head: WDL (draw, p1_win, p2_win)
        self.dual_head = DualHead(h_dim, 3, dropout)

        # Auxiliary head: moves to end, 0-42
        self.apply(self.init_weights)
        nn.init.constant_(self.policy_head.out.weight, 0)
        nn.init.constant_(self.dual_head.value_out.weight, 0)
        nn.init.constant_(self.dual_head.aux_out.weight, 0)

        self.opt = torch.optim.AdamW([
            {'params': self.hidden.parameters()},
            {'params': self.dual_head.parameters()},
            {'params': self.piece_emb.parameters(), 'weight_decay': 0},
            {'params': self.pos_emb.parameters(), 'weight_decay': 0},
            {'params': self.policy_head.parameters(), 'lr': lr * policy_lr_scale},
        ], lr=lr, weight_decay=1e-2)

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
            return
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def name(self):
        return 'CNN'

    @staticmethod
    def _normalize_action_mask(action_mask, device):
        if action_mask is None:
            return None
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.from_numpy(action_mask)
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        return action_mask.to(device=device, dtype=torch.bool)

    def _embed_state(self, state):
        """将 (B, 3, 6, 7) 状态转为 (B, embed_dim, 6, 7) embedding 表示。"""
        B = state.size(0)
        # ch0=1 → 己方棋子, ch1=1 → 对方棋子; 空格位置没有 piece embedding
        own = state[:, 0].view(B, _ROWS * _COLS)
        opp = state[:, 1].view(B, _ROWS * _COLS)
        emb_own = self.piece_emb.weight[0]
        emb_opp = self.piece_emb.weight[1]
        pe = own.unsqueeze(-1) * emb_own + opp.unsqueeze(-1) * emb_opp
        po = self.pos_emb(self.orbit_map)        # (42, d)

        x = pe + po.unsqueeze(0)                 # (B, 42, d)
        return x.permute(0, 2, 1).view(B, self.embed_dim, _ROWS, _COLS)

    def forward(self, x, action_mask=None):
        action_mask = self._normalize_action_mask(action_mask, x.device)
        x = self._embed_state(x)
        hidden = self.hidden(x)
        log_prob = self.policy_head(hidden, action_mask)
        value, steps_pred = self.dual_head(hidden)
        return log_prob, value, steps_pred

    def _to_state_tensor(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        return state.to(self.device, dtype=torch.float32)

    @torch.no_grad()
    def policy(self, state, action_mask=None):
        state = self._to_state_tensor(state)
        action_mask = self._normalize_action_mask(action_mask, state.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device != 'cpu'):
            return self(state, action_mask=action_mask)[0].exp().cpu().numpy()

    @torch.no_grad()
    def value(self, state, action_mask=None):
        state = self._to_state_tensor(state)
        action_mask = self._normalize_action_mask(action_mask, state.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device != 'cpu'):
            return self(state, action_mask=action_mask)[1].exp().cpu().numpy()

    @torch.no_grad()
    def predict(self, state, action_mask=None):
        t = torch.from_numpy(state) if isinstance(state, np.ndarray) else state
        if self.device != 'cpu':
            t = t.pin_memory().to(self.device, dtype=torch.float32, non_blocking=True)
        else:
            t = t.to(self.device, dtype=torch.float32)
        action_mask = self._normalize_action_mask(action_mask, t.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device != 'cpu'):
            log_prob, value_log_prob, steps_norm = self(t, action_mask=action_mask)
        wdl = value_log_prob.exp()
        expected_steps = steps_norm * float(self.aux_target_offset)

        if self.device != 'cpu':
            p = log_prob.float().exp().to('cpu', non_blocking=True)
            w = wdl.float().to('cpu', non_blocking=True)
            m = expected_steps.float().view(-1, 1).to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            return p.numpy(), w.numpy(), m.numpy()
        return (log_prob.exp().cpu().numpy(),
                wdl.cpu().numpy(),
                expected_steps.cpu().view(-1, 1).numpy())
