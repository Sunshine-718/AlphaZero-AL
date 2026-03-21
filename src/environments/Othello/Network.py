import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from ..NetworkBase import Base


# D4 对称群下 8x8 棋盘的 10 个轨道映射
# 对称变换：4 旋转 (0°/90°/180°/270°) × 2 反射 (恒等/主对角线)
# 位置的战略价值是 D4 对称的（角=角、X格=X格），与棋子颜色无关
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
    """
    Standard Residual Block with BatchNorm, SiLU activation, and Dropout.
    Conv -> BN -> SiLU -> Dropout -> Conv -> BN -> Add -> SiLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.norm = nn.GroupNorm(1, in_channels)
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        if in_channels == out_channels:
            self.resid = True
        else:
            self.resid = False

    def forward(self, x):
        residual = x if self.resid else 0
        x = self.norm(x)
        x = self.conv(x)
        x = nn.functional.silu(x)
        return self.dropout(x) + residual


class Attention(nn.Module):
    def __init__(self, h_dim: int, num_head: int, dropout: float = 0.):
        super().__init__()
        assert h_dim % num_head == 0
        self.num_heads = num_head
        self.head_dim = h_dim // num_head
        self.prenorm = nn.RMSNorm(h_dim, eps=1e-5)
        self.qkv_proj = nn.Linear(h_dim, 3 * h_dim, bias=False)   # ① fused QKV
        self.o_proj = nn.Linear(h_dim, h_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        residual = x
        x = self.prenorm(x)

        qkv = self.qkv_proj(x).view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                    # 3 × (B, S, H, d)

        q = self.q_norm(q).transpose(1, 2)          # (B, H, S, d)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        out = nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.,
        )

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out) + residual


class AttentionBlock(nn.Module):
    def __init__(self, h_dim, num_head, dropout=0.):
        super().__init__()
        self.attn = Attention(h_dim, num_head, dropout)
    
    def forward(self, x):
        batch_size, num_channels, h, w = x.shape
        x = x.reshape(batch_size, num_channels, h * w).transpose(1, 2)
        attn_output = self.attn(x)
        output = attn_output.transpose(1, 2).reshape(batch_size, num_channels, h, w)
        return output


class CNN(Base):
    aux_target_offset = 64
    score_scale = 8.0  # atan mapping scale, synced from SearchConfig at runtime

    def __init__(self, lr, embed_dim=64, h_dim=32, out_dim=65, dropout=0.1, device='cpu', num_res_blocks=2, policy_lr_scale=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_dim = 3  # 保持与 server.py ReplayBuffer 的兼容性
        self.device = device
        self.n_actions = out_dim

        # Embedding 层
        self.piece_emb = nn.Embedding(2, embed_dim)    # 0=己方, 1=对方 (空格无 piece embedding)
        self.pos_emb = nn.Embedding(10, embed_dim)     # 10 个轨道 (D4 对称)
        self.register_buffer('orbit_map', torch.tensor(_ORBIT_MAP, dtype=torch.long))

        # Body: Stem + Residual Blocks (spatial dims stay 8x8, all convs use padding=1)
        self.hidden = nn.Sequential(
            ResidualBlock(embed_dim, h_dim, dropout=dropout),
            *[ResidualBlock(h_dim, h_dim, dropout=dropout) for _ in range(num_res_blocks)],
            nn.GroupNorm(1, h_dim),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1, bias=False),
            AttentionBlock(h_dim, 4, dropout),
        )

        # Policy head: conv → flatten → linear → log_softmax
        policy_conv_flat = 5 * 8 * 8
        self.policy_head = nn.Sequential(
            ResidualBlock(h_dim, 5, dropout=dropout),
            nn.Flatten(),
            nn.RMSNorm(policy_conv_flat, eps=1e-5),
            nn.Linear(policy_conv_flat, out_dim),
            nn.LogSoftmax(dim=-1)
        )

        # Value head: WDL (draw, p1_win, p2_win)
        self.value_head = nn.Sequential(
            ResidualBlock(h_dim, 5, dropout=dropout),
            nn.Flatten(),
            nn.Linear(policy_conv_flat, policy_conv_flat),
            nn.SiLU(inplace=True),
            nn.RMSNorm(policy_conv_flat, eps=1e-5),
            nn.Linear(policy_conv_flat, 3),
            nn.LogSoftmax(dim=-1)
        )

        # Auxiliary head: terminal disc diff from current-player perspective, in [-64, 64].
        self.aux_head = nn.Sequential(
            ResidualBlock(h_dim, 5, dropout=dropout),
            nn.Flatten(),
            nn.RMSNorm(policy_conv_flat, eps=1e-5),
            nn.Linear(policy_conv_flat, 129),
            nn.LogSoftmax(dim=-1)
        )

        # SPR (Self-Predictive Representations) predictor
        # 直接在 feature map 上预测 target network 的 hidden 输出
        self.spr_predictor = nn.Sequential(
            ResidualBlock(h_dim, h_dim, dropout=dropout),
            nn.Conv2d(h_dim, h_dim, kernel_size=1),
        )

        self.apply(self.init_weights)
        nn.init.constant_(self.policy_head[-2].weight, 0)
        nn.init.constant_(self.value_head[-2].weight, 0)
        nn.init.constant_(self.aux_head[-2].weight, 0)

        self.opt = torch.optim.AdamW([
            {'params': self.hidden.parameters()},
            {'params': self.value_head.parameters()},
            {'params': self.aux_head.parameters()},
            {'params': self.spr_predictor.parameters()},
            {'params': self.piece_emb.parameters(), 'weight_decay': 1e-4},
            {'params': self.pos_emb.parameters(), 'weight_decay': 1e-4},
            {'params': self.policy_head.parameters(), 'lr': lr * policy_lr_scale},
        ], lr=lr, weight_decay=1e-2)

        scheduler_warmup = LinearLR(self.opt, start_factor=0.001, total_iters=100)
        scheduler_train = LinearLR(self.opt, start_factor=1, end_factor=0.1, total_iters=1000)
        self.scheduler = SequentialLR(self.opt, schedulers=[scheduler_warmup, scheduler_train], milestones=[100])
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

    def _embed_state(self, state):
        """将 (B, 3, 8, 8) 状态转为 (B, embed_dim, 8, 8) embedding 表示。"""
        B = state.size(0)
        # 己方=ch0, 对方=ch1; 空格位置无 piece embedding (仅 pos_emb)
        own = state[:, 0].view(B, 64)       # (B, 64) float 0/1
        opp = state[:, 1].view(B, 64)       # (B, 64) float 0/1

        emb_own = self.piece_emb.weight[0]   # (d,)
        emb_opp = self.piece_emb.weight[1]   # (d,)
        pe = own.unsqueeze(-1) * emb_own + opp.unsqueeze(-1) * emb_opp  # (B, 64, d)
        po = self.pos_emb(self.orbit_map)    # (64, d)

        x = pe + po.unsqueeze(0)             # (B, 64, d)
        return x.permute(0, 2, 1).view(B, self.embed_dim, 8, 8)

    def forward(self, x):
        x = self._embed_state(x)
        hidden = self.hidden(x)
        spr_pred = self.spr_predictor(hidden)
        log_prob = self.policy_head(hidden)
        value = self.value_head(hidden)
        steps_pred = self.aux_head(hidden)
        return log_prob, value, steps_pred, spr_pred

    @torch.no_grad()
    def policy(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        x = self._embed_state(state)
        hidden = self.hidden(x)
        return self.policy_head(hidden).exp().cpu().numpy()

    @torch.no_grad()
    def value(self, state):
        x = self._embed_state(state)
        hidden = self.hidden(x)
        return self.value_head(hidden).exp().cpu().numpy()

    @torch.no_grad()
    def predict(self, state):
        t = torch.from_numpy(state)
        if self.device != 'cpu':
            t = t.pin_memory().to(self.device, dtype=torch.float32, non_blocking=True)
        else:
            t = t.float()
        x = self._embed_state(t)
        hidden = self.hidden(x)
        log_prob = self.policy_head(hidden)
        value_log_prob = self.value_head(hidden)
        log_steps = self.aux_head(hidden)
        # Value head outputs: [P(draw), P(win to-move), P(loss to-move)]
        wdl = value_log_prob.exp()  # (batch, 3)

        # Third head predicts terminal disc diff in current-player perspective.
        # Compute E[atan(x/scale)] * (2/pi) — uncertainty-aware score utility.
        steps_prob = log_steps.exp()
        idx = torch.arange(log_steps.shape[-1], dtype=torch.float32, device=self.device)
        disc_diff = idx - float(self.aux_target_offset)
        score_scale = getattr(self, 'score_scale', 8.0)
        atan_vals = torch.atan(disc_diff / score_scale) * (2.0 / math.pi)
        expected_utility = (steps_prob * atan_vals).sum(dim=1)

        return (log_prob.exp().cpu().numpy(),
                wdl.cpu().numpy(),
                expected_utility.cpu().view(-1, 1).numpy())
