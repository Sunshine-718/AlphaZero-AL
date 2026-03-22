import math
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


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with BatchNorm, SiLU activation, and Dropout.
    Conv -> BN -> SiLU -> Dropout -> Conv -> BN -> Add -> SiLU
    """

    def __init__(self, in_channels, out_channels, dropout_rate=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.silu(out)


class CNN(Base):
    def __init__(self, lr, embed_dim=32, h_dim=64, out_dim=7, dropout=0.2, device='cpu', num_res_blocks=3, policy_lr_scale=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_dim = 3  # 保持与 server.py ReplayBuffer 的兼容性
        self.device = device
        self.n_actions = out_dim

        # Embedding 层
        self.piece_emb = nn.Embedding(3, embed_dim)    # 0=空, 1=己方, 2=对方
        self.pos_emb = nn.Embedding(24, embed_dim)     # 24 个轨道 (左右镜像对称)
        self.register_buffer('orbit_map', torch.tensor(_ORBIT_MAP, dtype=torch.long))

        # 6x7 board with padding=2 → 8x9 feature maps
        policy_conv_flat = 5 * 8 * 9

        # Body: Stem + Residual Blocks
        self.hidden = nn.Sequential(
            nn.Conv2d(embed_dim, h_dim, kernel_size=3, padding=2, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.SiLU(inplace=True),
            *[ResidualBlock(h_dim, h_dim, dropout_rate=dropout) for _ in range(num_res_blocks)]
        )

        # Policy head: conv → flatten → linear → log_softmax
        self.policy_head = nn.Sequential(
            nn.Conv2d(h_dim, 5, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.LayerNorm(policy_conv_flat),
            nn.Linear(policy_conv_flat, out_dim),
            nn.LogSoftmax(dim=-1)
        )

        # Value head: WDL (draw, p1_win, p2_win)
        self.value_head = nn.Sequential(
            nn.Conv2d(h_dim, 5, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(policy_conv_flat, policy_conv_flat),
            nn.SiLU(inplace=True),
            nn.LayerNorm(policy_conv_flat),
            nn.Linear(policy_conv_flat, 3),
            nn.LogSoftmax(dim=-1)
        )

        # Auxiliary head: moves to end, 0-42
        self.steps_head = nn.Sequential(
            nn.Conv2d(h_dim, 5, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.LayerNorm(policy_conv_flat),
            nn.Linear(policy_conv_flat, 43),
            nn.LogSoftmax(dim=-1)
        )

        self.apply(self.init_weights)
        nn.init.constant_(self.policy_head[-2].weight, 0)
        nn.init.constant_(self.value_head[-2].weight, 0)
        nn.init.constant_(self.steps_head[-2].weight, 0)

        self.opt = torch.optim.AdamW([
            {'params': self.hidden.parameters()},
            {'params': self.value_head.parameters()},
            {'params': self.steps_head.parameters()},
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
            if m is self.piece_emb:
                # 正交初始化: 空/己方/对方 三向量两两正交，等距分布
                nn.init.orthogonal_(m.weight)
            elif m is self.pos_emb:
                # 正交初始化 + 按战略价值缩放范数
                # 行0=顶行(最难填到), 行5=底行(最先填); 列0=边, 列3=中心
                nn.init.orthogonal_(m.weight)
                norm_scale = torch.tensor([
                    # row 0 (top): col 0, 1, 2, 3
                    0.30, 0.35, 0.40, 0.45,
                    # row 1
                    0.40, 0.50, 0.60, 0.65,
                    # row 2
                    0.50, 0.60, 0.75, 0.80,
                    # row 3
                    0.60, 0.70, 0.85, 0.90,
                    # row 4
                    0.65, 0.75, 0.90, 0.95,
                    # row 5 (bottom): most accessible
                    0.70, 0.80, 0.95, 1.00,
                ])
                m.weight.data.mul_(norm_scale.unsqueeze(1))
            return
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def name(self):
        return 'CNN'

    def _embed_state(self, state):
        """将 (B, 3, 6, 7) 状态转为 (B, embed_dim, 6, 7) embedding 表示。"""
        B = state.size(0)
        # ch0=1 → 己方棋子, ch1=1 → 对方棋子, 两者都为 0 → 空
        piece_ids = (state[:, 0] + state[:, 1] * 2).long().view(B, 42)

        pe = self.piece_emb(piece_ids)           # (B, 42, d)
        po = self.pos_emb(self.orbit_map)        # (42, d)

        x = pe + po.unsqueeze(0)                 # (B, 42, d)
        return x.permute(0, 2, 1).view(B, self.embed_dim, 6, 7)

    def forward(self, x, action_mask=None):
        x = self._embed_state(x)
        hidden = self.hidden(x)
        log_prob = self.policy_head(hidden)
        value = self.value_head(hidden)
        steps_pred = self.steps_head(hidden)
        return log_prob, value, steps_pred

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
        wdl = value_log_prob.exp()  # (batch, 3)

        steps_prob = log_steps.exp()
        idx = torch.arange(43, dtype=torch.float32, device=self.device)
        expected_steps = (steps_prob * idx).sum(dim=1)

        return (log_prob.exp().cpu().numpy(),
                wdl.cpu().numpy(),
                expected_steps.cpu().view(-1, 1).numpy())
