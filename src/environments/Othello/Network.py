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
    aux_target_offset = 64
    score_scale = 8.0  # atan mapping scale, synced from SearchConfig at runtime

    def __init__(self, lr, embed_dim=32, h_dim=64, out_dim=65, dropout=0.2, device='cpu', num_res_blocks=4, policy_lr_scale=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_dim = 3  # 保持与 server.py ReplayBuffer 的兼容性
        self.device = device
        self.n_actions = out_dim

        # Embedding 层
        self.piece_emb = nn.Embedding(3, embed_dim)    # 0=空, 1=己方, 2=对方
        self.pos_emb = nn.Embedding(10, embed_dim)     # 10 个轨道 (D4 对称)
        self.register_buffer('orbit_map', torch.tensor(_ORBIT_MAP, dtype=torch.long))

        # 8x8 board with padding=2 → 10x10 feature maps
        FLAT = h_dim * 10 * 10

        # Body: Stem + Residual Blocks
        self.hidden = nn.Sequential(
            nn.Conv2d(embed_dim, h_dim, kernel_size=3, padding=2, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.SiLU(inplace=True),
            *[ResidualBlock(h_dim, h_dim, dropout_rate=dropout) for _ in range(num_res_blocks)]
        )

        # Policy head: conv → flatten → linear → log_softmax
        policy_conv_flat = 5 * 10 * 10
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

        # Auxiliary head: terminal disc diff from current-player perspective, in [-64, 64].
        self.steps_head = nn.Sequential(
            nn.Conv2d(h_dim, 5, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.LayerNorm(policy_conv_flat),
            nn.Linear(policy_conv_flat, 129),
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
                # 反对称: 空=零, 己方=+v, 对方=-v (L2 norm=1.0, 与 pos_emb 对齐)
                v = torch.randn(self.embed_dim)
                v = v / v.norm()
                m.weight.data[0].zero_()
                m.weight.data[1] = v
                m.weight.data[2] = -v
            elif m is self.pos_emb:
                # 正交初始化 + 按战略价值缩放范数（从训练模型提取的比例）
                # 轨道: 0=corner, 1=C-sq, 2=edge2, 3=edge3, 4=X-sq,
                #        5=inner1, 6=inner2, 7=XX-sq, 8=near-ctr, 9=center
                nn.init.orthogonal_(m.weight)
                norm_scale = torch.tensor([
                    1.00,  # 0 corner    — 最高战略价值，不可翻转
                    0.63,  # 1 C-square  — 紧邻角落的边缘格
                    0.56,  # 2 edge-2    — 边缘中段
                    0.59,  # 3 edge-3    — 边缘靠中
                    0.61,  # 4 X-square  — 对角邻角，高危格
                    0.46,  # 5 inner-1   — 内圈
                    0.45,  # 6 inner-2   — 内圈
                    0.51,  # 7 XX-square — 次对角邻角
                    0.58,  # 8 near-ctr  — 近中心
                    0.61,  # 9 center    — 中心 4 格
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
        """将 (B, 3, 8, 8) 状态转为 (B, embed_dim, 8, 8) embedding 表示。"""
        B = state.size(0)
        # ch0=1 → 己方棋子, ch1=1 → 对方棋子, 两者都为 0 → 空
        piece_ids = (state[:, 0] + state[:, 1] * 2).long().view(B, 64)

        pe = self.piece_emb(piece_ids)           # (B, 64, d)
        po = self.pos_emb(self.orbit_map)        # (64, d)

        x = pe + po.unsqueeze(0)                 # (B, 64, d)
        return x.permute(0, 2, 1).view(B, self.embed_dim, 8, 8)

    def forward(self, x):
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
        log_prob, value_log_prob, log_steps = self.forward(t)
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
