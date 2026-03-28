import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from ..NetworkBase import Base


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
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(self.norm1(x))
        x = nn.functional.silu(x)
        x = self.dropout(x)
        x = self.conv2(self.norm2(x))
        return nn.functional.silu(x + residual)


class PolicyHead(nn.Module):
    def __init__(self, in_channels, out_dim, dropout=0.0):
        super().__init__()
        hidden_channels = 5
        flat_dim = hidden_channels * 8 * 8
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.norm = nn.RMSNorm(flat_dim, eps=1e-5)
        self.fc = nn.Linear(flat_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, action_mask=None):
        x = self.conv(x)
        x = nn.functional.silu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.norm(x))
        logits = self.fc(x)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return nn.functional.log_softmax(logits, dim=-1)


class DualHead(nn.Module):
    def __init__(self, in_channels, dropout=0.0):
        super().__init__()
        hidden_channels = 5
        flat_dim = hidden_channels * 8 * 8
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.norm = nn.RMSNorm(flat_dim, eps=1e-5)
        self.fc = nn.Linear(flat_dim, flat_dim)
        self.out_norm = nn.RMSNorm(flat_dim, eps=1e-5)
        self.value_out = nn.Linear(flat_dim, 3)
        self.aux_out = nn.Linear(flat_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.silu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.norm(x))
        h = self.out_norm(nn.functional.silu(self.fc(x)))
        value = nn.functional.log_softmax(self.value_out(h), dim=-1)
        aux = torch.tanh(self.aux_out(h).squeeze(-1))
        return value, aux


class CNN(Base):
    aux_target_offset = 64
    score_scale = 8.0

    def __init__(
        self,
        lr,
        embed_dim=32,
        h_dim=256,
        out_dim=65,
        dropout=0.2,
        device='cpu',
        num_res_blocks=4,
        policy_lr_scale=0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_dim = 3
        self.device = device
        self.n_actions = out_dim

        self.piece_emb = nn.Embedding(3, embed_dim)
        self.pos_emb = nn.Embedding(10, embed_dim)
        self.legal_emb = nn.Embedding(2, embed_dim)
        self.register_buffer('orbit_map', torch.tensor(_ORBIT_MAP, dtype=torch.long))

        self.hidden = nn.Sequential(
            nn.Conv2d(embed_dim, h_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.SiLU(inplace=True),
            *[ResidualBlock(h_dim, dropout=dropout) for _ in range(num_res_blocks)],
        )

        self.policy_head = PolicyHead(h_dim, out_dim, dropout)
        self.dual_head = DualHead(h_dim, dropout)

        self.apply(self.init_weights)
        nn.init.constant_(self.policy_head.fc.weight, 0)
        nn.init.constant_(self.dual_head.value_out.weight, 0)
        nn.init.constant_(self.dual_head.aux_out.weight, 0)

        self.opt = torch.optim.AdamW(
            [
                {'params': self.hidden.parameters()},
                {'params': self.dual_head.parameters()},
                {'params': self.piece_emb.parameters(), 'weight_decay': 0},
                {'params': self.pos_emb.parameters(), 'weight_decay': 0},
                {'params': self.legal_emb.parameters(), 'weight_decay': 0},
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

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.orthogonal_(module.weight)
            return
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

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

    def _to_state_tensor(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        return state.to(self.device, dtype=torch.float32)

    def _embed_state(self, state, action_mask=None):
        batch_size = state.size(0)
        own = state[:, 0].view(batch_size, 64) > 0.5
        opp = state[:, 1].view(batch_size, 64) > 0.5
        piece_id = own.long() + (opp.long() << 1)
        x = self.piece_emb(piece_id) + self.pos_emb(self.orbit_map).unsqueeze(0)
        if action_mask is not None:
            board_legal = action_mask[:, :64].to(dtype=torch.long)
            x = x + self.legal_emb(board_legal)
        return x.transpose(1, 2).reshape(batch_size, self.embed_dim, 8, 8)

    def forward(self, x, action_mask=None):
        action_mask = self._normalize_action_mask(action_mask, x.device)
        x = self._embed_state(x, action_mask=action_mask)
        hidden = self.hidden(x)
        log_prob = self.policy_head(hidden, action_mask)
        value, aux = self.dual_head(hidden)
        return log_prob, value, aux

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
        tensor = torch.from_numpy(state) if isinstance(state, np.ndarray) else state
        if self.device != 'cpu':
            tensor = tensor.pin_memory().to(self.device, dtype=torch.float32, non_blocking=True)
        else:
            tensor = tensor.to(self.device, dtype=torch.float32)
        action_mask = self._normalize_action_mask(action_mask, tensor.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device != 'cpu'):
            log_prob, value_log_prob, aux_scalar = self(tensor, action_mask=action_mask)
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
