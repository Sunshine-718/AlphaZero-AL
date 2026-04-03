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
        assert out_dim == 65
        self.stem = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(in_channels),
                                  nn.SiLU(True),
                                  nn.Dropout(dropout))
        self.board_out = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        self.pass_norm = nn.RMSNorm(in_channels, eps=1e-5)
        self.pass_fc = nn.Linear(in_channels, 1)

    def forward(self, x, action_mask):
        x = self.stem(x)
        board_logits = self.board_out(x).flatten(start_dim=1)
        pass_feat = self.pass_norm(x.mean(dim=(2, 3)))
        pass_logit = self.pass_fc(pass_feat)
        logits = torch.cat([board_logits, pass_logit], dim=1)
        logits = logits.masked_fill(~action_mask, -1e9)
        return nn.functional.log_softmax(logits, dim=-1)

    def reset_output_parameters(self):
        nn.init.constant_(self.board_out.weight, 0)
        if self.board_out.bias is not None:
            nn.init.constant_(self.board_out.bias, 0)
        nn.init.constant_(self.pass_fc.weight, 0)
        if self.pass_fc.bias is not None:
            nn.init.constant_(self.pass_fc.bias, 0)


class TriHead(nn.Module):
    def __init__(self, in_channels, dropout=0.0):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.value_out = nn.Sequential(nn.BatchNorm2d(in_channels),
                                       nn.SiLU(True),
                                       nn.Dropout(dropout),
                                       nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, bias=False),
                                       nn.BatchNorm2d(in_channels),
                                       nn.SiLU(True),
                                       nn.Dropout2d(dropout),
                                       nn.Conv2d(in_channels, in_channels, kernel_size=3, bias=False),
                                       nn.BatchNorm2d(in_channels),
                                       nn.SiLU(True),
                                       nn.Dropout2d(dropout),
                                       nn.Flatten(),
                                       nn.Linear(in_channels, 3))
        self.aux_out = nn.Sequential(nn.Flatten(),
                                     nn.SiLU(True),
                                     nn.RMSNorm(3 * 8 * 8, eps=1e-5),
                                     nn.Linear(3 * 8 * 8, 1))

    def forward(self, x):
        h = self.stem(x)
        ownership_logit = h[:, :3]
        value = nn.functional.log_softmax(self.value_out(h), dim=-1)
        ownership = nn.functional.log_softmax(ownership_logit, dim=1)
        aux = torch.tanh(self.aux_out(ownership_logit).squeeze(-1))
        return value, aux, ownership

    def reset_output_parameters(self):
        nn.init.constant_(self.value_out[-1].weight, 0)
        if self.value_out[-1].bias is not None:
            nn.init.constant_(self.value_out[-1].bias, 0)
        nn.init.constant_(self.aux_out[-1].weight, 0)
        if self.aux_out[-1].bias is not None:
            nn.init.constant_(self.aux_out[-1].bias, 0)


class CNN(Base):
    aux_target_offset = 64
    score_scale = 8.0
    ownership_loss_weight = 1.0
    ownership_class_order = ('empty', 'own', 'opp')

    def __init__(
        self,
        lr,
        embed_dim=32,
        h_dim=256,
        out_dim=65,
        dropout=0.2,
        device='cpu',
        num_res_blocks=3,
        policy_lr_scale=0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_dim = 3
        self.device = device
        self.n_actions = out_dim

        self.piece_emb = nn.Embedding(2, embed_dim)
        self.pos_emb = nn.Embedding(10, embed_dim)
        self.legal_emb = nn.Embedding(2, embed_dim)
        self.register_buffer('orbit_map', torch.tensor(_ORBIT_MAP, dtype=torch.long))

        self.stem = nn.Sequential(
            nn.Conv2d(embed_dim, h_dim, kernel_size=3, padding=2, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.SiLU(inplace=True),
            *[ResidualBlock(h_dim, dropout=dropout) for _ in range(num_res_blocks)],
            nn.Conv2d(h_dim, h_dim, kernel_size=3, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.SiLU(True),
            nn.Dropout2d(dropout)
        )

        self.policy_head = PolicyHead(h_dim, out_dim, dropout)
        self.tri_head = TriHead(h_dim, dropout)

        self.apply(self.init_weights)
        self.policy_head.reset_output_parameters()
        self.tri_head.reset_output_parameters()

        self.opt = torch.optim.AdamW(
            [
                {'params': self.stem.parameters()},
                {'params': self.tri_head.parameters()},
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
            raise ValueError('Othello CNN requires action_mask from the environment.')
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.from_numpy(action_mask)
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        return action_mask.to(device=device, dtype=torch.bool)

    def _to_state_tensor(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        return state.to(self.device, dtype=torch.float32)

    def _embed_state(self, state, action_mask):
        batch_size = state.size(0)
        own = state[:, 0].view(batch_size, 64) > 0.5
        opp = state[:, 1].view(batch_size, 64) > 0.5
        empty = ~(own | opp)
        x = self.pos_emb(self.orbit_map).unsqueeze(0).expand(batch_size, -1, -1).clone()
        x = x + own.unsqueeze(-1) * self.piece_emb.weight[0]
        x = x + opp.unsqueeze(-1) * self.piece_emb.weight[1]
        board_legal = action_mask[:, :64].to(dtype=torch.long)
        x = x + empty.unsqueeze(-1) * self.legal_emb(board_legal)
        return x.transpose(1, 2).reshape(batch_size, self.embed_dim, 8, 8)

    def forward(self, x, action_mask):
        action_mask = self._normalize_action_mask(action_mask, x.device)
        x = self._embed_state(x, action_mask=action_mask)
        hidden = self.stem(x)
        log_prob = self.policy_head(hidden, action_mask)
        value, aux, ownership = self.tri_head(hidden)
        return log_prob, value, aux, ownership

    @torch.no_grad()
    def policy(self, state, action_mask):
        state = self._to_state_tensor(state)
        action_mask = self._normalize_action_mask(action_mask, state.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device != 'cpu'):
            return self(state, action_mask=action_mask)[0].exp().cpu().numpy()

    @torch.no_grad()
    def value(self, state, action_mask):
        state = self._to_state_tensor(state)
        action_mask = self._normalize_action_mask(action_mask, state.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device != 'cpu'):
            return self(state, action_mask=action_mask)[1].exp().cpu().numpy()

    @torch.no_grad()
    def ownership(self, state, action_mask):
        state = self._to_state_tensor(state)
        action_mask = self._normalize_action_mask(action_mask, state.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device != 'cpu'):
            ownership_log_prob = self(state, action_mask=action_mask)[3]
        return ownership_log_prob.exp().permute(0, 2, 3, 1).cpu().numpy()

    @torch.no_grad()
    def predict(self, state, action_mask, return_ownership=False):
        tensor = torch.from_numpy(state) if isinstance(state, np.ndarray) else state
        if self.device != 'cpu':
            tensor = tensor.pin_memory().to(self.device, dtype=torch.float32, non_blocking=True)
        else:
            tensor = tensor.to(self.device, dtype=torch.float32)
        action_mask = self._normalize_action_mask(action_mask, tensor.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device != 'cpu'):
            log_prob, value_log_prob, aux_scalar, ownership_log_prob = self(tensor, action_mask=action_mask)
        wdl = value_log_prob.exp()

        disc_diff = aux_scalar * float(self.aux_target_offset)
        score_scale = getattr(self, 'score_scale', 8.0)
        expected_utility = torch.atan(disc_diff / score_scale) * (2.0 / math.pi)

        if self.device != 'cpu':
            policy = log_prob.float().exp().to('cpu', non_blocking=True)
            value = wdl.float().to('cpu', non_blocking=True)
            utility = expected_utility.float().view(-1, 1).to('cpu', non_blocking=True)
            ownership = None
            if return_ownership:
                ownership = ownership_log_prob.float().exp().permute(0, 2, 3, 1).to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            if return_ownership:
                return policy.numpy(), value.numpy(), utility.numpy(), ownership.numpy()
            return policy.numpy(), value.numpy(), utility.numpy()
        result = (
            log_prob.exp().cpu().numpy(),
            wdl.cpu().numpy(),
            expected_utility.cpu().view(-1, 1).numpy(),
        )
        if return_ownership:
            return result + (ownership_log_prob.exp().permute(0, 2, 3, 1).cpu().numpy(),)
        return result
