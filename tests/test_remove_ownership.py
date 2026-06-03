import os
import tempfile
import unittest

import torch

from src.ReplayBuffer import ReplayBuffer
from src.environments.Connect4 import utils as connect4_utils
from src.environments.Othello import utils as othello_utils
from src.environments.Othello.Network import CNN


def _batch(rows, cols, action_dim):
    n = 2
    return (
        torch.zeros((n, 3, rows, cols), dtype=torch.float32),
        torch.full((n, action_dim), 1.0 / action_dim, dtype=torch.float32),
        torch.zeros((n, 1), dtype=torch.int8),
        torch.ones((n, 1), dtype=torch.int16),
        torch.zeros((n, 1), dtype=torch.int16),
        torch.zeros((n, 3), dtype=torch.float32),
        torch.ones((n, action_dim), dtype=torch.bool),
        torch.zeros((n, 3), dtype=torch.float32),
    )


class RemoveOwnershipTests(unittest.TestCase):
    def test_othello_network_has_three_outputs(self):
        net = CNN(lr=0.001, embed_dim=4, h_dim=8, dropout=0.0, device='cpu', num_res_blocks=1)
        state = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
        action_mask = torch.ones((2, 65), dtype=torch.bool)

        outputs = net(state, action_mask=action_mask)
        self.assertEqual(3, len(outputs))

        policy, value, utility = net.predict(state.numpy(), action_mask=action_mask.numpy())
        self.assertEqual((2, 65), policy.shape)
        self.assertEqual((2, 3), value.shape)
        self.assertEqual((2, 1), utility.shape)
        self.assertFalse(hasattr(net, 'ownership'))

    def test_replay_buffer_samples_eight_tensors_and_ignores_old_ownership_key(self):
        buffer = ReplayBuffer(3, capacity=4, action_dim=7, row=6, col=7, replay_ratio=1.0)
        buffer.store(
            torch.zeros((3, 6, 7), dtype=torch.float32),
            torch.full((7,), 1.0 / 7, dtype=torch.float32),
            0,
            valid_mask=torch.ones(7, dtype=torch.bool),
            future_root_wdl=torch.zeros(3, dtype=torch.float32),
        )
        self.assertEqual(8, len(buffer.sample(1).dataset.tensors))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'buffer.pt')
            state_dict = buffer.state_dict() if hasattr(buffer, 'state_dict') else {
                'state': buffer.state,
                'prob': buffer.prob,
                'winner': buffer.winner,
                'steps_to_end': buffer.steps_to_end,
                'aux_target': buffer.aux_target,
                'root_wdl': buffer.root_wdl,
                'valid_mask': buffer.valid_mask,
                'future_root_wdl': buffer.future_root_wdl,
                'ownership_target': torch.zeros((4, 6, 7), dtype=torch.int8),
                '_ptr': buffer._ptr,
                'current_capacity': buffer.current_capacity,
            }
            torch.save(state_dict, path)

            loaded = ReplayBuffer(3, capacity=4, action_dim=7, row=6, col=7, replay_ratio=1.0)
            loaded.load(path)
            self.assertEqual(8, len(loaded.sample(1).dataset.tensors))

    def test_augment_functions_keep_eight_tensor_batch_contract(self):
        self.assertEqual(8, len(connect4_utils.augment(_batch(6, 7, 7))))
        self.assertEqual(8, len(othello_utils.augment(_batch(8, 8, 65))))


if __name__ == '__main__':
    unittest.main()
