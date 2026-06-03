import ast
import unittest
from pathlib import Path


class OthelloLinearPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network_path = Path("src/environments/Othello/Network.py")
        cls.network_src = cls.network_path.read_text()
        cls.network_tree = ast.parse(cls.network_src)

    def _class_node(self, name):
        for node in self.network_tree.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        self.fail(f"class {name} not found")

    def _method_node(self, class_name, method_name):
        class_node = self._class_node(class_name)
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                return node
        self.fail(f"{class_name}.{method_name} not found")

    def test_policy_head_uses_linear_logits_without_legal_embedding(self):
        self.assertNotIn("self.legal_emb", self.network_src)
        self.assertIn("self.piece_emb = nn.Embedding(3, embed_dim)", self.network_src)
        self.assertIn("self.fc = nn.Linear(flat_dim, out_dim)", self.network_src)
        self.assertNotIn("self.board_out", self.network_src)
        self.assertNotIn("self.pass_fc", self.network_src)

    def test_predict_works_without_action_mask(self):
        predict_node = self._method_node("CNN", "predict")
        action_arg_index = [arg.arg for arg in predict_node.args.args].index("action_mask")
        default_index = action_arg_index - (
            len(predict_node.args.args) - len(predict_node.args.defaults)
        )
        self.assertGreaterEqual(default_index, 0)
        self.assertIsInstance(predict_node.args.defaults[default_index], ast.Constant)
        self.assertIsNone(predict_node.args.defaults[default_index].value)

        normalize_src = ast.get_source_segment(
            self.network_src,
            self._method_node("CNN", "_normalize_action_mask"),
        )
        self.assertIn("return None", normalize_src)
        self.assertNotIn("requires action_mask", normalize_src)

    def test_actor_paths_do_not_pass_action_mask_to_network(self):
        mcts_src = Path("src/MCTS_cpp.py").read_text()
        player_src = Path("src/player.py").read_text()

        self.assertNotIn("action_mask=", mcts_src)
        self.assertNotIn("valid_mask", mcts_src)
        self.assertNotIn("action_mask=", player_src)


if __name__ == "__main__":
    unittest.main()
