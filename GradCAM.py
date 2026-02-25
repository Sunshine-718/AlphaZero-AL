import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from src.environments.Connect4 import CNN
from src.environments.Connect4.env import Env
from src.environments.Connect4.Network import ResidualBlock


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []

        h1 = target_layer.register_forward_hook(self._forward_hook)
        h2 = target_layer.register_full_backward_hook(self._backward_hook)
        self.handlers.append(h1)
        self.handlers.append(h2)

    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()
        self.handlers = []

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, board, head='policy', class_idx=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            board: numpy array, shape (C, H, W) or (1, C, H, W)
            head: 'policy' | 'value' | 'steps'
            class_idx: target class index for backward.
                - policy: action index (0-6), default argmax
                - value: 0=win, 1=draw, 2=loss, default argmax
                - steps: 0-42, default argmax
        """
        if board.ndim == 3:
            board = board[np.newaxis, ...]

        board_tensor = torch.from_numpy(board).float().to(self.model.device)
        log_prob, value_logprob, steps_logprob = self.model(board_tensor)

        if head == 'policy':
            target = log_prob
        elif head == 'value':
            target = value_logprob
        elif head == 'steps':
            target = steps_logprob
        else:
            raise ValueError(f"Unknown head: {head!r}, expected 'policy', 'value', or 'steps'")

        if class_idx is None:
            class_idx = target[0].argmax().item()

        self.model.zero_grad()
        target[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            return None

        # Global Average Pooling over spatial dims -> channel weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()[0, 0]  # (feat_H, feat_W), e.g. (8, 9)

        # Crop center to match board spatial dims (padding border has no board correspondence)
        board_h, board_w = board.shape[2], board.shape[3]
        feat_h, feat_w = cam.shape
        crop_top = (feat_h - board_h) // 2
        crop_left = (feat_w - board_w) // 2
        cam = cam[crop_top:crop_top + board_h, crop_left:crop_left + board_w]

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def visualize(self, heatmap, title="Grad-CAM", show_values=True, save_path=None):
        if heatmap is None:
            return

        plt.figure(figsize=(6, 5))

        im = plt.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(im, label='Activation Intensity')
        plt.title(title)

        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.xticks(range(heatmap.shape[1]))
        plt.yticks(range(heatmap.shape[0]))

        if show_values:
            for i in range(heatmap.shape[0]):
                for j in range(heatmap.shape[1]):
                    value = heatmap[i, j]
                    text_color = 'white' if value > 0.7 or value < 0.2 else 'black'
                    font_weight = 'bold' if value > 0.7 else 'normal'
                    if value >= 0.01:
                        plt.text(j, i, f'{value:.2f}',
                                 ha='center', va='center',
                                 color=text_color, fontsize=8,
                                 fontweight=font_weight)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Saved: {save_path}")
            plt.close()
        else:
            plt.show()


def get_target_blocks(model):
    """Yield all ResidualBlock and Conv2d modules in the model."""
    for name, module in model.named_modules():
        if isinstance(module, (ResidualBlock, torch.nn.Conv2d)):
            yield name, module


HEAD_LABELS = {
    'policy': 'Action',
    'value':  'Value (0=W 1=D 2=L)',
    'steps':  'Steps left',
}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN(lr=0, device=device)

    try:
        checkpoint = torch.load("./params/AZ_Connect4_CNN_current.pt",
                                map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded pretrained weights")
    except Exception as e:
        print(f"Using random weights (Reason: {e})")
    model.eval()

    env = Env()
    board = env.current_state()

    pred, value, steps = model.predict(board)
    pred_class = pred.argmax()
    print(f"Predicted action: {pred_class}, value: {value.item():.3f}, "
          f"steps: {steps.item():.1f}")

    save_dir = "heatmaps"
    os.makedirs(save_dir, exist_ok=True)

    blocks = list(get_target_blocks(model))
    print(f"Found {len(blocks)} target layers.")

    for head in ('policy', 'value', 'steps'):
        head_dir = os.path.join(save_dir, head)
        os.makedirs(head_dir, exist_ok=True)

        for i, (name, layer) in enumerate(blocks):
            gradcam = GradCAM(model, layer)
            try:
                heatmap = gradcam.generate(board, head=head)
                if heatmap is not None:
                    safe_name = name.replace('.', '_')
                    file_name = f"{safe_name}.png"
                    save_path = os.path.join(head_dir, file_name)

                    layer_type = layer.__class__.__name__
                    title = (f"[{head}] {name} ({layer_type})\n"
                             f"{HEAD_LABELS[head]}: {pred_class if head == 'policy' else ''}")
                    gradcam.visualize(heatmap, title=title, show_values=True,
                                     save_path=save_path)
                else:
                    print(f"  Skipping {name}: no heatmap")
            except Exception as e:
                print(f"  Error {name}: {e}")
            finally:
                gradcam.remove_hooks()

        print(f"  [{head}] done.")

    print(f"\nAll heatmaps saved to '{save_dir}/'.")
