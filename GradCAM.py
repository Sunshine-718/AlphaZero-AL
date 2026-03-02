import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from src.environments.Othello import CNN, Env
from src.environments.Othello.Network import ResidualBlock


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []
        self._disabled_inplace = []

        h1 = target_layer.register_forward_hook(self._forward_hook)
        h2 = target_layer.register_full_backward_hook(self._backward_hook)
        self.handlers.append(h1)
        self.handlers.append(h2)

        # register_full_backward_hook wraps the module output as a view;
        # if the next layer is SiLU(inplace=True) it modifies that view
        # inplace → autograd error.  Temporarily disable inplace.
        for module in model.modules():
            if isinstance(module, torch.nn.Sequential):
                children = list(module.children())
                for i, child in enumerate(children):
                    if child is target_layer and i + 1 < len(children):
                        nxt = children[i + 1]
                        if hasattr(nxt, 'inplace') and nxt.inplace:
                            nxt.inplace = False
                            self._disabled_inplace.append(nxt)

    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()
        self.handlers = []
        for m in self._disabled_inplace:
            m.inplace = True
        self._disabled_inplace = []

    def _forward_hook(self, module, input, output):
        self.activations = output.clone()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, board, head='policy', class_idx=None, include_padding=False):
        """
        Generate Grad-CAM heatmap.

        Args:
            board: numpy array, shape (C, H, W) or (1, C, H, W)
            head: 'policy' | 'value' | 'steps'
            class_idx: target class index for backward.
                - policy: action index (0-6), default argmax
                - value: 0=win, 1=draw, 2=loss, default argmax
                - steps: 0-42, default argmax
            include_padding: if True, manually pad the input and remove the
                first conv's internal padding so the heatmap covers the full
                padded input (e.g. 12x12 instead of 8x8 for Othello).
        """
        if board.ndim == 3:
            board = board[np.newaxis, ...]

        board_tensor = torch.from_numpy(board).float().to(self.model.device)

        first_conv = self.model.hidden[0]
        padded_h = padded_w = 0
        if include_padding:
            py, px = first_conv.padding
            # Save turn indicator before padding (position (0,0) will land in padding zone)
            turn_val = board_tensor[:, 2, 0, 0].clone()
            board_tensor = F.pad(board_tensor, (px, px, py, py), value=0)
            padded_h, padded_w = board_tensor.shape[2], board_tensor.shape[3]
            # Restore turn value at (0,0) so _route_policy reads the correct player
            board_tensor[:, 2, 0, 0] = turn_val
            first_conv.padding = (0, 0)

        try:
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

            if include_padding:
                # Upsample feature-map CAM to padded-input resolution
                cam = F.interpolate(cam, size=(padded_h, padded_w),
                                    mode='bilinear', align_corners=False)

            cam = cam.detach().cpu().numpy()[0, 0]

            if not include_padding:
                # Crop center to match board spatial dims
                board_h, board_w = board.shape[2], board.shape[3]
                feat_h, feat_w = cam.shape
                crop_top = (feat_h - board_h) // 2
                crop_left = (feat_w - board_w) // 2
                cam = cam[crop_top:crop_top + board_h, crop_left:crop_left + board_w]

            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam
        finally:
            if include_padding:
                first_conv.padding = (py, px)

    def visualize(self, heatmap, title="Grad-CAM", show_values=True, save_path=None,
                  board_shape=None):
        """
        Visualize heatmap.

        Args:
            board_shape: (H, W) of the actual board. When provided, draws a
                white dashed rectangle around the real board region inside
                the padded heatmap and dims the padding cells.
        """
        if heatmap is None:
            return

        plt.figure(figsize=(6, 5))
        ax = plt.gca()

        im = ax.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(im, label='Activation Intensity')
        ax.set_title(title)

        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_xticks(range(heatmap.shape[1]))
        ax.set_yticks(range(heatmap.shape[0]))

        # Draw board boundary and label padding cells
        if board_shape is not None:
            bh, bw = board_shape
            fh, fw = heatmap.shape
            pad_top = (fh - bh) // 2
            pad_left = (fw - bw) // 2
            from matplotlib.patches import Rectangle
            rect = Rectangle((pad_left - 0.5, pad_top - 0.5), bw, bh,
                              linewidth=2, edgecolor='white',
                              facecolor='none', linestyle='--')
            ax.add_patch(rect)
            # Dim padding cells with semi-transparent overlay
            for i in range(fh):
                for j in range(fw):
                    if i < pad_top or i >= pad_top + bh or j < pad_left or j >= pad_left + bw:
                        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               facecolor='black', alpha=0.35))
            # Custom tick labels: mark padding with 'p'
            xlabels = [f'p' if (c < pad_left or c >= pad_left + bw) else str(c - pad_left)
                        for c in range(fw)]
            ylabels = [f'p' if (r < pad_top or r >= pad_top + bh) else str(r - pad_top)
                        for r in range(fh)]
            ax.set_xticklabels(xlabels)
            ax.set_yticklabels(ylabels)

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
        checkpoint = torch.load("./params/AZ_Othello_CNN_current.pt",
                                map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded pretrained weights")
    except Exception as e:
        print(f"Using random weights (Reason: {e})")
    model.eval()

    env = Env()
    board = env.current_state()
    board_shape = (board.shape[2], board.shape[3])  # (H, W)

    pred, value, steps = model.predict(board)
    pred_class = pred.argmax()
    print(f"Predicted action: {pred_class}, value: {value}, "
          f"steps: {steps.item():.1f}")

    save_dir = "heatmaps"
    os.makedirs(save_dir, exist_ok=True)

    blocks = list(get_target_blocks(model))
    print(f"Found {len(blocks)} target layers.")

    for head in ('policy', 'value', 'steps'):
        for include_padding in (False, True):
            suffix = "_padded" if include_padding else ""
            head_dir = os.path.join(save_dir, head + suffix)
            os.makedirs(head_dir, exist_ok=True)

            for i, (name, layer) in enumerate(blocks):
                gradcam = GradCAM(model, layer)
                try:
                    heatmap = gradcam.generate(board, head=head,
                                               include_padding=include_padding)
                    if heatmap is not None:
                        safe_name = name.replace('.', '_')
                        file_name = f"{safe_name}.png"
                        save_path = os.path.join(head_dir, file_name)

                        layer_type = layer.__class__.__name__
                        pad_tag = " [+pad]" if include_padding else ""
                        title = (f"[{head}] {name} ({layer_type}){pad_tag}\n"
                                 f"{HEAD_LABELS[head]}: {pred_class if head == 'policy' else ''}")
                        gradcam.visualize(
                            heatmap, title=title, show_values=True,
                            save_path=save_path,
                            board_shape=board_shape if include_padding else None,
                        )
                    else:
                        print(f"  Skipping {name}: no heatmap")
                except Exception as e:
                    print(f"  Error {name}: {e}")
                finally:
                    gradcam.remove_hooks()

            print(f"  [{head}{suffix}] done.")

    print(f"\nAll heatmaps saved to '{save_dir}/'.")
