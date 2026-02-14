import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# 导入模型和环境
from src.environments.Connect4 import CNN
from src.environments.Connect4.env import Env
# 关键：导入定义 Block 的类，以便我们能在遍历时识别它们
from src.environments.Connect4.Network import ResidualGLUConv2d, ConvGLU

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = [] 

        # 注册 hook
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
        # grad_output 是一个 tuple，通常第一个元素是对于输出的梯度
        self.gradients = grad_output[0]
    
    def generate(self, board, class_idx=None):
        if board.ndim == 3:
            board = board[np.newaxis, ...]
            
        board_tensor = torch.from_numpy(board).to(self.model.device)
        prob, value = self.model(board_tensor)

        if class_idx is None:
            class_idx = prob.argmax().item()
        
        self.model.zero_grad()
        prob[0, class_idx].backward()
        
        if self.gradients is None or self.activations is None:
            return None

        # 对特征图的每个通道计算权重 (Global Average Pooling)
        # gradients shape: [1, C, H, W] -> weights shape: [1, C, 1, 1]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # 加权求和
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam) # ReLU 去除负激活

        cam = cam.detach().cpu().numpy()[0, 0]
        
        # Resize 到棋盘大小
        target_shape = (board.shape[3], board.shape[2])
        cam = cv2.resize(cam, target_shape)

        # 归一化
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
        plt.xticks(range(7))
        plt.yticks(range(6))
        
        if show_values:
            for i in range(heatmap.shape[0]):
                for j in range(heatmap.shape[1]):
                    value = heatmap[i, j]
                    text_color = 'white' if value > 0.7 or value < 0.2 else 'black'
                    font_weight = 'bold' if value > 0.7 else 'normal'
                    if value >= 0.01:
                        text = f'{value:.2f}'
                        plt.text(j, i, text, 
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
    """
    遍历模型，寻找所有的 ResidualGLUConv2d 模块。
    这些是网络中的主要卷积块。
    """
    target_types = (ConvGLU, torch.nn.Conv2d) # 你也可以把 ConvGLU 加进去: (ResidualGLUConv2d, ConvGLU)
    
    for name, module in model.named_modules():
        if isinstance(module, target_types):
            yield name, module

if __name__ == "__main__":
    # 1. 初始化
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN(lr=0, in_dim=3, h_dim=32, out_dim=7, dropout=0.05, device=device)
    
    try:
        # 请确保这里的路径是正确的
        checkpoint = torch.load("./params/AZ_Connect4_CNN_best.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded pretrained weights")
    except Exception as e:
        print(f"Using random weights (Reason: {e})")
    model.eval()
    
    env = Env()
    # 模拟一步棋 (例如中间列) 以产生有意义的激活
    # env.step(3)
    
    board = env.current_state()
    
    pred, value = model.predict(board)
    pred_class = pred.argmax().item()
    print(f"Predicted class: {pred_class}")

    # 2. 创建保存目录
    save_dir = "heatmaps"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving heatmaps to '{save_dir}/' ...")

    # 3. 遍历所有 Block 并生成热力图
    blocks = list(get_target_blocks(model))
    print(f"Found {len(blocks)} target blocks.")
    
    for i, (name, layer) in enumerate(blocks):
        gradcam = GradCAM(model, layer)
        
        try:
            heatmap = gradcam.generate(board, pred_class)
            
            if heatmap is not None:
                # 格式化文件名：hidden.0 -> hidden_0
                safe_name = name.replace('.', '_')
                # 标注这是 Block 的输出
                file_name = f"Block_{safe_name}.png"
                save_path = os.path.join(save_dir, file_name)
                
                # 标题包含类名，方便识别是哪个Block类型
                layer_type = layer.__class__.__name__
                title = f"Layer: {name}\nType: {layer_type} | Class: {pred_class}"
                
                gradcam.visualize(heatmap, title=title, show_values=True, save_path=save_path)
            else:
                print(f"Skipping {name}: No heatmap generated")
                
        except Exception as e:
            print(f"Error processing {name}: {e}")
        finally:
            gradcam.remove_hooks()
            
    print("\nDone! Check the folder.")