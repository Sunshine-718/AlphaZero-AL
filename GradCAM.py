import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.environments.Connect4 import CNN
from src.environments.Connect4.env import Env


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate(self, board, class_idx=None):
        board = torch.from_numpy(board).to(self.model.device)
        prob, value = self.model(board)

        if class_idx is None:
            class_idx = prob.argmax().item()
        
        self.model.zero_grad()
        prob[0, class_idx].backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.detach().cpu().numpy()[0, 0]
        cam = cv2.resize(cam, (board.shape[3], board.shape[2]))

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    
    def visualize(self, heatmap, title="Grad-CAM Heatmap", show_values=True):
        """可视化热力图，可选项显示数值"""
        plt.figure(figsize=(10, 8))
        
        # 显示热力图
        im = plt.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(im, label='Activation Intensity')
        plt.title(title)
        
        # 显示行列坐标
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.xticks(range(7))
        plt.yticks(range(6))
        
        # 在每个格子上显示数值
        if show_values:
            for i in range(heatmap.shape[0]):  # 行
                for j in range(heatmap.shape[1]):  # 列
                    value = heatmap[i, j]
                    
                    # 智能文本颜色和样式选择
                    if value > 0.7:  # 高激活值 - 白色粗体
                        text_color = 'white'
                        font_weight = 'bold'
                        font_size = 10
                    elif value > 0.4:  # 中等激活值 - 白色正常
                        text_color = 'white'
                        font_weight = 'normal'
                        font_size = 9
                    else:  # 低激活值 - 黑色
                        text_color = 'black'
                        font_weight = 'normal'
                        font_size = 8
                    
                    # 智能数值格式化
                    if value >= 0.01:
                        text = f'{value:.3f}'
                    elif value >= 0.001:
                        text = f'{value:.4f}'
                    else:
                        text = f'{value:.2e}'
                    
                    plt.text(j, i, text, 
                            ha='center', va='center',
                            color=text_color, fontsize=font_size,
                            fontweight=font_weight)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    model = CNN(lr=0, in_dim=3, h_dim=32, out_dim=7, dropout=0.05, device="cuda")
    print(model)
    try:
        checkpoint = torch.load("./params/AZ_Connect4_CNN_current.pt", map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded pretrained weights")
    except:
        print("Using random weights")
    model.eval()
    
    target_layer = model.policy_head[0].conv
    print(f"Target layer: {target_layer}")
    
    gradcam = GradCAM(model, target_layer)

    env = Env()
    env.step(3)
    env.step(3)

    board = env.current_state()
    
    pred, value = model.predict(board)
    pred_class = pred.argmax().item()
    # pred_class = 3
    print(f"Predicted column: {pred_class}")

    heatmap = gradcam.generate(board, pred_class)
    gradcam.visualize(heatmap, f"Grad-CAM for class {pred_class}", show_values=True)

    col_activations = np.mean(heatmap, axis=0)
    for i, act in enumerate(col_activations):
        print(f"  Column {i}: {act:.3f}")
