import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageGrab
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageTk
from model import CNNModel

model = CNNModel()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

class DrawApp:
    def __init__(self, master):
        self.master = master
        self.master.title("MNIST CNN")
        
        # 内存图像：280x280 白色背景
        self.image = Image.new('L', (280, 280), 255)  # 255 = 白色
        self.draw = ImageDraw.Draw(self.image)
        
        # 画布（仅用于显示，不作为截图来源）
        self.canvas = tk.Canvas(master, width=280, height=280, bg='white')
        self.canvas.pack()
        
        # 按钮
        self.button_predict = tk.Button(master, text="Predict", command=self.predict)
        self.button_predict.pack(side='left')
        
        self.button_clear = tk.Button(master, text="Clear", command=self.clear)
        self.button_clear.pack(side='left')
        
        self.button_debug = tk.Button(master, text="Debug", command=self.debug_visuals)
        self.button_debug.pack(side='right')
        
        # 绘图绑定
        self.last_x, self.last_y = None, None
        self.canvas.bind('<B1-Motion>', self.on_draw)
        self.canvas.bind('<ButtonRelease-1>', self.reset_last)
        
        # # 加载模型
        # self.model = MNISTCNN()
        # self.model.load_state_dict(torch.load('mnist_model.pth', map_location='cpu'))
        # self.model.eval()
    
    def on_draw(self, event):
        """鼠标拖动时同时在画布和内存图像上画线"""
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            # 在画布上画线（用于显示）
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=15,
                fill='black',
                capstyle='round',
                smooth=True
            )
            # 在内存图像上画线（用于预测）
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=0,       # 0 = 黑色
                width=15
            )
        self.last_x, self.last_y = x, y
    
    def reset_last(self, event):
        self.last_x, self.last_y = None, None
    
    def clear(self):
        """清除画布和内存图像"""
        self.canvas.delete('all')
        self.image = Image.new('L', (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
    
    def get_processed_image(self):
        """从内存图像直接获取模型输入（无需截图）"""
        img_resized = self.image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        # 模型期望：白底黑字，值范围[-1,1]，形状(1,1,28,28)
        img_array = 1.0 - img_array
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = (img_tensor - 0.5) / 0.5
        return img_array, img_tensor
    
    def predict(self):
        """执行预测并显示结果"""
        _, img_tensor = self.get_processed_image()
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            prob = torch.nn.functional.softmax(output, dim=1)[0][pred].item()
        print(f"Prediction: {pred} (confidence: {prob:.2%})")
        # 可选：弹窗显示
        # tk.messagebox.showinfo("Result", f"Predicted: {pred}")
    
    def debug_visuals(self):
        img_array, img_tensor = self.get_processed_image()
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            x = model.conv1(img_tensor)
            x = F.relu(x).squeeze(0)  # shape [32,28,28]
    
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])
        
        ax1 = fig.add_subplot(gs[0,0])
        ax1.imshow(img_array, cmap="gray")
        ax1.set_title("Input image (28x28)")
        ax1.axis("off")
        
        ax2 = fig.add_subplot(gs[0,1])
        ax2.bar(range(10), probs)
        ax2.set_xlabel("Digits")
        ax2.set_ylabel("Probability")
        ax2.set_title("Prediction probabilities")
        
        num_features = min(32, x.shape[0])
        cols = 8
        rows = int(np.ceil(num_features / cols))

        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_title("First Conv Layer Feature Maps")
        ax3.axis("off")

        for i in range(num_features):
            row_idx = i // cols
            col_idx = i % cols
            ax_inset = ax3.inset_axes((
                col_idx / cols,              # x0
                1 - (row_idx + 1) / rows,    # y0
                1 / cols,                    # width
                1 / rows                     # height
            ))
            ax_inset.imshow(x[i].cpu(), cmap="gray")
            ax_inset.axis("off")

        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
