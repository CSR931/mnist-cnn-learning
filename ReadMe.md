# MNIST CNN

![Python](https://img.shields.io/badge/python-3.13-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0-orange)
![Mnist](https://img.shields.io/badge/mnist-dataset-white)
![Accuracy](https://img.shields.io/badge/accuracy-98%25-brightgreen)
![Epochs](https://img.shields.io/badge/epochs-50-blue)
![Epochs](https://img.shields.io/badge/Streamlit-Online-red)

![mnist](images/mnist.png)

## Overview

**MNIST CNN** is an interactive handwritten digit recognition project that demonstrates the power of convolutional neural networks (CNNs) using **PyTorch**. The system allows users to draw digits, see real-time predictions, and explore the internal workings of the model through visualizations.

Key highlights:

- CNN achieves **high accuracy** on MNIST (over 98% on test set).
- Interactive **Tkinter GUI** for drawing digits.
- Visualization of **input images**, **prediction probabilities**, and **feature maps** from the first convolutional layer.
- Graphs show **training progress** over epochs (accuracy and loss).

---

## The MNIST CNN Online

Want to see the MNIST digit recognizer in action?

You can **draw your own digits** or upload an image, and the model will predict what number you wrote in real-time. It even shows:

- The **probabilities for each digit**
- The **processed input image**
- The **feature maps** from the first convolutional layer

Check it out here: [https://aitaneuh-mnist-cnn.streamlit.app/](https://aitaneuh-mnist-cnn.streamlit.app/)

> **Tip:** Use a white brush on a black background for best results, just like MNIST images.

---

## Project Structure

```text
├── model.py        # CNN architecture
├── train.py        # Training script with graphs
├── draw.py         # GUI for drawing digits and visualization
├── graphs/         # Example plots (accuracy, loss, feature maps)
└── README.md
```

---
## <mark>学习成果<mark>
bash
```
cd mnist-cnn-learning
python -m venv venv
source venv/Scripts/activate

如果SSL证书问题：pip config set global.index-url https://tsinghua.edu.cn/simple
pip install -r requirements.txt
```
修改代码后
```
git add .
git commit -m "..."
git push origin main
```

### model.py
```
[输入图片] ──> [卷积+池化] ──> [卷积+池化] ──> [拉平展维] ──> [全连接] ──> [输出结果]
 28x28       提取边缘特征      提取数字部件      变成长列表       综合判断       0~9的得分
 1个通道     (32个通道)       (64个通道)       3136个数值      128个特征     10个分数
```

**概念辨析：**

尺寸：图片的长宽。（H, W）

通道：不同层，每层以不同视角扫描图片，关注一种特征（横线、竖线、圆圈...）。（通道数，高度，宽度）

维度：需要多少个数字才能唯一确定一个数据点。	3维（通道数，高度，宽度）+batch=4维（批次，通道，高，宽）


### draw.py
### <mark>解决**原仓库代码识别率极低**，只会猜测0和8问题：<mark>
问题示例
<img width="1110" height="462" alt="89f8f9fc4c1cdae0d5829c55865ee6eb" src="https://github.com/user-attachments/assets/4e7532d1-ce49-4e59-ae3a-cd4a921945bd" />


1.Windows 系统在高 DPI 缩放（如 125%、150%）下，winfo_rootx() 返回的逻辑坐标与实际物理像素坐标不一致。
直接使用 winfo_rootx() + canvas.winfo_x() 会导致截图区域向左/上偏移，截到窗口边框。
```
def get_processed_image(self):
        """从内存图像直接获取模型输入（无需截图）"""
        img_resized = self.image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        # 模型期望：白底黑字，值范围[-1,1]，形状(1,1,28,28)
        img_array = 1.0 - img_array
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = (img_tensor - 0.5) / 0.5
        return img_array, img_tensor
```
改为内存绘图，只记录笔迹，不再出现窗口边缘

img_tensor = (img_tensor - 0.5) / 0.5对齐训练时的归一化参数


2.画的是黑底白字，训练学的是白底黑字
```
self.canvas = tk.Canvas(master, width=280, height=280, bg='white')  # 白底
# ...
self.canvas.create_line(..., fill='black', ...)  # 黑笔
```
```
img_array = 1.0 - img_array
```




---

## Features

### Interactive Drawing & Prediction

- Draw digits directly on a 280x280 canvas.
- Real-time prediction of digits 0–9.
- Clear the canvas at any time to draw a new digit.

### Model Visualizations (Debug Mode)

- **Input image** as seen by the CNN.
- **Softmax probabilities** for all 10 digits.
- **Feature maps** from the first convolutional layer (reveals learned patterns).

### Training Progress

- Track **accuracy** and **loss** during training.
- Graphs illustrate how the CNN improves over epochs.

---

## Results & Visualizations

### Loss and Accuracy over 50 Epochs

![Accuracy Graph](graphs/training.png)

### Debug Window (with number 1)

![Debug Window](graphs/debug.png)

### Tkinter GUI Screenshot

![GUI Screenshot](graphs/gui.png)

---

## Learning Outcomes

By exploring this project, you will understand:

- How to implement and train a CNN in PyTorch.
- How to preprocess images for model input.
- How to integrate a PyTorch model with a **Tkinter GUI**.
- How to visualize **CNN internals** and **prediction probabilities**.
- How to interpret **training graphs** and model performance.

---

## Installation

```bash
git clone https://github.com/Aitaneuh/mnist-cnn.git
cd mnist-cnn
pip install -r requirements.txt
python train.py      # Train the model
python draw.py       # Launch the drawing GUI
```

---
