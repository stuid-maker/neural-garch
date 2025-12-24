# Neural-GARCH 模型

使用神经网络增强的 GARCH 模型，用于金融时间序列的条件波动率预测和风险管理。

## 模型简介

Neural-GARCH 模型结合了传统 GARCH 模型的统计框架和神经网络的非线性建模能力。与传统 GARCH 模型的线性参数形式不同，Neural-GARCH 使用神经网络来建模条件方差的递归演化：

\[
\sigma_t^2 = f_\theta(r_{t-1}^2, \sigma_{t-1}^2)
\]

其中 \( f_\theta(\cdot) \) 是一个可训练的神经网络。

### 核心特点

- ✅ **保留 GARCH 的金融解释性**：维持条件波动率的递归结构
- ✅ **非线性建模能力**：使用神经网络捕捉复杂的波动率动态
- ✅ **概率预测**：输出完整的条件分布，而非点预测
- ✅ **风险度量**：支持 VaR、CVaR 等风险指标计算
- ✅ **轻量级设计**：参数规模可控，避免过拟合

## 安装

### 环境要求

- Python 3.7+
- PyTorch 1.9+

### 安装步骤

1. 克隆或下载本项目

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 基本使用示例

运行示例代码：

```bash
python example.py
```

这将：
- 生成模拟的 GARCH(1,1) 数据
- 训练 Neural-GARCH 模型
- 进行波动率预测
- 计算风险指标（VaR、CVaR）
- 生成可视化结果

### 2. 训练模型

使用合成数据训练：

```bash
python train.py --data_type synthetic --n_samples 2000 --epochs 200
```

使用自己的数据文件训练：

```bash
python train.py --data_type file --data_path your_data.csv --epochs 100
```

**数据文件格式要求**：
- CSV 格式
- 包含价格列（默认列名：`Close`）
- 可选：包含日期列用于排序

### 3. 模型预测

使用训练好的模型进行预测：

```bash
python predict.py --model_path neural_garch_model.pth --data_path your_data.csv --n_steps 5
```

## 代码结构

```
.
├── neural_garch.py      # Neural-GARCH 模型核心实现
├── data_utils.py        # 数据生成和加载工具
├── train.py             # 模型训练主脚本
├── predict.py           # 预测和风险评估工具
├── example.py           # 基本使用示例
├── requirements.txt     # 依赖包列表
└── README.md           # 本文件
```

## 核心模块说明

### NeuralGARCH 类

主要的模型类，包含：

- `forward()`: 前向传播，预测条件方差
- `predict_volatility_sequence()`: 递归计算整个序列的条件方差
- `nll_loss()`: 负对数似然损失（MLE 目标函数）

### 训练函数

`train_neural_garch()`: 训练模型，支持：
- Adam / RMSprop 优化器
- 梯度裁剪（防止数值不稳定）
- 自定义训练轮数和学习率

### 预测函数

- `predict_next_volatility()`: 预测下一时刻条件方差
- `predict_conditional_distribution()`: 多步预测和蒙特卡洛模拟
- `calculate_risk_metrics()`: 计算 VaR 和 CVaR

## 模型参数说明

### 模型架构参数

- `input_dim`: 输入维度（默认 2：r_{t-1}^2 和 σ_{t-1}^2）
- `hidden_dims`: 隐藏层维度列表，例如 `[32, 16]`
- `activation`: 激活函数，`'relu'` 或 `'tanh'`
- `output_activation`: 输出激活函数，`'softplus'` 或 `'exp'`（确保输出为正）

### 训练参数

- `num_epochs`: 训练轮数（推荐 100-300）
- `learning_rate`: 学习率（推荐 0.001-0.01）
- `optimizer_type`: 优化器类型，`'adam'` 或 `'rmsprop'`
- `clip_grad_norm`: 梯度裁剪阈值（推荐 0.5-2.0）

## 损失函数

模型使用**极大似然估计（MLE）**作为训练目标，假设条件收益率分布为正态分布：

\[
r_t \mid \mathcal{F}_{t-1} \sim \mathcal{N}(0, \sigma_t^2)
\]

负对数似然损失：

\[
\mathcal{L} = \sum_t \left( \log \sigma_t^2 + \frac{r_t^2}{\sigma_t^2} \right)
\]

## 风险评估

模型支持计算以下风险指标：

### Value-at-Risk (VaR)

在给定置信水平下，预期最大损失：

\[
\text{VaR}_\alpha = z_\alpha \cdot \sigma_t
\]

其中 \( z_\alpha \) 是标准正态分布的分位数。

### Conditional VaR (CVaR / Expected Shortfall)

超过 VaR 的预期损失：

\[
\text{CVaR}_\alpha = -\sigma_t \cdot \frac{\phi(z_\alpha)}{1-\alpha}
\]

其中 \( \phi \) 是标准正态分布的密度函数。

## 使用示例

### Python API 使用

```python
import torch
from neural_garch import NeuralGARCH, train_neural_garch, predict_next_volatility
from data_utils import generate_garch_data, prepare_data

# 1. 准备数据
returns = generate_garch_data(n_samples=2000, seed=42)
train_returns, test_returns = prepare_data(returns, train_ratio=0.8)

# 2. 创建模型
model = NeuralGARCH(
    input_dim=2,
    hidden_dims=[32, 16],
    activation='relu',
    output_activation='softplus'
)

# 3. 训练模型
initial_sigma_squared = torch.var(train_returns, unbiased=False)
history = train_neural_garch(
    model=model,
    returns=train_returns,
    num_epochs=100,
    learning_rate=0.001
)

# 4. 预测
r_last = train_returns[-1].item()
sigma_last_squared = initial_sigma_squared.item()
sigma_next_squared = predict_next_volatility(model, r_last, sigma_last_squared)
print(f"预测波动率: {np.sqrt(sigma_next_squared):.6f}")
```

## 注意事项

1. **数据预处理**：模型会自动对收益率进行去均值处理，保留波动率信息。

2. **数值稳定性**：
   - 输出层使用 Softplus 或指数函数确保条件方差为正
   - 损失函数中加入小的 epsilon 防止除零
   - 训练时建议使用梯度裁剪

3. **递归计算**：模型是递归的，需要按时间顺序处理数据，不能打乱。

4. **初始化**：初始条件方差通常设为样本方差，这是一个合理的起点。

5. **模型复杂度**：建议使用轻量级网络（1-2层隐藏层，16-64个神经元），避免过拟合。

## 模型评估

模型训练完成后，建议评估：

- **训练损失**：负对数似然应该随着训练下降
- **预测波动率 vs 已实现波动率**：计算相关性评估预测质量
- **风险指标合理性**：检查 VaR 和 CVaR 是否在合理范围内

## 扩展方向

- **扩展输入**：可以加入更多滞后项、成交量等辅助变量
- **其他分布假设**：可以尝试 t 分布或其他分布
- **多资产建模**：扩展为多元 Neural-GARCH
- **长短期记忆**：尝试 LSTM 或 GRU 替代 MLP

## 参考文献

本实现基于以下理论框架：

- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics.
- 神经网络在金融建模中的应用

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，欢迎提出 Issue 或 Pull Request。

