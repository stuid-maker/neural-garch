# Neural-GARCH 模型

使用神经网络增强的 GARCH 模型，用于金融时间序列的条件波动率预测与风险管理。

---

## 模型简介

Neural-GARCH 模型结合了传统 GARCH 模型的统计结构与神经网络的非线性建模能力。  
不同于传统 GARCH 的线性参数形式，Neural-GARCH 使用神经网络来刻画条件方差的递归演化关系：

<p align="center">
  <img src="https://latex.codecogs.com/png.latex?\sigma_t^2=f_\theta(r_{t-1}^2,\sigma_{t-1}^2)" />
</p>

其中：

- **rₜ**：去均值后的收益率  
- **σₜ²**：条件方差  
- **f_θ(·)**：带参数 θ 的神经网络  

---

## 核心特点

- ✅ 保留 GARCH 的金融解释性（条件方差递归结构）
- ✅ 利用神经网络建模非线性波动率动态
- ✅ 输出条件分布而非点预测
- ✅ 支持 VaR、CVaR 等风险度量
- ✅ 轻量级模型设计，避免过拟合

---

## 安装

### 环境要求

- Python 3.7+
- PyTorch 1.9+

### 安装依赖

```bash
pip install -r requirements.txt
快速开始
1. 运行示例
python example.py


示例流程包括：

生成模拟 GARCH(1,1) 数据

训练 Neural-GARCH 模型

预测条件波动率

计算 VaR / CVaR

可视化结果

2. 模型训练

使用合成数据：

python train.py --data_type synthetic --n_samples 2000 --epochs 200


使用自定义数据文件：

python train.py --data_type file --data_path your_data.csv --epochs 100

数据格式说明

CSV 文件

包含价格列（默认列名：Close）

可选日期列（用于排序）

3. 模型预测
python predict.py \
  --model_path neural_garch_model.pth \
  --data_path your_data.csv \
  --n_steps 5

项目结构
.
├── neural_garch.py      # 模型核心实现
├── data_utils.py        # 数据生成与加载
├── train.py             # 训练脚本
├── predict.py           # 预测与风险评估
├── example.py           # 使用示例
├── requirements.txt     # 依赖列表
└── README.md

核心模块说明
NeuralGARCH 类

主要方法：

forward()
计算当前条件方差

predict_volatility_sequence()
递归预测条件波动率序列

nll_loss()
基于极大似然估计的损失函数

损失函数（极大似然）

假设条件收益率服从正态分布：

<p align="center"> <img src="https://latex.codecogs.com/png.latex?r_t|\mathcal{F}_{t-1}\sim\mathcal{N}(0,\sigma_t^2)" /> </p>

完整的高斯负对数似然为：

<p align="center"> <img src="https://latex.codecogs.com/png.latex?\mathcal{L}=\frac12\sum_t\left[\log(2\pi)+\log\sigma_t^2+\frac{r_t^2}{\sigma_t^2}\right]" /> </p>

训练中通常忽略常数项，得到优化目标：

<p align="center"> <img src="https://latex.codecogs.com/png.latex?\mathcal{L}=\sum_t\left(\log\sigma_t^2+\frac{r_t^2}{\sigma_t^2}\right)" /> </p>
风险评估
Value-at-Risk (VaR)

在置信水平 α 下的最大潜在损失：

<p align="center"> <img src="https://latex.codecogs.com/png.latex?\mathrm{VaR}_\alpha=-z_\alpha\,\sigma_t,\quad z_\alpha=\Phi^{-1}(\alpha)" /> </p>
Conditional Value-at-Risk (CVaR / ES)

超过 VaR 的期望损失：

<p align="center"> <img src="https://latex.codecogs.com/png.latex?\mathrm{CVaR}_\alpha=\sigma_t\frac{\phi(z_\alpha)}{1-\alpha}" /> </p>

其中 φ(·) 为标准正态分布密度函数。
