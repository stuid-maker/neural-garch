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
- **fθ(·)**：带参数 θ 的神经网络  

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

## 联系方式
