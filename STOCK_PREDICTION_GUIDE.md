# A股股票波动率预测使用指南

## 重要说明

**Neural-GARCH 模型预测的是波动率（不确定性），而不是价格涨跌方向！**

这个模型的核心价值在于：
- ✅ **风险评估**：预测股票未来的波动风险
- ✅ **风险管理**：计算 VaR/CVaR 等风险指标
- ✅ **仓位管理**：根据波动率调整仓位大小
- ❌ **不预测价格方向**：不预测明天涨还是跌

---

## 快速开始

### 步骤 1: 准备股票数据

您需要准备一个 CSV 文件，包含股票的日线数据。数据格式要求：

**必需列：**
- `Close` 或 `收盘价`：收盘价（数值）

**可选列（推荐）：**
- `Date` 或 `日期`：交易日期（用于数据排序）

**示例 CSV 格式：**

```csv
Date,Open,High,Low,Close,Volume
2024-01-01,10.50,10.80,10.30,10.60,1000000
2024-01-02,10.60,10.90,10.40,10.75,1200000
2024-01-03,10.75,10.95,10.50,10.80,1100000
...
```

**数据获取方式：**

1. **从财经网站导出**（如东方财富、同花顺等）
2. **使用 Python 库下载**（如 `akshare`, `tushare`, `yfinance` 等）
3. **自己整理的历史数据**

**示例：使用 akshare 下载A股数据**

```python
import akshare as ak
import pandas as pd

# 下载股票数据（例如：平安银行 000001）
stock_code = "000001"  # 股票代码（不带市场前缀）
stock_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")

# 保存为CSV
stock_data.to_csv("stock_data.csv", index=False)
```

---

### 步骤 2: 运行预测脚本

**基本用法：**

```bash
python predict_stock.py --data_path your_stock_data.csv
```

**完整参数：**

```bash
python predict_stock.py \
    --data_path stock_data.csv \
    --price_column Close \
    --date_column Date \
    --model_path stock_model.pth \
    --epochs 200 \
    --retrain
```

**参数说明：**
- `--data_path`: CSV数据文件路径（必需）
- `--price_column`: 价格列名（默认: Close）
- `--date_column`: 日期列名（默认: Date）
- `--model_path`: 模型保存路径（默认: stock_model.pth）
- `--epochs`: 训练轮数（默认: 200，仅在新训练时使用）
- `--retrain`: 强制重新训练模型（默认: 使用已有模型）

---

### 步骤 3: 查看结果

运行脚本后，您将得到：

1. **控制台输出**：
   - 当前波动率
   - 预测的明日波动率
   - VaR 和 CVaR 风险指标
   - 未来5个交易日的波动率预测

2. **可视化图表**（`stock_prediction_results.png`）：
   - 历史收益率与预测波动率对比
   - 未来5日波动率预测柱状图
   - 风险指标（VaR/CVaR）可视化

---

## 结果解读

### 波动率预测

**示例输出：**
```
预测明日波动率: 0.0234 (2.34%)
```

**含义：**
- 2.34% 表示预测明日收益率的标准差为 2.34%
- 如果当前价格为 100 元，预期价格在 97.66 元到 102.34 元之间波动（1个标准差）
- 波动率越大，价格波动风险越高

### VaR (Value-at-Risk)

**示例输出：**
```
90% 置信水平: -3.82% (最大可能亏损)
```

**含义：**
- 在 90% 的置信水平下，明日最大可能亏损为 3.82%
- 也就是说，有 90% 的概率亏损不会超过 3.82%
- 有 10% 的概率亏损可能超过 3.82%

### CVaR (Conditional VaR / Expected Shortfall)

**示例输出：**
```
90% 置信水平: -4.85% (超过VaR时的平均亏损)
```

**含义：**
- 当亏损超过 VaR 时（即最坏的 10% 情况），平均亏损为 4.85%
- CVaR 比 VaR 更保守，考虑了极端情况

---

## 实际应用场景

### 1. 风险评估

```python
# 如果预测波动率很高，应该：
if predicted_volatility > 0.05:  # 5%以上
    print("警告：高风险期，建议减少仓位")
```

### 2. 仓位管理

```python
# 根据波动率调整仓位
# 波动率越高，仓位越小
base_position = 0.3  # 基础仓位 30%
volatility_adjusted_position = base_position * (0.02 / predicted_volatility)
volatility_adjusted_position = min(volatility_adjusted_position, 0.5)  # 最大50%
```

### 3. 止损设置

```python
# 根据 VaR 设置止损
stop_loss = current_price * (1 + var_95)  # 使用95% VaR作为止损点
```

---

## 常见问题

### Q1: 为什么不能预测价格涨跌？

A: Neural-GARCH 模型的设计目标是预测波动率（不确定性），而不是价格方向。这是因为：
- 价格方向预测在金融学中极其困难
- 波动率预测更稳定、更可靠
- 波动率预测对风险管理更有价值

### Q2: 需要多少历史数据？

A: 建议至少 500 个交易日（约2年）的数据，推荐 1000+ 个交易日（4-5年）以获得更好的预测效果。

### Q3: 模型训练需要多长时间？

A: 在普通CPU上，1000个交易日的数据训练200个epoch大约需要几分钟。GPU会更快。

### Q4: 如何提高预测 accuracy？

A: 
- 使用更多的历史数据（1000+交易日）
- 增加训练轮数（epochs）
- 调整模型参数（hidden_dims, learning_rate等）
- 确保数据质量（去除异常值、处理停牌等）

### Q5: 支持哪些数据格式？

A: 目前支持 CSV 格式。如果您有其他格式（Excel、JSON等），可以先用 pandas 转换为 CSV。

---

## 完整示例

**下载股票数据并预测：**

```python
# 1. 下载数据（使用 akshare）
import akshare as ak
stock_data = ak.stock_zh_a_hist(symbol="000001", period="daily", adjust="qfq")
stock_data.to_csv("pingan_bank.csv", index=False)

# 2. 运行预测
# 在命令行执行：
# python predict_stock.py --data_path pingan_bank.csv --retrain
```

**使用自己的数据：**

```python
# 如果您有自己的数据文件
# 确保CSV文件包含 'Close' 列（或使用 --price_column 指定列名）

python predict_stock.py --data_path my_stock_data.csv --price_column 收盘价 --date_column 交易日期
```

---

## 注意事项

1. **数据质量**：确保数据完整，处理停牌、节假日等情况
2. **数据量**：数据量太少（<200个交易日）可能影响预测效果
3. **模型更新**：建议定期重新训练模型（如每季度），以适应市场变化
4. **风险提示**：本模型仅供学习和研究使用，投资需谨慎

---

## 技术支持

如有问题，请查看：
- `README.md`：项目总体说明
- `example.py`：基础使用示例
- 代码注释：详细的函数说明

