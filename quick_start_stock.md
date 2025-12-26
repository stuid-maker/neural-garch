# 快速开始：预测A股波动率

## 三步预测流程

### 步骤 1: 准备数据文件

准备一个 CSV 文件，至少包含价格列（如 `Close` 或 `收盘价`）

**示例数据格式：**
```csv
Date,Close
2023-01-01,10.50
2023-01-02,10.60
2023-01-03,10.75
...
```

### 步骤 2: 运行预测脚本

```bash
python predict_stock.py --data_path your_stock.csv --retrain
```

### 步骤 3: 查看结果

脚本会输出：
- 📊 预测的明日波动率
- 📈 风险指标（VaR, CVaR）
- 📉 未来5日波动率预测
- 📁 可视化图表（`stock_prediction_results.png`）

---

## 实际示例

假设您有一个名为 `pingan_bank.csv` 的股票数据文件：

```bash
# 第一次运行（训练新模型）
python predict_stock.py --data_path pingan_bank.csv --retrain

# 之后运行（使用已有模型，更快）
python predict_stock.py --data_path pingan_bank.csv
```

---

## 输出示例

```
============================================================
A股股票波动率预测
============================================================

1. 加载股票数据...
   ✓ 成功加载数据，共 1200 个交易日的数据
   ✓ 收益率统计: 均值=0.000123, 标准差=0.021456

2. 准备训练和测试数据...
   ✓ 训练集: 960 个交易日
   ✓ 测试集: 240 个交易日

3. 训练模型...
Epoch [10/200], Loss: -0.123456
...
   ✓ 模型已保存至: stock_model.pth

4. 预测未来波动率...
   当前波动率: 0.0234 (2.34%)
   预测明日波动率: 0.0256 (2.56%)

5. 风险指标（明日预测）:
   Value-at-Risk (VaR):
     90% 置信水平: -3.82% (最大可能亏损)
     95% 置信水平: -4.68% (最大可能亏损)
     99% 置信水平: -6.12% (最大可能亏损)

   Conditional VaR (CVaR / Expected Shortfall):
     90% 置信水平: -4.85% (超过VaR时的平均亏损)
     95% 置信水平: -5.67% (超过VaR时的平均亏损)
     99% 置信水平: -7.23% (超过VaR时的平均亏损)

6. 未来5个交易日波动率预测:
   第1个交易日: 2.56%
   第2个交易日: 2.48%
   第3个交易日: 2.41%
   第4个交易日: 2.35%
   第5个交易日: 2.30%

7. 生成可视化图表...
   ✓ 图表已保存至: stock_prediction_results.png
```

---

## 重要提示

⚠️ **本模型预测的是波动率（风险），不是价格方向！**

- ✅ 可以预测：明天价格波动会有多大（风险）
- ❌ 不能预测：明天价格是涨还是跌

**实际应用：**
- 风险管理：根据波动率调整仓位
- 止损设置：根据 VaR 设置止损点
- 风险评估：判断当前是否高风险期

---

## 数据获取建议

如果您需要A股数据，可以使用以下方式：

1. **akshare**（推荐）:
   ```python
   import akshare as ak
   data = ak.stock_zh_a_hist(symbol="000001", period="daily")
   data.to_csv("stock.csv", index=False)
   ```

2. **tushare**（需要注册）:
   ```python
   import tushare as ts
   ts.set_token('your_token')
   pro = ts.pro_api()
   data = pro.daily(ts_code='000001.SZ')
   data.to_csv("stock.csv", index=False)
   ```

3. **从财经网站导出**（如东方财富、同花顺）

---

## 更多帮助

详细文档请查看：`STOCK_PREDICTION_GUIDE.md`

