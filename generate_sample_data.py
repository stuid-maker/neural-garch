"""
生成示例股票数据文件（用于测试）

生成模拟的股票价格数据，保存为CSV格式
"""

import numpy as np
import pandas as pd
from data_utils import generate_garch_data


def generate_sample_stock_data(
    output_path: str = "sample_stock.csv",
    n_days: int = 500,
    initial_price: float = 10.0,
    seed: int = 42
):
    """
    生成示例股票数据
    
    Args:
        output_path: 输出文件路径
        n_days: 生成的天数
        initial_price: 初始价格
        seed: 随机种子
    """
    print(f"生成 {n_days} 天的示例股票数据...")
    
    # 生成收益率序列（使用GARCH过程）
    returns = generate_garch_data(n_samples=n_days, seed=seed)
    
    # 生成价格序列
    prices = [initial_price]
    for ret in returns:
        prices.append(prices[-1] * np.exp(ret))
    
    prices = prices[1:]  # 移除初始价格
    
    # 创建DataFrame
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='D')
    
    # 过滤掉周末（仅保留工作日）
    dates = dates[dates.weekday < 5]
    prices = prices[:len(dates)]
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.randn(len(prices)) * 0.001),  # 开盘价略波动
        'High': prices * (1 + np.abs(np.random.randn(len(prices)) * 0.005)),  # 最高价
        'Low': prices * (1 - np.abs(np.random.randn(len(prices)) * 0.005)),  # 最低价
        'Close': prices,  # 收盘价
        'Volume': np.random.randint(1000000, 10000000, len(prices))  # 成交量
    })
    
    # 确保 High >= Close >= Low
    df['High'] = np.maximum(df['High'], df['Close'])
    df['Low'] = np.minimum(df['Low'], df['Close'])
    df['Open'] = np.clip(df['Open'], df['Low'], df['High'])
    
    # 保存为CSV
    df.to_csv(output_path, index=False)
    print(f"[OK] Data saved to: {output_path}")
    print(f"  - Rows: {len(df)}")
    print(f"  - Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  - Price range: {df['Close'].min():.2f} to {df['Close'].max():.2f}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成示例股票数据')
    parser.add_argument('--output', type=str, default='sample_stock.csv',
                       help='输出文件路径（默认: sample_stock.csv）')
    parser.add_argument('--days', type=int, default=500,
                       help='生成的天数（默认: 500）')
    parser.add_argument('--initial_price', type=float, default=10.0,
                       help='初始价格（默认: 10.0）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认: 42）')
    
    args = parser.parse_args()
    
    generate_sample_stock_data(
        output_path=args.output,
        n_days=args.days,
        initial_price=args.initial_price,
        seed=args.seed
    )

