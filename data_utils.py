"""
数据生成和加载工具

用于生成模拟金融时间序列数据，或加载真实金融数据。
"""

import numpy as np
import torch
from typing import Optional, Tuple
import pandas as pd


def generate_garch_data(
    n_samples: int = 1000,
    omega: float = 0.01,
    alpha: float = 0.1,
    beta: float = 0.85,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    生成 GARCH(1,1) 过程的模拟数据
    
    Args:
        n_samples: 样本数量
        omega: GARCH 参数 ω
        alpha: GARCH 参数 α
        beta: GARCH 参数 β
        seed: 随机种子
        
    Returns:
        returns: 收益率序列
    """
    if seed is not None:
        np.random.seed(seed)
    
    returns = np.zeros(n_samples)
    sigma_squared = np.zeros(n_samples)
    
    # 初始化
    sigma_squared[0] = omega / (1 - alpha - beta)  # 无条件方差
    
    # 生成序列
    for t in range(1, n_samples):
        # GARCH(1,1) 更新
        sigma_squared[t] = omega + alpha * returns[t-1]**2 + beta * sigma_squared[t-1]
        
        # 生成收益率
        z_t = np.random.randn()
        returns[t] = np.sqrt(sigma_squared[t]) * z_t
    
    return returns


def generate_random_walk_returns(
    n_samples: int = 1000,
    volatility: float = 0.02,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    生成随机游走过程的收益率（简单基准）
    
    Args:
        n_samples: 样本数量
        volatility: 波动率水平
        seed: 随机种子
        
    Returns:
        returns: 收益率序列
    """
    if seed is not None:
        np.random.seed(seed)
    
    returns = np.random.normal(0, volatility, n_samples)
    return returns


def load_price_data(
    file_path: str,
    price_column: str = 'Close',
    date_column: Optional[str] = None
) -> np.ndarray:
    """
    从CSV文件加载价格数据并计算对数收益率
    
    Args:
        file_path: CSV文件路径
        price_column: 价格列名（默认'Close'）
        date_column: 日期列名（可选，用于排序）
        
    Returns:
        returns: 对数收益率序列
    """
    df = pd.read_csv(file_path)
    
    # 如果有日期列，按日期排序
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
    
    # 提取价格序列
    if price_column not in df.columns:
        raise ValueError(f"列 '{price_column}' 不存在于数据文件中")
    
    prices = df[price_column].values
    
    # 计算对数收益率
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    
    return returns


def prepare_data(
    returns: np.ndarray,
    train_ratio: float = 0.8,
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    准备训练和测试数据
    
    Args:
        returns: 收益率序列
        train_ratio: 训练集比例
        normalize: 是否标准化（减去均值）
        
    Returns:
        train_returns: 训练集 [tensor]
        test_returns: 测试集 [tensor]
    """
    returns = returns.copy()
    
    # 标准化（仅减去均值，保留波动率信息）
    if normalize:
        mean_return = np.mean(returns)
        returns = returns - mean_return
    
    # 划分训练集和测试集
    n_train = int(len(returns) * train_ratio)
    train_returns = returns[:n_train]
    test_returns = returns[n_train:]
    
    # 转换为 torch.Tensor
    train_tensor = torch.tensor(train_returns, dtype=torch.float32)
    test_tensor = torch.tensor(test_returns, dtype=torch.float32)
    
    return train_tensor, test_tensor


def compute_realized_volatility(
    returns: np.ndarray,
    window: int = 20
) -> np.ndarray:
    """
    计算已实现波动率（滑动窗口标准差）
    
    用于与模型预测的波动率进行对比评估。
    
    Args:
        returns: 收益率序列
        window: 滑动窗口大小
        
    Returns:
        realized_vol: 已实现波动率序列
    """
    realized_vol = np.zeros_like(returns)
    
    for i in range(len(returns)):
        start_idx = max(0, i - window + 1)
        realized_vol[i] = np.std(returns[start_idx:i+1]) * np.sqrt(window)
    
    return realized_vol

