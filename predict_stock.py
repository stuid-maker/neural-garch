"""
A股股票波动率预测示例

使用 Neural-GARCH 模型预测股票的条件波动率（用于风险管理，而非价格方向预测）
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural_garch import NeuralGARCH, train_neural_garch, predict_next_volatility
from data_utils import load_price_data, prepare_data
from predict import predict_conditional_distribution, calculate_risk_metrics


def predict_stock_volatility(
    data_path: str,
    price_column: str = 'Close',
    date_column: str = 'Date',
    model_path: str = 'stock_model.pth',
    train_ratio: float = 0.8,
    epochs: int = 200,
    retrain: bool = False
):
    """
    预测股票波动率的完整流程
    
    Args:
        data_path: CSV数据文件路径
        price_column: 价格列名（默认'Close'）
        date_column: 日期列名（默认'Date'）
        model_path: 模型保存路径
        train_ratio: 训练集比例
        epochs: 训练轮数
        retrain: 是否重新训练模型（False则加载已有模型）
    """
    print("=" * 60)
    print("A股股票波动率预测")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载股票数据...")
    try:
        returns = load_price_data(data_path, price_column=price_column, date_column=date_column)
        print(f"   [OK] Successfully loaded data, {len(returns)} trading days")
        print(f"   [OK] Return statistics: mean={np.mean(returns):.6f}, std={np.std(returns):.6f}")
    except Exception as e:
        print(f"   [ERROR] Failed to load data: {e}")
        print("\nData file format requirements:")
        print("  - CSV format")
        print("  - Must contain price column (e.g., 'Close')")
        print("  - Recommended: date column (e.g., 'Date') for sorting")
        return
    
    # 2. 准备数据
    print("\n2. 准备训练和测试数据...")
    train_returns, test_returns = prepare_data(returns, train_ratio=train_ratio, normalize=True)
    print(f"   [OK] Training set: {len(train_returns)} trading days")
    print(f"   [OK] Test set: {len(test_returns)} trading days")
    
    # 3. 训练或加载模型
    if retrain or not os.path.exists(model_path):
        print("\n3. 训练模型...")
        model = NeuralGARCH(
            input_dim=2,
            hidden_dims=[32, 16],
            activation='relu',
            output_activation='softplus'
        )
        
        initial_sigma_squared = torch.var(train_returns, unbiased=False)
        
        history = train_neural_garch(
            model=model,
            returns=train_returns,
            num_epochs=epochs,
            learning_rate=0.001,
            optimizer_type='adam',
            initial_sigma_squared=initial_sigma_squared,
            clip_grad_norm=1.0,
            verbose=True
        )
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': 2,
                'hidden_dims': [32, 16],
                'activation': 'relu',
                'output_activation': 'softplus'
            },
            'train_loss': history['loss']
        }, model_path)
        print(f"   [OK] Model saved to: {model_path}")
    else:
        print("\n3. 加载已有模型...")
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint['model_config']
        model = NeuralGARCH(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   [OK] Model loaded (train loss: {checkpoint['train_loss'][-1]:.6f})")
    
    # 4. 预测未来波动率
    print("\n4. 预测未来波动率...")
    model.eval()
    
    # 使用最近的数据作为历史（建议至少100个交易日）
    history_length = min(100, len(returns))
    recent_returns = torch.tensor(returns[-history_length:], dtype=torch.float32)
    
    # 减去均值（与训练时保持一致）
    mean_return = np.mean(returns)
    recent_returns = recent_returns - mean_return
    
    with torch.no_grad():
        # 计算当前条件方差
        initial_sigma = torch.var(recent_returns, unbiased=False)
        sigma_seq = model.predict_volatility_sequence(
            recent_returns.unsqueeze(0),
            initial_sigma_squared=initial_sigma
        ).squeeze(0)
        
        current_sigma_squared = sigma_seq[-1].item()
        current_return = recent_returns[-1].item()
    
    # 预测下一交易日
    next_sigma_squared = predict_next_volatility(
        model, current_return, current_sigma_squared
    )
    next_volatility = np.sqrt(next_sigma_squared)
    
    print(f"   当前波动率: {np.sqrt(current_sigma_squared):.4f} ({np.sqrt(current_sigma_squared)*100:.2f}%)")
    print(f"   预测明日波动率: {next_volatility:.4f} ({next_volatility*100:.2f}%)")
    
    # 5. 计算风险指标
    print("\n5. 风险指标（明日预测）:")
    risk_metrics = calculate_risk_metrics(
        model, current_return, current_sigma_squared,
        confidence_levels=[0.90, 0.95, 0.99]
    )
    
    print(f"   Value-at-Risk (VaR):")
    for cl in [0.90, 0.95, 0.99]:
        var_pct = risk_metrics['var'][cl] * 100
        print(f"     {int(cl*100)}% 置信水平: {var_pct:.2f}% (最大可能亏损)")
    
    print(f"\n   Conditional VaR (CVaR / Expected Shortfall):")
    for cl in [0.90, 0.95, 0.99]:
        cvar_pct = risk_metrics['cvar'][cl] * 100
        print(f"     {int(cl*100)}% 置信水平: {cvar_pct:.2f}% (超过VaR时的平均亏损)")
    
    # 6. 多步预测
    print("\n6. 未来5个交易日波动率预测:")
    predictions = predict_conditional_distribution(
        model, recent_returns, n_steps_ahead=5, n_simulations=10000
    )
    
    for i in range(5):
        vol_pct = np.sqrt(predictions['sigma_squared_pred'][i]) * 100
        print(f"   第{i+1}个交易日: {vol_pct:.2f}%")
    
    # 7. 可视化
    print("\n7. 生成可视化图表...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 历史收益率和波动率
    returns_array = returns[-200:]  # 最近200个交易日
    with torch.no_grad():
        recent_history = torch.tensor(returns_array, dtype=torch.float32) - mean_return
        initial_sigma_viz = torch.var(recent_history, unbiased=False)
        sigma_seq_viz = model.predict_volatility_sequence(
            recent_history.unsqueeze(0),
            initial_sigma_squared=initial_sigma_viz
        ).squeeze(0).detach().numpy()
    volatility_array = np.sqrt(sigma_seq_viz) * 100  # 转换为百分比
    
    axes[0].plot(returns_array * 100, alpha=0.6, label='Daily Returns (%)', linewidth=0.8)
    axes[0].plot(volatility_array, 'r-', label='Predicted Volatility (%)', linewidth=1.5)
    axes[0].plot(-volatility_array, 'r--', linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel('Trading Days')
    axes[0].set_ylabel('Returns / Volatility (%)')
    axes[0].set_title('Historical Returns and Predicted Volatility')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 预测未来波动率
    future_vol = np.sqrt(predictions['sigma_squared_pred']) * 100
    future_days = np.arange(1, 6)
    axes[1].bar(future_days, future_vol, alpha=0.7, color='green', label='Predicted Volatility')
    axes[1].axhline(y=volatility_array[-1], color='r', linestyle='--', label='Current Volatility')
    axes[1].set_xlabel('Days Ahead')
    axes[1].set_ylabel('Volatility (%)')
    axes[1].set_title('Future Volatility Prediction (Next 5 Trading Days)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticks(future_days)
    
    # VaR 可视化
    var_values = [risk_metrics['var'][cl] * 100 for cl in [0.90, 0.95, 0.99]]
    cvar_values = [risk_metrics['cvar'][cl] * 100 for cl in [0.90, 0.95, 0.99]]
    confidence_levels = ['90%', '95%', '99%']
    
    x_pos = np.arange(len(confidence_levels))
    width = 0.35
    
    axes[2].bar(x_pos - width/2, var_values, width, label='VaR', alpha=0.8, color='orange')
    axes[2].bar(x_pos + width/2, cvar_values, width, label='CVaR', alpha=0.8, color='red')
    axes[2].set_xlabel('Confidence Level')
    axes[2].set_ylabel('Loss (%)')
    axes[2].set_title('Risk Metrics (Next Trading Day)')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(confidence_levels)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = 'stock_prediction_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   [OK] Chart saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("预测完成！")
    print("=" * 60)
    print("\n重要说明:")
    print("  • 本模型预测的是波动率（不确定性），而非价格涨跌方向")
    print("  • 波动率预测可用于风险评估、仓位管理和止损设置")
    print("  • 高波动率意味着更大的价格波动风险")
    print("  • VaR/CVaR 表示在给定置信水平下的最大可能亏损")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='A股股票波动率预测')
    parser.add_argument('--data_path', type=str, required=True,
                       help='股票数据CSV文件路径')
    parser.add_argument('--price_column', type=str, default='Close',
                       help='价格列名（默认: Close）')
    parser.add_argument('--date_column', type=str, default='Date',
                       help='日期列名（默认: Date）')
    parser.add_argument('--model_path', type=str, default='stock_model.pth',
                       help='模型保存/加载路径')
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数（仅在新训练时使用）')
    parser.add_argument('--retrain', action='store_true',
                       help='强制重新训练模型')
    
    args = parser.parse_args()
    
    predict_stock_volatility(
        data_path=args.data_path,
        price_column=args.price_column,
        date_column=args.date_column,
        model_path=args.model_path,
        epochs=args.epochs,
        retrain=args.retrain
    )

