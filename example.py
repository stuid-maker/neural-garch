"""
Neural-GARCH 模型使用示例

展示如何使用模型进行训练和预测的基本流程。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_garch import NeuralGARCH, train_neural_garch, predict_next_volatility
from predict import calculate_risk_metrics
from data_utils import generate_garch_data, prepare_data


def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("Neural-GARCH 基本使用示例")
    print("=" * 60)
    
    # 1. 生成模拟数据
    print("\n1. 生成 GARCH(1,1) 模拟数据...")
    returns = generate_garch_data(n_samples=2000, seed=42)
    print(f"   数据形状: {returns.shape}")
    print(f"   收益率统计: 均值={np.mean(returns):.6f}, 标准差={np.std(returns):.6f}")
    
    # 2. 准备数据
    print("\n2. 准备训练和测试数据...")
    train_returns, test_returns = prepare_data(returns, train_ratio=0.8, normalize=True)
    print(f"   训练集大小: {len(train_returns)}")
    print(f"   测试集大小: {len(test_returns)}")
    
    # 3. 创建模型
    print("\n3. 创建 Neural-GARCH 模型...")
    model = NeuralGARCH(
        input_dim=2,              # 输入：r_{t-1}^2 和 σ_{t-1}^2
        hidden_dims=[32, 16],     # 两层隐藏层
        activation='relu',        # ReLU 激活函数
        output_activation='softplus'  # Softplus 确保输出为正
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   总参数量: {total_params:,}")
    
    # 4. 训练模型
    print("\n4. 训练模型...")
    initial_sigma_squared = torch.var(train_returns, unbiased=False)
    
    history = train_neural_garch(
        model=model,
        returns=train_returns,
        num_epochs=100,
        learning_rate=0.001,
        optimizer_type='adam',
        initial_sigma_squared=initial_sigma_squared,
        clip_grad_norm=1.0,
        verbose=True
    )
    
    print(f"   最终训练损失: {history['loss'][-1]:.6f}")
    
    # 5. 预测下一时刻波动率
    print("\n5. 预测下一时刻条件方差...")
    model.eval()
    
    # 使用训练集最后一个数据点
    r_last = train_returns[-1].item()
    sigma_last_squared = initial_sigma_squared.item()
    
    sigma_next_squared = predict_next_volatility(model, r_last, sigma_last_squared)
    sigma_next = np.sqrt(sigma_next_squared)
    
    print(f"   当前收益率: {r_last:.6f}")
    print(f"   当前条件方差: {sigma_last_squared:.6f}")
    print(f"   预测下一时刻条件方差: {sigma_next_squared:.6f}")
    print(f"   预测下一时刻波动率: {sigma_next:.6f}")
    
    # 6. 计算风险指标
    print("\n6. 计算风险指标（VaR 和 CVaR）...")
    risk_metrics = calculate_risk_metrics(
        model, r_last, sigma_last_squared, confidence_levels=[0.90, 0.95, 0.99]
    )
    
    print(f"   预测波动率: {risk_metrics['sigma_next']:.6f}")
    for cl in [0.90, 0.95, 0.99]:
        print(f"   VaR({int(cl*100)}%): {risk_metrics['var'][cl]:.6f}")
        print(f"   CVaR({int(cl*100)}%): {risk_metrics['cvar'][cl]:.6f}")
    
    # 7. 可视化结果
    print("\n7. 生成可视化结果...")
    
    model.eval()
    with torch.no_grad():
        # 计算测试集的条件方差序列
        test_initial_sigma = torch.var(train_returns, unbiased=False)
        test_sigma_squared = model.predict_volatility_sequence(
            test_returns.unsqueeze(0),
            initial_sigma_squared=test_initial_sigma
        ).squeeze(0).detach().numpy()
    
    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 训练损失
    axes[0].plot(history['loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Negative Log Likelihood Loss')
    axes[0].set_title('Training Loss Curve')
    axes[0].grid(True, alpha=0.3)
    
    # 训练集收益率和波动率
    with torch.no_grad():
        train_returns_np = train_returns.detach().numpy()
        train_sigma_squared = model.predict_volatility_sequence(
            train_returns.unsqueeze(0),
            initial_sigma_squared=initial_sigma_squared
        ).squeeze(0).detach().numpy()
    train_volatility = np.sqrt(train_sigma_squared)
    
    axes[1].plot(train_returns_np, alpha=0.5, label='Returns', linewidth=0.5)
    axes[1].plot(train_volatility, 'r-', label='Predicted Volatility', linewidth=1.5)
    axes[1].plot(-train_volatility, 'r--', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Returns / Volatility')
    axes[1].set_title('Training Set: Returns and Predicted Volatility')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 测试集收益率和波动率
    test_returns_np = test_returns.detach().numpy()
    test_volatility = np.sqrt(test_sigma_squared)
    
    axes[2].plot(test_returns_np, alpha=0.5, label='Returns', linewidth=0.5)
    axes[2].plot(test_volatility, 'g-', label='Predicted Volatility', linewidth=1.5)
    axes[2].plot(-test_volatility, 'g--', linewidth=1.5, alpha=0.7)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Returns / Volatility')
    axes[2].set_title('Test Set: Returns and Predicted Volatility')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_results.png', dpi=150, bbox_inches='tight')
    print("   可视化结果已保存至: example_results.png")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == '__main__':
    example_basic_usage()

