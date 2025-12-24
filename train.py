"""
Neural-GARCH 模型训练主脚本

用法示例：
    python train.py --data_type synthetic --n_samples 2000 --epochs 200
    python train.py --data_type file --data_path stock_data.csv --epochs 100
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_garch import NeuralGARCH, train_neural_garch, predict_next_volatility
from data_utils import (
    generate_garch_data,
    generate_random_walk_returns,
    load_price_data,
    prepare_data,
    compute_realized_volatility
)


def main():
    parser = argparse.ArgumentParser(description='训练 Neural-GARCH 模型')
    
    # 数据参数
    parser.add_argument('--data_type', type=str, default='synthetic',
                       choices=['synthetic', 'random_walk', 'file'],
                       help='数据类型：synthetic (GARCH), random_walk, 或 file')
    parser.add_argument('--data_path', type=str, default=None,
                       help='数据文件路径（当 data_type=file 时使用）')
    parser.add_argument('--n_samples', type=int, default=2000,
                       help='生成数据的样本数量（当使用 synthetic 时）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    
    # 模型参数
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 16],
                       help='隐藏层维度列表，例如：--hidden_dims 32 16')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'tanh'],
                       help='激活函数类型')
    parser.add_argument('--output_activation', type=str, default='softplus',
                       choices=['softplus', 'exp'],
                       help='输出层激活函数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'rmsprop'],
                       help='优化器类型')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                       help='梯度裁剪阈值（0表示不裁剪）')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save_model', type=str, default='neural_garch_model.pth',
                       help='模型保存路径')
    parser.add_argument('--plot', action='store_true',
                       help='是否绘制训练曲线和预测结果')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载或生成数据
    print("=" * 50)
    print("加载数据...")
    
    if args.data_type == 'synthetic':
        print(f"生成 GARCH(1,1) 模拟数据（样本数: {args.n_samples}）")
        returns = generate_garch_data(n_samples=args.n_samples, seed=args.seed)
    elif args.data_type == 'random_walk':
        print(f"生成随机游走数据（样本数: {args.n_samples}）")
        returns = generate_random_walk_returns(n_samples=args.n_samples, seed=args.seed)
    elif args.data_type == 'file':
        if args.data_path is None:
            raise ValueError("使用 file 数据类型时必须提供 --data_path")
        print(f"从文件加载数据: {args.data_path}")
        returns = load_price_data(args.data_path)
    
    print(f"数据形状: {returns.shape}")
    print(f"收益率统计: 均值={np.mean(returns):.6f}, 标准差={np.std(returns):.6f}")
    
    # 准备训练和测试数据
    train_returns, test_returns = prepare_data(
        returns, 
        train_ratio=args.train_ratio,
        normalize=True
    )
    
    print(f"训练集大小: {len(train_returns)}")
    print(f"测试集大小: {len(test_returns)}")
    
    # 创建模型
    print("\n" + "=" * 50)
    print("创建模型...")
    model = NeuralGARCH(
        input_dim=2,
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        output_activation=args.output_activation
    )
    
    print(f"模型结构:")
    print(f"  输入维度: {model.input_dim}")
    print(f"  隐藏层: {args.hidden_dims}")
    print(f"  激活函数: {args.activation}")
    print(f"  输出激活: {args.output_activation}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    
    # 训练模型
    print("\n" + "=" * 50)
    print("开始训练...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化条件方差（使用训练集方差）
    initial_sigma_squared = torch.var(train_returns, unbiased=False)
    
    history = train_neural_garch(
        model=model,
        returns=train_returns,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        initial_sigma_squared=initial_sigma_squared,
        clip_grad_norm=args.clip_grad_norm if args.clip_grad_norm > 0 else None,
        verbose=True,
        device=device
    )
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.input_dim,
            'hidden_dims': model.hidden_dims,
            'activation': args.activation,
            'output_activation': model.output_activation
        },
        'train_loss': history['loss']
    }, args.save_model)
    print(f"\n模型已保存至: {args.save_model}")
    
    # 评估和可视化
    if args.plot:
        print("\n" + "=" * 50)
        print("生成可视化结果...")
        
        model.eval()
        with torch.no_grad():
            # 计算训练集的条件方差序列
            train_sigma_squared = model.predict_volatility_sequence(
                train_returns.unsqueeze(0),
                initial_sigma_squared=initial_sigma_squared
            ).squeeze(0).numpy()
            
            # 计算测试集的条件方差序列
            test_initial_sigma = train_sigma_squared[-1]
            test_sigma_squared = model.predict_volatility_sequence(
                test_returns.unsqueeze(0),
                initial_sigma_squared=torch.tensor([test_initial_sigma])
            ).squeeze(0).numpy()
        
        # 绘图
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. 训练损失曲线
        axes[0].plot(history['loss'])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Negative Log Likelihood Loss')
        axes[0].set_title('Training Loss Curve')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 收益率和预测波动率（训练集）
        train_returns_np = train_returns.numpy()
        train_volatility = np.sqrt(train_sigma_squared)
        
        axes[1].plot(train_returns_np, alpha=0.6, label='Returns', linewidth=0.5)
        axes[1].plot(train_volatility, label='Predicted Volatility', linewidth=1.5)
        axes[1].plot(-train_volatility, '--', linewidth=1.5, alpha=0.7)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Returns / Volatility')
        axes[1].set_title('Training Set: Returns and Predicted Volatility')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 收益率和预测波动率（测试集）
        test_returns_np = test_returns.numpy()
        test_volatility = np.sqrt(test_sigma_squared)
        
        axes[2].plot(test_returns_np, alpha=0.6, label='Returns', linewidth=0.5)
        axes[2].plot(test_volatility, label='Predicted Volatility', linewidth=1.5)
        axes[2].plot(-test_volatility, '--', linewidth=1.5, alpha=0.7)
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Returns / Volatility')
        axes[2].set_title('Test Set: Returns and Predicted Volatility')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
        print("可视化结果已保存至: training_results.png")
        
        # 计算评估指标
        print("\n" + "=" * 50)
        print("模型评估指标:")
        
        # 已实现波动率（作为参考）
        train_realized_vol = compute_realized_volatility(train_returns_np, window=20)
        test_realized_vol = compute_realized_volatility(test_returns_np, window=20)
        
        # 计算相关性
        train_corr = np.corrcoef(train_volatility, train_realized_vol)[0, 1]
        test_corr = np.corrcoef(test_volatility, test_realized_vol)[0, 1]
        
        print(f"训练集：预测波动率 vs 已实现波动率相关性 = {train_corr:.4f}")
        print(f"测试集：预测波动率 vs 已实现波动率相关性 = {test_corr:.4f}")
        
        print("\n训练完成！")


if __name__ == '__main__':
    main()

