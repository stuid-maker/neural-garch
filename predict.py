"""
Neural-GARCH 模型预测和风险评估工具

提供概率预测、VaR、CVaR 等风险指标计算功能。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from scipy import stats
from neural_garch import NeuralGARCH, predict_next_volatility, compute_var, compute_cvar
from data_utils import load_price_data, prepare_data


def predict_conditional_distribution(
    model: NeuralGARCH,
    returns_history: torch.Tensor,
    n_steps_ahead: int = 1,
    n_simulations: int = 10000,
    device: Optional[torch.device] = None
) -> dict:
    """
    预测未来 n_steps_ahead 步的条件分布
    
    通过蒙特卡洛模拟生成未来收益率的分布。
    
    Args:
        model: 训练好的 NeuralGARCH 模型
        returns_history: 历史收益率序列 [sequence_length]
        n_steps_ahead: 预测步数
        n_simulations: 蒙特卡洛模拟次数
        device: 计算设备
        
    Returns:
        results: 包含预测结果的字典
            - mean_returns: 均值预测 [n_steps_ahead]
            - std_returns: 标准差预测 [n_steps_ahead]
            - sigma_squared_pred: 条件方差预测 [n_steps_ahead]
            - simulated_returns: 模拟的收益率 [n_simulations, n_steps_ahead]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    if returns_history.dim() == 1:
        returns_history = returns_history.unsqueeze(0)
    
    returns_history = returns_history.to(device)
    
    # 计算当前时刻的条件方差
    with torch.no_grad():
        # 使用历史数据计算当前条件方差
        current_sigma_squared_seq = model.predict_volatility_sequence(
            returns_history,
            initial_sigma_squared=torch.var(returns_history, dim=1, unbiased=False).unsqueeze(1)
        )
        current_r_squared = returns_history[:, -1] ** 2
        current_sigma_squared = current_sigma_squared_seq[:, -1]
    
    # 存储预测结果
    sigma_squared_pred = []
    simulated_returns_list = []
    
    # 当前状态
    r_current = returns_history[0, -1].item()
    sigma_squared_current = current_sigma_squared[0].item()
    
    # 蒙特卡洛模拟
    np.random.seed(42)
    
    for step in range(n_steps_ahead):
        # 预测下一时刻条件方差
        r_squared_tensor = torch.tensor([[r_current ** 2]], dtype=torch.float32, device=device)
        sigma_squared_tensor = torch.tensor([[sigma_squared_current]], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            sigma_squared_next = model.forward(r_squared_tensor, sigma_squared_tensor)
            sigma_squared_next_val = sigma_squared_next.item()
        
        sigma_squared_pred.append(sigma_squared_next_val)
        
        # 生成模拟收益率
        sigma_next = np.sqrt(sigma_squared_next_val)
        simulated_r = np.random.normal(0, sigma_next, n_simulations)
        simulated_returns_list.append(simulated_r)
        
        # 更新状态（使用模拟的均值作为下一步的输入）
        r_current = np.mean(simulated_r)
        sigma_squared_current = sigma_squared_next_val
    
    # 转换为 numpy 数组
    sigma_squared_pred = np.array(sigma_squared_pred)
    simulated_returns = np.column_stack(simulated_returns_list)
    
    # 计算统计量
    mean_returns = np.mean(simulated_returns, axis=0)
    std_returns = np.std(simulated_returns, axis=0)
    
    results = {
        'mean_returns': mean_returns,
        'std_returns': std_returns,
        'sigma_squared_pred': sigma_squared_pred,
        'simulated_returns': simulated_returns
    }
    
    return results


def calculate_risk_metrics(
    model: NeuralGARCH,
    r_t: float,
    sigma_t_squared: float,
    confidence_levels: list = [0.90, 0.95, 0.99],
    device: Optional[torch.device] = None
) -> dict:
    """
    计算风险指标（VaR 和 CVaR）
    
    Args:
        model: 训练好的 NeuralGARCH 模型
        r_t: 当前时刻收益率
        sigma_t_squared: 当前时刻条件方差
        confidence_levels: 置信水平列表
        device: 计算设备
        
    Returns:
        metrics: 风险指标字典
            - sigma_next_squared: 下一时刻条件方差预测
            - var: VaR 值字典 {confidence_level: var_value}
            - cvar: CVaR 值字典 {confidence_level: cvar_value}
    """
    # 预测下一时刻条件方差
    sigma_next_squared = predict_next_volatility(model, r_t, sigma_t_squared, device)
    
    # 计算各置信水平的 VaR 和 CVaR
    var_dict = {}
    cvar_dict = {}
    
    for cl in confidence_levels:
        var_val = compute_var(sigma_next_squared, cl)
        cvar_val = compute_cvar(sigma_next_squared, cl)
        var_dict[cl] = var_val
        cvar_dict[cl] = cvar_val
    
    metrics = {
        'sigma_next_squared': sigma_next_squared,
        'sigma_next': np.sqrt(sigma_next_squared),
        'var': var_dict,
        'cvar': cvar_dict
    }
    
    return metrics


def plot_prediction_intervals(
    returns_history: np.ndarray,
    predictions: dict,
    n_plot_steps: int = 5,
    confidence_levels: list = [0.90, 0.95, 0.99],
    save_path: Optional[str] = None
):
    """
    绘制预测区间图
    
    Args:
        returns_history: 历史收益率序列
        predictions: predict_conditional_distribution 的返回结果
        n_plot_steps: 绘制的预测步数
        confidence_levels: 置信水平列表
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 历史收益率
    n_history = len(returns_history)
    history_idx = np.arange(n_history)
    future_idx = np.arange(n_history, n_history + n_plot_steps)
    
    # 1. 历史收益率和预测均值
    axes[0].plot(history_idx, returns_history, 'b-', alpha=0.6, linewidth=1, label='history returns')
    axes[0].plot(future_idx[:n_plot_steps], 
                 predictions['mean_returns'][:n_plot_steps], 
                 'r-', linewidth=2, label='predicted mean', marker='o')
    axes[0].axvline(x=n_history-1, color='gray', linestyle='--', alpha=0.5, label='current time')
    axes[0].set_xlabel('time')
    axes[0].set_ylabel('returns')
    axes[0].set_title('returns prediction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 预测波动率（条件标准差）
    axes[1].plot(future_idx[:n_plot_steps],
                 np.sqrt(predictions['sigma_squared_pred'][:n_plot_steps]),
                 'g-', linewidth=2, marker='s', label='predicted volatility')
    
    # 绘制置信区间
    colors = ['orange', 'purple', 'brown']
    for i, cl in enumerate(confidence_levels[:3]):
        alpha_val = (1 - cl)
        z_score = stats.norm.ppf(1 - alpha_val / 2)
        
        upper_bound = predictions['mean_returns'][:n_plot_steps] + \
                     z_score * predictions['std_returns'][:n_plot_steps]
        lower_bound = predictions['mean_returns'][:n_plot_steps] - \
                     z_score * predictions['std_returns'][:n_plot_steps]
        
        axes[1].fill_between(future_idx[:n_plot_steps], 
                             lower_bound, 
                             upper_bound,
                             alpha=0.2, 
                             color=colors[i],
                             label=f'{int(cl*100)}% confidence interval')
    
    axes[1].set_xlabel('time')
    axes[1].set_ylabel('volatility / confidence interval')
    axes[1].set_title('volatility prediction and confidence interval')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"prediction plot saved to: {save_path}")
    else:
        plt.show()


def main_example():
    """示例：使用训练好的模型进行预测"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural-GARCH 预测示例')
    parser.add_argument('--model_path', type=str, default='neural_garch_model.pth',
                       help='模型文件路径')
    parser.add_argument('--data_path', type=str, default=None,
                       help='数据文件路径（用于加载历史数据）')
    parser.add_argument('--n_steps', type=int, default=5,
                       help='预测步数')
    parser.add_argument('--confidence_levels', type=float, nargs='+',
                       default=[0.90, 0.95, 0.99],
                       help='置信水平列表')
    
    args = parser.parse_args()
    
    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model_config = checkpoint['model_config']
    
    model = NeuralGARCH(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        activation=model_config.get('activation', 'relu'),
        output_activation=model_config.get('output_activation', 'softplus')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型已加载（训练损失: {checkpoint['train_loss'][-1]:.6f}）")
    
    # 加载数据（如果提供）
    if args.data_path:
        print(f"加载数据: {args.data_path}")
        returns = load_price_data(args.data_path)
        _, test_returns = prepare_data(returns, train_ratio=0.8, normalize=True)
        
        # 使用最近的 100 个数据点作为历史
        history = test_returns[-100:]
        print(f"使用历史数据长度: {len(history)}")
    else:
        # 使用模拟数据
        print("使用模拟数据进行演示...")
        from data_utils import generate_garch_data
        returns = generate_garch_data(n_samples=500, seed=42)
        _, test_returns = prepare_data(returns, train_ratio=0.8, normalize=True)
        history = test_returns[-100:]
    
    # 进行预测
    print(f"\n预测未来 {args.n_steps} 步...")
    predictions = predict_conditional_distribution(
        model, history, n_steps_ahead=args.n_steps, n_simulations=10000
    )
    
    print("\n预测结果:")
    for i in range(args.n_steps):
        print(f"  步 {i+1}: 预测波动率 = {np.sqrt(predictions['sigma_squared_pred'][i]):.6f}")
    
    # 计算风险指标
    print("\n风险指标（基于第一步预测）:")
    r_current = history[-1].item()
    sigma_current_squared = torch.var(history, unbiased=False).item()
    
    risk_metrics = calculate_risk_metrics(
        model, r_current, sigma_current_squared, args.confidence_levels
    )
    
    print(f"  预测波动率: {risk_metrics['sigma_next']:.6f}")
    for cl in args.confidence_levels:
        print(f"  VaR({int(cl*100)}%): {risk_metrics['var'][cl]:.6f}")
        print(f"  CVaR({int(cl*100)}%): {risk_metrics['cvar'][cl]:.6f}")
    
    # 绘图
    plot_prediction_intervals(
        history.numpy(),
        predictions,
        n_plot_steps=args.n_steps,
        confidence_levels=args.confidence_levels,
        save_path='prediction_results.png'
    )
    
    print("\n预测完成！")


if __name__ == '__main__':
    main_example()

