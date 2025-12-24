"""
Neural-GARCH 模型实现

使用神经网络替代传统 GARCH 模型的线性参数形式，
用于金融时间序列的条件波动率预测。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, Union


class NeuralGARCH(nn.Module):
    """
    Neural-GARCH 模型
    
    使用神经网络建模条件方差的递归演化：
    σ_t^2 = f_θ(r_{t-1}^2, σ_{t-1}^2)
    
    Args:
        input_dim: 输入维度（默认2：r_{t-1}^2 和 σ_{t-1}^2）
        hidden_dims: 隐藏层维度列表，例如 [32, 16]
        activation: 激活函数类型，'relu' 或 'tanh'
        output_activation: 输出层激活函数，'softplus' 或 'exp'（确保非负）
        dropout: Dropout 概率（可选）
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list = [32, 16],
        activation: str = 'relu',
        output_activation: str = 'softplus',
        dropout: float = 0.0
    ):
        super(NeuralGARCH, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_activation = output_activation
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"不支持的激活函数: {activation}")
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # 输出层（条件方差，必须为正）
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化参数
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(
        self,
        r_squared: torch.Tensor,
        sigma_squared: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            r_squared: 上一时刻平方收益率 [batch_size, 1] 或 [batch_size]
            sigma_squared: 上一时刻条件方差 [batch_size, 1] 或 [batch_size]
            
        Returns:
            sigma_t_squared: 当前时刻条件方差预测 [batch_size, 1]
        """
        # 确保输入维度一致
        if r_squared.dim() == 1:
            r_squared = r_squared.unsqueeze(1)
        if sigma_squared.dim() == 1:
            sigma_squared = sigma_squared.unsqueeze(1)
        
        # 拼接输入
        x = torch.cat([r_squared, sigma_squared], dim=1)
        
        # 通过网络
        output = self.network(x)
        
        # 应用输出激活函数确保非负
        if self.output_activation == 'softplus':
            # Softplus: log(1 + exp(x))，数值稳定版本
            sigma_t_squared = nn.functional.softplus(output) + 1e-8
        elif self.output_activation == 'exp':
            # 指数函数，需要裁剪输入防止溢出
            output_clipped = torch.clamp(output, max=10.0)
            sigma_t_squared = torch.exp(output_clipped) + 1e-8
        else:
            raise ValueError(f"不支持的输出激活函数: {self.output_activation}")
        
        return sigma_t_squared
    
    def predict_volatility_sequence(
        self,
        returns: torch.Tensor,
        initial_sigma_squared: Optional[torch.Tensor] = None,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        递归计算整个序列的条件方差
        
        这是 Neural-GARCH 的核心递归计算机制。
        
        Args:
            returns: 收益率序列 [sequence_length] 或 [batch_size, sequence_length]
            initial_sigma_squared: 初始条件方差（默认使用样本方差）
            eps: 数值稳定性参数
            
        Returns:
            sigma_squared_seq: 条件方差序列 [sequence_length] 或 [batch_size, sequence_length]
        """
        if returns.dim() == 1:
            returns = returns.unsqueeze(0)  # [1, seq_len]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len = returns.shape
        device = returns.device
        
        # 初始化条件方差
        if initial_sigma_squared is None:
            # 使用样本方差作为初始值
            initial_sigma_squared = torch.var(returns, dim=1, unbiased=False).unsqueeze(1)
        else:
            # 确保 initial_sigma_squared 在正确的设备上
            initial_sigma_squared = initial_sigma_squared.to(device)
            
            # 处理不同维度的输入，统一转换为 [batch_size, 1]
            if initial_sigma_squared.dim() == 0:
                # 标量，扩展为 [batch_size, 1]
                initial_sigma_squared = initial_sigma_squared.unsqueeze(0).unsqueeze(0)
                initial_sigma_squared = initial_sigma_squared.expand(batch_size, 1)
            elif initial_sigma_squared.dim() == 1:
                # 一维张量
                if initial_sigma_squared.shape[0] == 1:
                    # 单个值，扩展到所有batch
                    initial_sigma_squared = initial_sigma_squared.unsqueeze(1).expand(batch_size, 1)
                elif initial_sigma_squared.shape[0] == batch_size:
                    # 每个batch一个值
                    initial_sigma_squared = initial_sigma_squared.unsqueeze(1)
                else:
                    raise ValueError(f"initial_sigma_squared 第一维大小 ({initial_sigma_squared.shape[0]}) 与 batch_size ({batch_size}) 不匹配")
            elif initial_sigma_squared.dim() == 2:
                # 二维张量，检查形状
                if initial_sigma_squared.shape[0] != batch_size:
                    raise ValueError(f"initial_sigma_squared 第一维大小 ({initial_sigma_squared.shape[0]}) 与 batch_size ({batch_size}) 不匹配")
                if initial_sigma_squared.shape[1] != 1:
                    initial_sigma_squared = initial_sigma_squared[:, 0:1]
        
        # 存储条件方差序列
        sigma_squared_seq = torch.zeros(batch_size, seq_len, device=device)
        sigma_squared_seq[:, 0] = initial_sigma_squared.squeeze(1)
        
        # 递归计算
        for t in range(1, seq_len):
            r_prev_squared = (returns[:, t-1] ** 2).unsqueeze(1)
            sigma_prev_squared = sigma_squared_seq[:, t-1].unsqueeze(1)
            
            # 预测当前时刻条件方差
            sigma_t_squared = self.forward(r_prev_squared, sigma_prev_squared)
            sigma_squared_seq[:, t] = sigma_t_squared.squeeze(1)
        
        if squeeze_output:
            sigma_squared_seq = sigma_squared_seq.squeeze(0)
        
        return sigma_squared_seq
    
    def nll_loss(
        self,
        returns: torch.Tensor,
        initial_sigma_squared: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算负对数似然损失（MLE目标）
        
        假设 r_t | F_{t-1} ~ N(0, σ_t^2)
        
        负对数似然 = Σ [log(σ_t^2) + r_t^2 / σ_t^2]
        
        Args:
            returns: 收益率序列 [sequence_length] 或 [batch_size, sequence_length]
            initial_sigma_squared: 初始条件方差
            
        Returns:
            loss: 标量损失值
        """
        # 递归计算条件方差序列
        sigma_squared_seq = self.predict_volatility_sequence(
            returns, initial_sigma_squared
        )
        
        if returns.dim() == 1:
            returns = returns.unsqueeze(0)
        
        # 计算负对数似然
        # loss = log(σ_t^2) + r_t^2 / σ_t^2
        log_sigma_squared = torch.log(sigma_squared_seq)
        r_squared = returns ** 2
        normalized_r_squared = r_squared / (sigma_squared_seq + 1e-8)
        
        nll = log_sigma_squared + normalized_r_squared
        
        # 对所有时间步和批次求平均
        loss = torch.mean(nll)
        
        return loss


def train_neural_garch(
    model: NeuralGARCH,
    returns: torch.Tensor,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    optimizer_type: str = 'adam',
    initial_sigma_squared: Optional[torch.Tensor] = None,
    clip_grad_norm: Optional[float] = 1.0,
    verbose: bool = True,
    device: Optional[torch.device] = None
) -> dict:
    """
    训练 Neural-GARCH 模型
    
    Args:
        model: NeuralGARCH 模型实例
        returns: 收益率序列 [sequence_length] 或 [batch_size, sequence_length]
        num_epochs: 训练轮数
        learning_rate: 学习率
        optimizer_type: 优化器类型，'adam' 或 'rmsprop'
        initial_sigma_squared: 初始条件方差
        clip_grad_norm: 梯度裁剪阈值（None表示不裁剪）
        verbose: 是否打印训练信息
        device: 计算设备
        
    Returns:
        history: 训练历史字典，包含损失列表
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.train()
    
    # 确保 returns 在正确的设备上
    if returns.dim() == 1:
        returns = returns.unsqueeze(0)
    returns = returns.to(device)
    
    # 初始化优化器
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    history = {'loss': []}
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 计算损失
        loss = model.nll_loss(returns, initial_sigma_squared)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止数值不稳定）
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        loss_value = loss.item()
        history['loss'].append(loss_value)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_value:.6f}")
    
    return history


def predict_next_volatility(
    model: NeuralGARCH,
    r_t: float,
    sigma_t_squared: float,
    device: Optional[torch.device] = None
) -> float:
    """
    预测下一时刻的条件方差
    
    Args:
        model: 训练好的 NeuralGARCH 模型
        r_t: 当前时刻收益率
        sigma_t_squared: 当前时刻条件方差
        device: 计算设备
        
    Returns:
        sigma_next_squared: 下一时刻条件方差预测值
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    with torch.no_grad():
        r_squared = torch.tensor([r_t ** 2], dtype=torch.float32, device=device)
        sigma_squared = torch.tensor([sigma_t_squared], dtype=torch.float32, device=device)
        
        sigma_next_squared = model.forward(r_squared, sigma_squared)
        
        return sigma_next_squared.item()


def compute_var(
    sigma_squared: float,
    confidence_level: float = 0.95
) -> float:
    """
    计算 Value-at-Risk (VaR)
    
    假设 r_t ~ N(0, σ_t^2)
    
    Args:
        sigma_squared: 条件方差
        confidence_level: 置信水平（默认0.95）
        
    Returns:
        var: VaR 值（负值表示损失）
    """
    from scipy import stats
    sigma = np.sqrt(sigma_squared)
    z_score = stats.norm.ppf(1 - confidence_level)
    var = z_score * sigma
    return var


def compute_cvar(
    sigma_squared: float,
    confidence_level: float = 0.95
) -> float:
    """
    计算 Conditional Value-at-Risk (CVaR / Expected Shortfall)
    
    Args:
        sigma_squared: 条件方差
        confidence_level: 置信水平（默认0.95）
        
    Returns:
        cvar: CVaR 值（负值表示损失）
    """
    from scipy import stats
    sigma = np.sqrt(sigma_squared)
    z_score = stats.norm.ppf(1 - confidence_level)
    
    # CVaR = -σ * φ(z_α) / (1 - α)
    # 其中 φ 是标准正态分布的密度函数
    phi_z = stats.norm.pdf(z_score)
    cvar = -sigma * phi_z / (1 - confidence_level)
    
    return cvar

