import torch
import torch.nn as nn
import torch.nn.functional as F
from .CRNN import CRNN

class DFineSED(nn.Module):
    def __init__(self, n_in_channel=1, nclass=10, n_RNN_cell=128, n_layers_RNN=2, dropout=0.5):
        super().__init__()
        
        # 基础的CRNN编码器
        self.encoder = CRNN(
            n_in_channel=n_in_channel,
            nclass=nclass,
            n_RNN_cell=n_RNN_cell,
            n_layers_RNN=n_layers_RNN,
            dropout=dropout
        )
        
        # 边界预测头
        self.boundary_head = nn.Sequential(
            nn.Linear(n_RNN_cell * 2, n_RNN_cell),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_RNN_cell, 2)  # 2表示起始和终止时间
        )
        
        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(n_RNN_cell * 2, n_RNN_cell),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_RNN_cell, nclass)
        )
        
    def forward(self, x, return_boundaries=True):
        """前向传播
        
        Args:
            x: 输入特征 [batch_size, n_channels, n_frames, n_freq]
            return_boundaries: 是否返回边界概率分布
            
        Returns:
            output_dict: 包含预测结果的字典
        """
        # 获取CRNN特征
        features = self.encoder(x)
        batch_size, time_steps, hidden_size = features.shape
        
        # 预测边界概率分布
        boundaries = self.boundary_head(features)  # [batch_size, time_steps, 2]
        boundaries = boundaries.transpose(1, 2)  # [batch_size, 2, time_steps]
        
        # 预测类别
        logits = self.cls_head(features)  # [batch_size, time_steps, nclass]
        
        # 整合输出
        output_dict = {
            'pred_logits': logits,
            'pred_boundaries': boundaries
        }
        
        return output_dict

class DFineSEDWithDistillation(DFineSED):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 创建教师网络
        self.teacher = DFineSED(*args, **kwargs)
        
        # 冻结教师网络参数
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # 获取学生网络的预测
        student_output = super().forward(x)
        
        # 获取教师网络的预测
        with torch.no_grad():
            teacher_output = self.teacher(x)
        
        # 添加教师网络的预测到输出字典
        student_output['teacher_boundaries'] = teacher_output['pred_boundaries']
        
        return student_output