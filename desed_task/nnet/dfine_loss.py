import torch
import torch.nn as nn
import torch.nn.functional as F

class DFineLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, num_classes=10):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, pred_boundaries, target_boundaries, pred_logits=None, target_labels=None):
        """计算声音事件边界的概率分布损失

        Args:
            pred_boundaries: 预测的边界概率分布 [batch_size, num_events, 2, time_steps]
            target_boundaries: 目标边界概率分布 [batch_size, num_events, 2, time_steps]
            pred_logits: 预测的事件类别概率 [batch_size, num_events, num_classes]
            target_labels: 目标事件类别 [batch_size, num_events]

        Returns:
            loss_dict: 包含边界损失和分类损失的字典
        """
        # 计算边界概率分布的KL散度损失
        kl_loss = F.kl_div(
            F.log_softmax(pred_boundaries, dim=-1),
            F.softmax(target_boundaries, dim=-1),
            reduction='none'
        ).mean()

        loss_dict = {'boundary_loss': kl_loss}

        # 如果提供了分类信息，计算分类损失
        if pred_logits is not None and target_labels is not None:
            # 计算Focal Loss
            ce_loss = F.cross_entropy(pred_logits.view(-1, self.num_classes), 
                                     target_labels.view(-1), 
                                     reduction='none')
            p = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
            loss_dict['cls_loss'] = focal_loss.mean()

        return loss_dict

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_boundaries, teacher_boundaries):
        """计算教师网络和学生网络之间的知识蒸馏损失

        Args:
            student_boundaries: 学生网络预测的边界概率分布
            teacher_boundaries: 教师网络预测的边界概率分布
        Returns:
            distillation_loss: 知识蒸馏损失
        """
        # 应用温度缩放
        student_dist = F.log_softmax(student_boundaries / self.temperature, dim=-1)
        teacher_dist = F.softmax(teacher_boundaries / self.temperature, dim=-1)

        # 计算KL散度
        distillation_loss = F.kl_div(student_dist, teacher_dist, reduction='batchmean')
        return distillation_loss * (self.temperature ** 2)