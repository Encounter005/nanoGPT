#!/usr/bin/env python3
from torch import Tensor
import torch

'''
preds = [0.1, 0.8, 0.6, 0.3] # 二元分类概率
lables = [0, 1, 1, 0]

1. preds = preds >= 0.5 = [0, 1 ,1 ,0]
2. preds = labels = [True, True, True, True]
3. .float() -> [1.0, 1.0, 1.0, 1.0] -> mean() -> 1.0 -> item() -> python.float

'''

def accuracy(preds: Tensor, labels: Tensor) -> float:
    """
    计算准确率，支持多种输入格式
    
    Args:
        preds: 模型输出
            - 多分类: (N, C) logits
            - 二分类: (N, 1) 概率 或 (N,) logits
            - 已经是预测值: (N,)
        labels: 真实标签
            - (N,) 或 (N, 1)
    
    Returns:
        float: 准确率 (0-1)
    """
    # 1. 处理 labels
    if labels.dim() > 1:
        labels = labels.squeeze()
    
    # 2. 处理 preds
    if preds.dim() > 1:
        if preds.shape[1] == 1:
            # 二分类
            preds = (preds >= 0.5).float().squeeze()
        else:
            # 多分类
            preds = preds.argmax(dim=1)
    else:
        # preds 已经是 (N,) 格式，但可能是logits（二分类）
        # 如果是二分类logits，需要sigmoid
        if preds.dtype == torch.float32:
            # 检查值是否在 [0,1] 范围内
            if preds.min() < 0 or preds.max() > 1:
                preds = torch.sigmoid(preds)
                preds = (preds >= 0.5).float()
    
    # 3. 确保形状一致
    preds = preds.reshape(labels.shape)
    
    # 4. 计算准确率
    return (preds == labels).float().mean().item()

