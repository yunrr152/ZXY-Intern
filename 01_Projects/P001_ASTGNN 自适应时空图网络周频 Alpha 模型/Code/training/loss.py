import torch
from torch.nn import functional as F


# 两项预测损失使用同一组有效标签。
def prediction_loss(scores, target, valid):
    return F.mse_loss(scores[valid], target[valid])


# 正交项在完整截面上计算，保留 Gram 矩阵对角线。
def orthogonality_loss(factors):
    num_stocks, num_factors = factors.shape
    gram = factors.T @ factors
    return torch.linalg.matrix_norm(gram, ord="fro") / (num_stocks * num_factors**2)
