from contextlib import contextmanager

import torch
from torch.nn import BatchNorm1d


@contextmanager
def suspend_running_statistics(model):
    # 对抗分支使用当前截面统计量，并且暂停统计量更新。
    layers = [
        layer for layer in model.modules()
        if isinstance(layer, BatchNorm1d) and layer.training and layer.track_running_stats
    ]
    try:
        for layer in layers:
            layer.track_running_stats = False
        yield
    finally:
        for layer in layers:
            layer.track_running_stats = True


# 沿输入梯度符号迭代上升，每步投影回原始输入的扰动范围。
def ifgsm(x, objective, epsilon, step_size, steps):
    original = x.detach()
    lower, upper = original - epsilon, original + epsilon
    adversarial = original
    for _ in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        gradient, = torch.autograd.grad(objective(adversarial), adversarial)
        with torch.no_grad():
            adversarial = (adversarial + step_size * gradient.sign()).clamp(lower, upper)
    return adversarial
