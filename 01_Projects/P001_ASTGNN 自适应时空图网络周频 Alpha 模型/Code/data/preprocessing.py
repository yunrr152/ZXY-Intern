import torch


class FeaturePreprocessor:
    """Fit per-feature statistics on historical panel rows, then reuse them."""

    def __init__(self, winsorize_mad_multiplier):
        self._statistics = None
        self.winsorize_mad_multiplier = winsorize_mad_multiplier

    @staticmethod
    def _as_float(features):
        if features.ndim < 2 or features.shape[-1] == 0:
            raise ValueError("Expected a tensor shaped [..., num_features].")
        return features if features.is_floating_point() else features.float()

    @staticmethod
    def _median_by_feature(panel):
        medians = []
        for column in panel.unbind(dim=1):
            values = column[torch.isfinite(column)]
            if values.numel() == 0:
                medians.append(column.new_full((), torch.nan))
                continue
            lower = values.kthvalue((values.numel() + 1) // 2).values
            upper = values.kthvalue(values.numel() // 2 + 1).values
            medians.append((lower + upper) / 2)
        return torch.stack(medians)

    # 用历史面板逐特征估计截断边界和标准化参数。
    @torch.no_grad()
    def fit(self, historical_panel):
        """Accept [observations, features], with one row per panel observation."""
        panel = self._as_float(historical_panel)
        if panel.ndim != 2 or panel.shape[0] == 0:
            raise ValueError("Fit requires a nonempty [observations, features] panel.")

        panel = panel.to(dtype=torch.float64)
        valid = torch.isfinite(panel) #标记有效数据
        panel = panel.masked_fill(~valid, torch.nan)
        median = self._median_by_feature(panel)
        mad = self._median_by_feature((panel - median).abs()) #MAD是Median Absolute Deviation（中位数绝对偏差），更稳健
        lower = median - self.winsorize_mad_multiplier * mad
        upper = median + self.winsorize_mad_multiplier * mad
        clipped = panel.clamp(min=lower, max=upper) #截断异常值，但没有删除样本
        count = valid.sum(dim=0).clamp_min(1)
        mean = clipped.nansum(dim=0) / count
        variance = (clipped - mean).square().nansum(dim=0) / count
        std = variance.sqrt()
        scale = torch.where(std > 0, std, 1.0)
        self._statistics = (lower, upper, mean, scale)
        return self

    # 复用已拟合参数，依次截断、标准化和填零。
    @torch.no_grad()
    def transform(self, features):
        """Transform panel rows or model windows with the feature axis last."""
        if self._statistics is None:
            raise RuntimeError("Fit the preprocessor on historical data first.")
        features = self._as_float(features)
        if features.shape[-1] != self._statistics[0].numel():
            raise ValueError("Features must match the fitted feature count and order.")

        lower, upper, mean, scale = (
            value.to(device=features.device) for value in self._statistics
        )
        valid = torch.isfinite(features)
        clipped = features.to(dtype=mean.dtype).clamp(min=lower, max=upper)
        standardized = (clipped - mean) / scale
        valid = valid & torch.isfinite(lower)
        return standardized.masked_fill(~valid, 0).to(dtype=features.dtype)
