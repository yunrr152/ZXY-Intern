import csv
from datetime import date

import torch


# 模型使用的六个输入特征，按此顺序排列。
FEATURES = ("high", "open", "low", "close", "vwap", "volume")


class DayPanel:
    """Daily observations keyed by trading date and stock code."""

    def __init__(self, path):
        observations = {}
        with open(path, encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required = {"date", "stock_id", *FEATURES}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
            for row in reader:
                try:
                    day = date.fromisoformat(row["date"].strip())
                    stock = row["stock_id"].strip()
                    values = [
                        float(value) if (value := row[name].strip()) else float("nan")
                        for name in FEATURES
                    ]
                except (AttributeError, ValueError) as error:
                    raise ValueError(f"Invalid observation on CSV line {reader.line_num}") from error
                daily = observations.setdefault(day, {})
                if not stock or stock in daily:
                    raise ValueError(f"Empty or duplicate stock_id on CSV line {reader.line_num}")
                daily[stock] = values
        if not observations:
            raise ValueError("The day CSV contains no observations.")

        self.dates = sorted(observations)
        self.stocks = sorted({stock for daily in observations.values() for stock in daily})
        index = {stock: i for i, stock in enumerate(self.stocks)}
        self.daily = []
        self.stock_rows = []
        for day in self.dates:
            daily = observations[day]
            ids = sorted(index[stock] for stock in daily)
            self.daily.append(torch.tensor([daily[self.stocks[i]] for i in ids], dtype=torch.float32))
            self.stock_rows.append(torch.tensor(ids, dtype=torch.long))

    # 按股票对齐历史窗口，缺失观测保留为 NaN。
    def window(self, end, length):
        ids = self.stock_rows[end]
        positions = torch.full((len(self.stocks),), -1, dtype=torch.long)
        positions[ids] = torch.arange(ids.numel())
        window = torch.full((ids.numel(), length, len(FEATURES)), float("nan"))
        for step, day in enumerate(range(max(0, end - length + 1), end + 1), max(0, length - end - 1)):
            rows = positions[self.stock_rows[day]]
            present = rows >= 0
            window[rows[present], step] = self.daily[day][present]
        return ids, window

    # 按同一股票计算 T+1 至 T+11 的收盘收益率。
    def target(self, end, ids):
        closes = []
        for day in (end + 1, end + 11):
            prices = torch.full((len(self.stocks),), float("nan"))
            prices[self.stock_rows[day]] = self.daily[day][:, FEATURES.index("close")]
            closes.append(prices[ids])
        start, finish = closes
        return torch.where((start > 0) & (finish > 0), finish / start - 1, torch.nan)
