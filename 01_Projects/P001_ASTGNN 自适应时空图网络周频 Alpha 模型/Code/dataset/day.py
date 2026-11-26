import csv
from datetime import date
from time import perf_counter

import torch
from tqdm import tqdm


# 模型使用的六个输入特征（高开低收，VWAP，成交量），按此顺序排列。
FEATURES = ("high", "open", "low", "close", "vwap", "volume")


class DayPanel:
    """Full daily observations plus an aligned, time-varying stock-pool mask."""

    def __init__(self, path, pool_mask_path, logger=None):
        panel_started = perf_counter()
        observations = {}
        feature_rows = 0
        csv_started = perf_counter()
        with open(path, encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required = {"date", "stock_id", *FEATURES}
            missing = required.difference(reader.fieldnames or []) #完整性检查
            if missing:
                raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
            for row in tqdm(reader, desc="Loading day feature CSV", unit="row"):
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
                feature_rows += 1
        
        self._report(f"CSV rows parsed in {perf_counter() - csv_started:.1f}s.", logger)
        if not observations:
            raise ValueError("The day CSV contains no observations.")

        #处理pool_mask
        self._report("Loading aligned daily pool mask...", logger)
        mask_started = perf_counter()
        mask_rows = 0
        with open(pool_mask_path, encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required = {"date", "stock_id", "in_pool"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing mask CSV columns: {', '.join(sorted(missing))}")
            for row in tqdm(reader, desc="Loading pool mask CSV", unit="row"):
                try:
                    day = date.fromisoformat(row["date"].strip())
                    stock = row["stock_id"].strip()
                    in_pool = int(row["in_pool"].strip())
                except (AttributeError, ValueError) as error:
                    raise ValueError(f"Invalid mask observation on CSV line {reader.line_num}") from error
                if in_pool not in (0, 1):
                    raise ValueError(f"Mask in_pool must be 0 or 1 on CSV line {reader.line_num}")
                
                #对齐掩码和行情数据
                daily = observations.get(day)
                if daily is None or stock not in daily:
                    raise ValueError(
                        f"Mask key has no matching feature row on CSV line {reader.line_num}: "
                        f"{day.isoformat()}, {stock}"
                    )
                if not isinstance(daily[stock], list):
                    raise ValueError(f"Duplicate mask key on CSV line {reader.line_num}: {day}, {stock}")
                daily[stock] = (daily[stock], bool(in_pool))
                mask_rows += 1
        if mask_rows != feature_rows:
            raise ValueError(
                f"Feature and mask row counts differ: {feature_rows} features, {mask_rows} masks"
            )
        self._report(f"Pool mask parsed in {perf_counter() - mask_started:.1f}s.", logger)

        tensor_started = perf_counter()
        self.dates = sorted(observations)
        self.stocks = sorted({stock for daily in observations.values() for stock in daily})
        index = {stock: i for i, stock in enumerate(self.stocks)}
        self.daily = []
        self.stock_rows = []
        self.pool_stock_rows = []
        for day in self.dates:
            daily = observations[day]
            missing_masks = [stock for stock, value in daily.items() if isinstance(value, list)]
            if missing_masks:
                raise ValueError(
                    f"Feature rows without a mask on {day.isoformat()}, e.g. {missing_masks[0]}"
                )
            ids = sorted(index[stock] for stock in daily)
            self.daily.append(
                torch.tensor([daily[self.stocks[i]][0] for i in ids], dtype=torch.float32)
            )
            self.stock_rows.append(torch.tensor(ids, dtype=torch.long)) #记录N
            pool_ids = sorted(index[stock] for stock in daily if daily[stock][1])
            self.pool_stock_rows.append(torch.tensor(pool_ids, dtype=torch.long))
        self._report(f"Loaded {len(self.dates):,} dates and {len(self.stocks):,} stocks.", logger)
        self._report(
            f"Panel tensors built in {perf_counter() - tensor_started:.1f}s; "
            f"total load time {perf_counter() - panel_started:.1f}s.",
            logger,
        )

    @staticmethod
    def _report(message, logger):
        print(message)
        if logger is not None:
            logger.info(message)

    # 返回第 end 个交易日属于 advanceWholeMarketNew 的股票全局 ID。
    def pool_rows(self, end):
        return self.pool_stock_rows[end]

    # 返回当日池内股票的完整原始行情，用于只在训练期池内样本上拟合预处理参数。
    def pool_features(self, end):
        ids = self.pool_stock_rows[end]
        positions = torch.searchsorted(self.stock_rows[end], ids)
        return self.daily[end][positions]

    # 按股票对齐历史窗口，缺失观测保留为 NaN。
    def window(self, end, length, stock_ids=None):
        ids = self.stock_rows[end] if stock_ids is None else stock_ids
        if ids.ndim != 1:
            raise ValueError("stock_ids must be a one-dimensional tensor.")
        positions = torch.full((len(self.stocks),), -1, dtype=torch.long) #-1初始化，这只股票不在当前交易日的股票集合中，或者尚未找到对应位置。
        positions[ids] = torch.arange(ids.numel())
        window = torch.full((ids.numel(), length, len(FEATURES)), float("nan"))
        for step, day in enumerate(range(max(0, end - length + 1), end + 1), max(0, length - end - 1)):
            rows = positions[self.stock_rows[day]]
            present = rows >= 0
            window[rows[present], step] = self.daily[day][present]
        return ids, window

    # 按同一股票计算 T+1 至 T+11 的收盘收益率。
    def target(self, end, ids, start_offset, end_offset):
        closes = []
        for day in (end + start_offset, end + end_offset):
            prices = torch.full((len(self.stocks),), float("nan"))
            prices[self.stock_rows[day]] = self.daily[day][:, FEATURES.index("close")]
            closes.append(prices[ids])
        start, finish = closes
        return torch.where((start > 0) & (finish > 0), finish / start - 1, torch.nan)
