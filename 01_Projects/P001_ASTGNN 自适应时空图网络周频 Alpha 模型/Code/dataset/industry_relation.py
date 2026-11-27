"""Build an industry prior from a date-by-stock classification table."""

from bisect import bisect_right
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class IndustryRelations:
    #读取行业分类数据并初始化
    def __init__(self, path, *, missing_codes=()):
        table = pd.read_parquet(path, engine="pyarrow")
        dates = table.pop("date") if "date" in table.columns else table.index
        dates = pd.to_datetime(dates.astype(str), format="%Y%m%d")
        self.classifications = table.set_index(dates, verify_integrity=True)
        self.missing_codes = tuple(missing_codes)

    def for_day(self, day: date, stock_codes):
        """Return (adjacency, valid_pairs) in the supplied stock order."""
        stamp = pd.Timestamp(day)
        if stamp not in self.classifications.index:
            raise KeyError(f"No industry classification row for {day}.")

        #提取指定日的股票行业分类
        daily_codes = self.classifications.loc[stamp].reindex(list(stock_codes))
        daily_codes = daily_codes.replace(r"^\s*$", pd.NA, regex=True)
        daily_codes = daily_codes.mask(daily_codes.isin(self.missing_codes))
        codes, _ = pd.factorize(daily_codes)
        codes = torch.from_numpy(codes)
        known = codes >= 0
        #构建有效关系掩码
        valid_pairs = known[:, None] & known[None, :]
        valid_pairs.fill_diagonal_(False)
        adjacency = (codes[:, None] == codes[None, :]) & valid_pairs
        return adjacency, valid_pairs


class SavedIndustryPrior:
    def __init__(self, directory):
        self.directory = Path(directory)
        if not (self.directory / "_complete").is_file():
            raise ValueError(f"Industry prior export is incomplete: {self.directory}")
        manifest = self.directory / "snapshots.json"
        self.snapshots = None
        if manifest.is_file():
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            self.start = date.fromisoformat(metadata["start"])
            self.end = date.fromisoformat(metadata["end"])
            self.snapshots = [date.fromisoformat(value) for value in metadata["dates"]]
        self.cached_day = None

    def for_day(self, day: date, stock_codes):
        #根据交易日定位对应的月度快照
        if self.snapshots is not None:
            if not self.start <= day <= self.end:
                raise ValueError(f"Industry prior does not cover {day}.")
            day = self.snapshots[bisect_right(self.snapshots, day) - 1]
        #定位对应npz文件
        path = self.directory / f"{day:%Y%m%d}.npz"
        #每次读取都缓存一下，月度发生变化，才重新读取，
        if self.cached_day != day:
            with np.load(path, allow_pickle=False) as saved:
                self.stock_index = pd.Index(saved["stock_codes"])
                self.matrices = (saved["adjacency"], saved["valid_pairs"])
            self.cached_day = day
        
        #股票顺序对齐，找模型股票在原始矩阵中的位置
        ids = self.stock_index.get_indexer(stock_codes)
        if (ids < 0).any():
            missing = np.asarray(stock_codes)[ids < 0]
            raise ValueError(f"Industry prior {path} omits model stocks: {missing[:5]}.")

        # Slice and decode the stored matrices in the model's stock order.
        return tuple(
            torch.from_numpy(((matrix[np.ix_(ids, ids // 8)] >> (7 - ids % 8)) & 1).astype(bool))
            for matrix in self.matrices
        )
