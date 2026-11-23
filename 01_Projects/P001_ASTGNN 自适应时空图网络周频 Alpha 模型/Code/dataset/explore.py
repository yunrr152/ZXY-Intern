from pathlib import Path
import numpy as np
import pandas as pd

# 设置 Pandas 显示格式
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

# 修改为你的数据所在目录
ROOT = Path(__file__).resolve().parent / "data"

# 四个数据库的路径
TRADE_DAY = ROOT / "LX_GEN_TIME_TRADEDAY"
STOCK_INFO = ROOT / "LX_STK_INFO_AH"
STOCK_DAILY = ROOT / "LX_STK_BASIC_DAILY"
STOCK_POOL = ROOT / "LX_STK_POOL_BASIC"


# 读取交易日
trade_days = np.fromfile(
    TRADE_DAY / "tradeDay.dat",
    dtype="<i4"
)

# 转换为 DataFrame
df_calendar = pd.DataFrame({
    "trade_date": trade_days
})

print("数据形状：", df_calendar.shape)
print("最早交易日：", df_calendar["trade_date"].min())
print("最晚交易日：", df_calendar["trade_date"].max())

print("\n前10行：")
print(df_calendar.head(10))

print("\n后10行：")
print(df_calendar.tail(10))

print("\n数据类型：")
print(df_calendar.dtypes)

print("重复交易日数量：", df_calendar.duplicated().sum())
print("是否按日期升序排列：", df_calendar["trade_date"].is_monotonic_increasing)

#读取证券主表
info_files = [
    "AShareL.parquet",
    "AShareLSDE.parquet",
    "HShareL.parquet",
    "HShareLSDE.parquet"
]

stock_info_dict = {}

for filename in info_files:

    df = pd.read_parquet(STOCK_INFO / filename)
    stock_info_dict[filename] = df

    print(f"\n{'=' * 60}")
    print(f"文件：{filename}")
    print(f"{'=' * 60}")

    print("数据形状：", df.shape)

    print("\n字段名称：")
    print(df.columns.tolist())

    print("\n字段类型：")
    print(df.dtypes)

    print("\n前5行：")
    print(df.head())

    print("\n缺失比例（%）：")
    print((df.isna().mean() * 100).round(2))

    #读取股票日线数据（重点）
    # 获取所有股票目录
stock_dirs = sorted([
    p for p in STOCK_DAILY.iterdir()
    if p.is_dir()
])

print("股票数量：", len(stock_dirs))

print("\n前20只股票：")
for p in stock_dirs[:20]:
    print(p.name)

# 选择一只股票
stock_code = "000001.XSHE"
stock_path = STOCK_DAILY / stock_code

# 查看全部字段文件
dat_files = sorted(stock_path.glob("*.dat"))

print("\n字段数量：", len(dat_files))

print("\n全部字段：")
for file in dat_files:
    print(file.stem)

#4.2 读取一只股票的全部日线字段
# 字段类型映射
DTYPE_MAP = {
    "di": "<i4",
    "turnoverValue": "<f8",
    "marketValue": "<f8",
    "negMarketValue": "<f8"
}

def read_stock_daily(stock_code):

    folder = STOCK_DAILY / stock_code

    if not folder.is_dir():
        raise FileNotFoundError(f"股票目录不存在：{folder}")

    data = {}

    for file in sorted(folder.glob("*.dat")):

        field = file.stem

        dtype = DTYPE_MAP.get(field, "<f4")

        data[field] = np.fromfile(
            file,
            dtype=dtype
        )

    if "di" not in data:
        raise ValueError(f"{stock_code} 缺少 di.dat")

    n = len(data["di"])
    for field, values in data.items():
        if len(values) != n:
            raise ValueError(
                f"{stock_code} 字段 {field} 长度为 {len(values)}，"
                f"与日期轴长度 {n} 不一致"
            )

    df = pd.DataFrame(data)

    return df.sort_values("di").reset_index(drop=True)


# 读取平安银行
df_daily = read_stock_daily("000001.XSHE")

print("数据形状：", df_daily.shape)

print("\n前5行：")
print(df_daily.head())

print("\n后5行：")
print(df_daily.tail())

print("\n字段类型：")
print(df_daily.dtypes)

#4.3 查看日线数据的统计信息
# 描述性统计
print(df_daily.describe().T)

# 缺失情况
missing = pd.DataFrame({
    "缺失数量": df_daily.isna().sum(),
    "缺失比例(%)": df_daily.isna().mean() * 100
})

print(missing.sort_values("缺失比例(%)", ascending=False))

#筛选实际有交易的记录：
df_trading = df_daily[
    df_daily["tradeStatus"] == 1
].copy()

print("全部日期数量：", len(df_daily))
print("实际交易记录数量：", len(df_trading))

print("\n有效交易记录：")
print(df_trading[[
    "di",
    "openN",
    "highN",
    "lowN",
    "closeN",
    "closeA",
    "ret1d",
    "volumeN",
    "turnoverValue"
]].tail(20))

#4.4 可视化股票收盘价
import matplotlib.pyplot as plt

df_plot = df_trading.copy()

df_plot["date"] = pd.to_datetime(
    df_plot["di"].astype(str),
    format="%Y%m%d"
)

plt.figure(figsize=(12, 5))

plt.plot(
    df_plot["date"],
    df_plot["closeA"]
)

plt.title("000001.XSHE Adjusted Closing Price")
plt.xlabel("Date")
plt.ylabel("Adjusted Close")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#五、读取股票池
pool_files = sorted(STOCK_POOL.glob("*.parquet"))

print("股票池数量：", len(pool_files))

for file in pool_files:
    print(file.name)

#先选一个进行详细观察：
pool_path = STOCK_POOL / "advanceWholeMarketNew.parquet"

df_pool = pd.read_parquet(pool_path)

print("数据形状：", df_pool.shape)

print("\n前5行、前10列：")
print(df_pool.iloc[:5, :10])

print("\n最后5行、前10列：")
print(df_pool.iloc[-5:, :10])

print("\n索引类型：", df_pool.index.dtype)
print("股票数量：", len(df_pool.columns))
print("日期数量：", len(df_pool.index))