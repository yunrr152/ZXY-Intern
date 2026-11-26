# ASTGNN 日频滚动预测

从日行情和每日股票池构建数据，通过 ASTGNN 提取因子，再用 LightGBM 生成逐日、逐股分数。集成入口用三个主种子独立训练，并平均最终分数。

## 环境与输入

在服务器上的 `Code` 目录运行命令。建议使用 Python 3.10 独立环境，安装与服务器驱动匹配的 CUDA 版 PyTorch，以及：

```bash
python -m pip install numpy pandas pyarrow tqdm lightgbm
python -c "import torch, pandas, pyarrow, tqdm, lightgbm; print(torch.cuda.is_available())"
```

使用 `--device cuda` 时，上述检查应输出 `True`。仓库现有的 `requirements.txt` 和 `environment.yml` 是 Windows 环境快照，包含本地路径或 Windows 专用包，以及 CPU 版 PyTorch，不适合直接用于 Linux GPU 环境。

`--source-root` 下应有：

```text
dataset/data/
├── LX_STK_BASIC_DAILY/
└── LX_STK_POOL_BASIC/
    └── advanceWholeMarketNew.parquet
```

构建器保留正常交易记录（`tradeStatus == 1`），生成六列日行情特征（high、open、low、close、vwap、volume）与逐行对齐的每日股票池。构建器不拟合预处理参数。

## 一键运行

以下为 Linux Bash 示例。日期为包含边界的 `YYYYMMDD`，输出目录必须不存在。
注意，需要进入code目录再运行
预计训练时长为3个随机种子*9个预测年*3小时

```bash
python -u -m training.ensemble `
    --source-root "dataset\data" `
    --start-date 20060101 `
    --end-date 20261117 `
    --first-prediction-year 2017 `
    --last-prediction-year 2025 `
    --device cuda `
    --workers 3 `
    --seed-devices cuda:0 cuda:1 cuda:2 `
    --epochs 20 `
    --output-dir "dataset\output\ensemble_2017_2025" 
    #--resume 断点重跑
```
python -u -m training.train `
  --data dataset/output/day_features_2006_2026.csv `
  --mask dataset/output/day_pool_mask_2006_2026.csv `
  --first-prediction-year 2017 `
  --last-prediction-year 2026 `
  --seed 3407 `
  --lambda-gp 0 `
  --device cuda `
  --epochs 20 `
  --checkpoint-dir dataset/output/checkpoints_seed3407_no_gp_2017_2026 `
  --log-file dataset/output/seed3407_no_gp_2017_2026.log `
  --output dataset/output/predict.parquet 

这个示例生成 2017–2025 年结果。输入日期范围需要覆盖最早预测年的前十年，以及最后一个预测年所需的交易日。

## 数据与训练流程

1. 构建一次特征和股票池 CSV。默认三个种子用独立进程并行，各自加载一份日频面板；指定 `--workers 1` 时顺序训练，并复用内存中的同一份面板。
2. 每个预测年独立训练。默认取前十个自然年，例如预测 2018 年用 2008–2017 年；预处理统计量只在训练期拟合，标签终点跨入预测年的样本不参加训练。
3. 每个样本使用截至当天的最近 30 个交易日特征。标签为第 11 个后续交易日收盘价除以第 1 个后续交易日收盘价再减 1。每日股票池确定当日训练和预测截面。
4. 日频入口关闭 ResNet。模型使用单路 GRU 的末步隐藏状态，再经自适应图卷积提取 64 个因子。训练损失包含预测误差、因子正交惩罚与对抗预测误差。
5. 冻结 ASTGNN，用训练期末段（默认最近一年）的因子和标签拟合 LightGBM；预测年按交易日生成分数。打分使用当天行情，不是开盘前可用的分数。
6. 主种子为 `0`、`47`、`427`。各进程独立初始化，GRU 不单独设种子，LightGBM 使用当次主种子。相同种子不保证跨硬件和依赖版本逐位复现。

## 输出与恢复

| 文件 | 内容 |
| --- | --- |
| `predict1.parquet`、`predict2.parquet`、`predict3.parquet` | 对应种子 0、47、427 的分数 |
| `predict.parquet` | 三次分数按键对齐后等权平均 |
| `day_features.csv`、`day_pool_mask.csv` | 构建的训练输入 |
| `seed_*.log`、`checkpoints/seed_*/` | 按种子隔离的训练日志及按种子、年份保存的检查点 |

多 GPU 服务器可以追加 `--seed-devices cuda:0 cuda:1 cuda:2`，按配置顺序分配设备。不指定时所有进程都使用 `--device`，会争用同一 GPU 和主机内存；单卡或内存紧张时建议追加 `--workers 1`。并行度不改变结果文件与种子的对应关系，也可以在续跑时调整。

预测列为 `date`、`stock_id`、`validation_year`、`C`；`C` 是 LightGBM 分数。集成要求三份预测的键唯一且完全一致，不按文件行序直接平均。

首次运行使用新输出目录。中断后先确认原进程已停止，再用原命令、原输出目录和相同配置追加 `--resume`。程序复用完整的输入 CSV 和已完成的种子结果；未完成的种子按年份加载检查点，从最后保存的训练位置继续。已训完但尚未写出种子结果的年份会重新生成预测，不会重训已保存的轮次。恢复时仍需要重新加载面板。

新版使用 `ensemble_run.json` 检查运行参数和 CSV 文件大小。旧版运行若已有 `ensemble.log`，可以用原参数续训，但旧版没有配置记录，需要自行确认参数和数据未变；若中断在建数阶段，恢复时会重新构建数据。旧三分支 GRU 检查点不能加载到当前单路模型。

## 配置与代码入口

| 位置 | 负责内容 |
| --- | --- |
| `config.py` | 主要默认超参数：窗口、标签、模型维度、预处理、优化器、epoch、检查点频率、对抗训练和 LightGBM |
| `training/ensemble.py` | 读取 `ModelConfig.ensemble_seeds`，一键构建、训练和汇总；路径、日期、预测年度、设备和轮数由 CLI 指定 |
| `dataset/build_day_dataset.py` | 原始行情字段映射、股票池来源和独立构建入口 |
| `dataset/day.py`、`dataset/preprocessing.py` | 面板、窗口、标签和训练期预处理 |
| `models/astgnn.py`、`modules/`、`layers/` | 模型组装、单路 GRU、自适应图卷积、因子预测和可选 ResNet |
| `training/train.py`、`training/test.py` | 已有 CSV 上的单种子滚动训练和指定年度训练 |
| `training/lgbols.py`、`training/loss.py`、`ops/ifgsm.py` | LightGBM 拟合、损失和对抗扰动 |

集成种子在 `config.py` 的 `ensemble_seeds` 中配置；`random_seed=42` 是单种子入口的默认值，可用 `--seed` 覆盖。GRU 不再配置独立种子。日频入口显式关闭 ResNet，不采用配置中的 `use_resnet=True` 默认值。

其余入口可用 `python -m dataset.build_day_dataset --help`、`python -m training.train --help` 和 `python -m training.test --help` 查看参数。
