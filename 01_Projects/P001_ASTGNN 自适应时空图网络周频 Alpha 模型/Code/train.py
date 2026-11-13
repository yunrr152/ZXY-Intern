import torch

from config import ModelConfig
from models.astgnn import ASTGNN


def main():
    config = ModelConfig()
    num_stocks = 20
    X = torch.randn(num_stocks, config.seq_len, config.num_days, config.num_features)
    model = ASTGNN(num_stocks=num_stocks, config=config)
    model.eval()
    with torch.no_grad():
        outputs = model(X)

    print("输入数据:", X.shape)
    print("ResNet输出:", outputs["X"].shape)
    print("GRU输出:", outputs["G"].shape)
    print("TGC输出:", outputs["T"].shape)
    print("NN因子矩阵:", outputs["Z"].shape)
    print("最终预测:", outputs["C"].shape)


if __name__ == "__main__":
    main()
