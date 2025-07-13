# Physical Field Autoencoder Pipeline

A modular deep learning project for physical field data reconstruction based on convolutional autoencoders and parameter-to-latent mapping.

> 项目简介：本项目用于研究和复现实验“基于卷积自编码器的物理场重构”，可支持自定义网络结构、物理条件与潜空间映射、场数据可视化和大规模数据集批量实验。

---

## Directory Structure

```
your-project/
├── data/                  # (建议)原始和预处理数据
├── models/                # 网络结构定义（CAE, VAE等）
│   └── autoencoder.py
├── utils/                 # 工具函数（数据加载、归一化等）
│   └── dataloader.py
├── scripts/               # 主流程脚本
│   ├── train.py           # 自编码器训练
│   ├── evaluate.py        # 重构效果评估
│   ├── train_mapping.py   # FCNN条件-潜空间映射训练
│   └── evaluate_mapping.py # 映射重构可视化
├── checkpoints/           # 训练保存的模型文件
├── requirements.txt
└── README.md
```

---

## Environment

### Recommended

- Python >=3.8
- Ubuntu 20.04/22.04, Windows 10/11, 或 macOS (建议使用Anaconda环境)
- GPU (推荐RTX系列显卡) + CUDA 11.x（如用GPU训练）
- TensorFlow >=2.8（或 tf-gpu >=2.8）

### Install dependencies

```bash
pip install -r requirements.txt
```
如需Jupyter开发：
```bash
pip install jupyter ipython
```

---

## Quick Start

1. **Prepare your data**

   - 将你的 `.vtk` 文件放入 `./Data` 文件夹（结构可参考 `utils/dataloader.py` 说明）
   - 如无数据，请联系项目维护者

2. **Train autoencoder models**

   ```bash
   python scripts/train.py
   ```

3. **Evaluate reconstruction**

   ```bash
   python scripts/evaluate.py
   ```

4. **Train and evaluate mapping network**

   ```bash
   python scripts/train_mapping.py
   python scripts/evaluate_mapping.py
   ```

5. **(Optional) Use main.py for unified entry**

   ```bash
   python main.py --mode train
   ```

---

## Configuration

- 主要训练参数（latent_dim, epochs, batch_size 等）可在每个脚本开头直接调整。
- 支持自定义网络层数、数据增强倍数等，详情见脚本内部注释。

---

## Advanced Usage

- 支持不同 latent 维度（如 latent=4, latent=8）和不同模型结构的批量对比。
- 可扩展更复杂的网络、更多自定义 loss、结合物理损失项等。
- 可按需加 config.yaml 实现参数管理/批量实验。

---

## Citation

If you use this repository in your research, please cite:

> [Your Name], "Time Series Driven Incremental Deep Learning Model for State Estimation in Batteries", [Your University], 202X.

---

## Contact

- Maintainer: Qianshun Zhu
- Email: qianshun.zhu@campus.tu-berlin.de

---
