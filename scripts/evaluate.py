import os
import numpy as np
import matplotlib.pyplot as plt
from models.autoencoder import ConvAE, VariationalAutoencoder
from utils.dataloader import load_vtk_data, remove_outliers, normalize_data, split_data
import tensorflow as tf

# =========================
# 参数设定（需和train.py保持一致）
# =========================
data_path = './Data'
path_sep = '\\'
x = np.arange(0.006, 0.0135, (0.0135 - 0.006) / 300)
y = np.arange(0, 0.0025, 0.0025 / 75)
outlier_indices = [10, 88, 100]
latent_dim = 8
test_ratio = 0.15

# 模型权重路径
cae_path = './checkpoints/cae_latent8'
cae_aug_path = './checkpoints/cae_aug_latent8'
vae_path = './checkpoints/vae_latent8'

# =========================
# 1. 数据加载与准备
# =========================
print("Loading data for evaluation ...")
Qdot, condition = load_vtk_data(data_path, x, y, path_sep=path_sep)
Qdot, condition = remove_outliers(Qdot, condition, outlier_indices)
Qdot_norm, condition_norm, normaliser = normalize_data(Qdot, condition)
train_data, test_data, label_train, label_test = split_data(Qdot_norm, condition_norm, test_size=test_ratio)
print(f"Test set size: {test_data.shape[0]}")

# =========================
# 2. 加载模型
# =========================
print("Loading trained models ...")
cae = tf.keras.models.load_model(cae_path, compile=False)
cae_aug = tf.keras.models.load_model(cae_aug_path, compile=False)
vae = tf.keras.models.load_model(vae_path, compile=False)

# =========================
# 3. 生成重构结果
# =========================
print("Generating reconstructions ...")
encoded_test_cae = cae.encoder(test_data).numpy()
decoded_test_cae = cae.decoder(encoded_test_cae).numpy()
encoded_test_cae_aug = cae_aug.encoder(test_data).numpy()
decoded_test_cae_aug = cae_aug.decoder(encoded_test_cae_aug).numpy()
# VAE的编码器输出mean和log_var
encoded_test_vae = vae.encoder(test_data).numpy()
mean_vae, log_var_vae = np.split(encoded_test_vae, 2, axis=1)
epsilon = np.random.normal(size=mean_vae.shape)
latent_vae = mean_vae + np.exp(0.5 * log_var_vae) * epsilon
decoded_test_vae = vae.decoder(latent_vae).numpy()

# =========================
# 4. 可视化对比
# =========================
def show_reconstructions(test_data, decoded_cae, decoded_cae_aug, decoded_vae, n_samples=8):
    """
    可视化原图、CAE重构、增强CAE重构、VAE重构
    """
    plt.figure(figsize=(16, n_samples * 3))
    for i in range(n_samples):
        # 原始图片
        plt.subplot(n_samples, 4, i * 4 + 1)
        plt.imshow(test_data[i].squeeze())
        plt.title("Original")
        plt.axis('off')
        # CAE
        plt.subplot(n_samples, 4, i * 4 + 2)
        plt.imshow(decoded_cae[i].squeeze())
        plt.title("CAE Recon.")
        plt.axis('off')
        # 增强版CAE
        plt.subplot(n_samples, 4, i * 4 + 3)
        plt.imshow(decoded_cae_aug[i].squeeze())
        plt.title("Aug. CAE Recon.")
        plt.axis('off')
        # VAE
        plt.subplot(n_samples, 4, i * 4 + 4)
        plt.imshow(decoded_vae[i].squeeze())
        plt.title("VAE Recon.")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("Plotting reconstruction comparison ...")
show_reconstructions(test_data, decoded_test_cae, decoded_test_cae_aug, decoded_test_vae, n_samples=8)

print("Evaluation & visualization finished.")
