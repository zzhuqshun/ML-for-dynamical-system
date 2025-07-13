import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.autoencoder import ConvAE
from utils.dataloader import load_vtk_data, remove_outliers, normalize_data, split_data


# ================
# 参数设定
# ================
data_path = './Data'
path_sep = '\\'
x = np.arange(0.006, 0.0135, (0.0135 - 0.006) / 300)
y = np.arange(0, 0.0025, 0.0025 / 75)
outlier_indices = [10, 88, 100]
latent_dim = 8
test_ratio = 0.15

# 路径
cae_aug_path = './checkpoints/cae_aug_latent8'
mapping_path = './checkpoints/fcnn_bs8_lr1e-4_latent8_nlayers8'
cae_aug4_path = './checkpoints/cae_aug_latent4'     # 如有latent=4模型可以添加
mapping4_path = './checkpoints/fcnn_bs8_lr1e-4_latent4_nlayers8'

# ================
# 1. 加载数据与准备
# ================
print("Loading data ...")
Qdot, condition = load_vtk_data(data_path, x, y, path_sep=path_sep)
Qdot, condition = remove_outliers(Qdot, condition, outlier_indices)
Qdot_norm, condition_norm, normaliser = normalize_data(Qdot, condition)
train_data, test_data, label_train, label_test = split_data(Qdot_norm, condition_norm, test_size=test_ratio)

# ================
# 2. 加载模型
# ================
print("Loading models ...")
AUG = tf.keras.models.load_model(cae_aug_path, compile=False)
mapping = tf.keras.models.load_model(mapping_path, compile=False)

# 如有latent=4模型，加载
has_latent4 = False
if os.path.exists(cae_aug4_path) and os.path.exists(mapping4_path):
    AUG4 = tf.keras.models.load_model(cae_aug4_path, compile=False)
    mapping4 = tf.keras.models.load_model(mapping4_path, compile=False)
    has_latent4 = True

# ================
# 3. 映射条件→latent→解码场图像
# ================
def plot_mapping_results(idx_list=[137, 19, 32, 100, 170, 230]):
    """
    输入指定测试集索引，分别展示原图、mapping→latent8→解码、(可选)mapping4→latent4→解码
    """
    for idx in idx_list:
        params = np.expand_dims(condition_norm[idx], axis=0)
        # latent8预测与解码
        latent_data_8 = mapping(params)
        pred_img_8 = AUG.decoder(latent_data_8).numpy()

        # latent4预测与解码
        if has_latent4:
            latent_data_4 = mapping4(params)
            pred_img_4 = AUG4.decoder(latent_data_4).numpy()

        fig = plt.figure(figsize=(12, 8))
        # 原图
        ax = fig.add_subplot(3, 1, 1)
        plt.imshow(np.squeeze(Qdot_norm[idx]))
        plt.axis('off')
        plt.title(f"Original | ER: {condition[idx][0]*0.001:.2f}, T: {condition[idx][1]}, "
                  f"U: {condition[idx][2]*0.01:.2f}, WallT: {condition[idx][3]}")
        # latent8解码
        ax = fig.add_subplot(3, 1, 2)
        plt.imshow(np.squeeze(pred_img_8))
        plt.axis('off')
        plt.title("Predicted Image from Latent 8")
        # latent4解码（可选）
        if has_latent4:
            ax = fig.add_subplot(3, 1, 3)
            plt.imshow(np.squeeze(pred_img_4))
            plt.axis('off')
            plt.title("Predicted Image from Latent 4")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

print("Evaluating mapping on test samples ...")
# 你可以自定义 idx_list，比如 [137, 19, 32, 100, 170, 230]（和notebook保持一致）
plot_mapping_results([137, 19, 32, 100, 170, 230])

print("Mapping evaluation and visualization finished.")