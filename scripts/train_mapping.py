import os
import numpy as np
import matplotlib.pyplot as plt
from models.autoencoder import ConvAE
from utils.dataloader import load_vtk_data, remove_outliers, normalize_data, split_data, augment_data
import tensorflow as tf

# ================
# 参数设定
# ================
data_path = './Data'
path_sep = '\\'
x = np.arange(0.006, 0.0135, (0.0135 - 0.006) / 300)
y = np.arange(0, 0.0025, 0.0025 / 75)
outlier_indices = [10, 88, 100]
latent_dim = 8
aug_times = 5
test_ratio = 0.15
epochs = 1000
batch_size = 8

# 路径
cae_aug_path = './checkpoints/cae_aug_latent8'
mapping_save_path = './checkpoints/fcnn_bs8_lr1e-4_latent8_nlayers8'

# ================
# 1. 加载和准备数据
# ================
print("Loading and preparing data ...")
Qdot, condition = load_vtk_data(data_path, x, y, path_sep=path_sep)
Qdot, condition = remove_outliers(Qdot, condition, outlier_indices)
Qdot_norm, condition_norm, normaliser = normalize_data(Qdot, condition)
train_data, test_data, label_train, label_test = split_data(Qdot_norm, condition_norm, test_size=test_ratio)
train_data_aug, label_train_aug = augment_data(train_data, label_train, times=aug_times)

# ================
# 2. 加载已训练好的增强版CAE
# ================
print("Loading pretrained augmented CAE ...")
cae_aug = tf.keras.models.load_model(cae_aug_path, compile=False)

# ================
# 3. 生成latent向量标签
# ================
print("Encoding training and test data to latent space ...")
encoded_train_mapping = cae_aug.encoder(train_data).numpy()
encoded_test_mapping = cae_aug.encoder(test_data).numpy()

# ================
# 4. 定义FCNN映射网络
# ================
param_dim = condition_norm.shape[1]
num_layers = 8

print("Building FCNN mapping model ...")
mapping = tf.keras.Sequential(
    [tf.keras.layers.InputLayer(input_shape=(param_dim,))] +
    [tf.keras.layers.Dense(2 ** (n + 2), activation='relu') for n in range(1, num_layers + 1)] +
    [tf.keras.layers.Dense(latent_dim, activation='linear')]
)
mapping.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

# ================
# 5. 训练映射网络
# ================
print("Training mapping network (FCNN) ...")
hist_mapping = mapping.fit(
    label_train, encoded_train_mapping,
    epochs=epochs, batch_size=batch_size,
    validation_data=(label_test, encoded_test_mapping),
    verbose=2
)
mapping.save(mapping_save_path)
print(f"Mapping FCNN saved at: {mapping_save_path}")

# ================
# 6. 损失曲线可视化
# ================
plt.plot(hist_mapping.history['loss'], label='Train Loss')
plt.plot(hist_mapping.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'FCNN Latent dimension: {latent_dim}')
plt.legend()
plt.tight_layout()
plt.show()

print("Mapping network training finished.")

