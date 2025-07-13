import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.autoencoder import ConvAE, VariationalAutoencoder
from utils.dataloader import load_vtk_data, remove_outliers, normalize_data, split_data, augment_data


# ========================
# Parameters and Paths
# ========================

data_path = './Data'
path_sep = '\\'  # Use '/' for Linux/MacOS

# Mesh grid parameters (same as your notebook)
x = np.arange(0.006, 0.0135, (0.0135 - 0.006) / 300)
y = np.arange(0, 0.0025, 0.0025 / 75)

outlier_indices = [10, 88, 100]

# Training hyperparameters
latent_dim = 8
epochs = 200
batch_size = 8
aug_times = 5
test_ratio = 0.15

# Model save paths
os.makedirs('./checkpoints', exist_ok=True)
cae_path = './checkpoints/cae_latent8'
cae_aug_path = './checkpoints/cae_aug_latent8'
vae_path = './checkpoints/vae_latent8'

# ========================
# 1. Data Loading & Preprocessing
# ========================
print("Step 1: Loading and processing data ...")
Qdot, condition = load_vtk_data(data_path, x, y, path_sep=path_sep)
print(f"Total samples: {Qdot.shape[0]}, Original condition shape: {condition.shape}")

Qdot, condition = remove_outliers(Qdot, condition, outlier_indices)
print(f"Samples after outlier removal: {Qdot.shape[0]}")

Qdot_norm, condition_norm, normaliser = normalize_data(Qdot, condition)
print(f"Qdot normalized shape: {Qdot_norm.shape}")

train_data, test_data, label_train, label_test = split_data(Qdot_norm, condition_norm, test_size=test_ratio)
print(f"Training samples: {train_data.shape[0]}, Test samples: {test_data.shape[0]}")

train_data_aug, label_train_aug = augment_data(train_data, label_train, times=aug_times)
print(f"Augmented training samples: {train_data_aug.shape[0]}")

# ========================
# 2. Model Definition & Training
# ========================

print("\nStep 2.1: Training vanilla CAE ...")
cae = ConvAE(latent_dim)
cae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
history_cae = cae.fit(train_data, train_data,
                      epochs=epochs, batch_size=batch_size,
                      validation_data=(test_data, test_data),
                      verbose=2)
cae.save(cae_path)
print(f"CAE model saved at: {cae_path}")

print("\nStep 2.2: Training CAE with augmentation ...")
cae_aug = ConvAE(latent_dim)
cae_aug.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
history_cae_aug = cae_aug.fit(train_data_aug, train_data_aug,
                              epochs=epochs, batch_size=batch_size,
                              validation_data=(test_data, test_data),
                              verbose=2)
cae_aug.save(cae_aug_path)
print(f"Augmented CAE model saved at: {cae_aug_path}")

print("\nStep 2.3: Training VAE ...")
vae = VariationalAutoencoder(latent_dim)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy')
history_vae = vae.fit(train_data_aug, train_data_aug,
                      epochs=epochs, batch_size=batch_size,
                      validation_data=(test_data, test_data),
                      verbose=2)
vae.save(vae_path)
print(f"VAE model saved at: {vae_path}")

# ========================
# 3. Plotting Loss Curves
# ========================
def plot_loss(history, title):
    """
    绘制训练/验证损失曲线
    """
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\nStep 3: Plotting loss curves ...")
plot_loss(history_cae, 'Vanilla CAE Loss')
plot_loss(history_cae_aug, 'Augmented CAE Loss')
plot_loss(history_vae, 'VAE Loss')

print("\nAll training steps are completed. Models are saved in ./checkpoints/")
