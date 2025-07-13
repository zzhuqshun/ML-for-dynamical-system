import tensorflow as tf

class ConvAE(tf.keras.Model):
    """
    卷积自编码器（Convolutional AutoEncoder, CAE）
    用于输入图像的低维表示学习和重构
    """
    def __init__(self, latent_dim):
        super(ConvAE, self).__init__()
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(75, 300, 1)),
            tf.keras.layers.Conv2D(16, 3, padding='same'),  # 卷积
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.MaxPooling2D(3),                # 池化
            tf.keras.layers.Conv2D(32, 5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.MaxPooling2D(5),
            tf.keras.layers.Conv2D(64, 5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.MaxPooling2D(5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim)               # 输出潜变量
        ])

        # 解码器
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(1 * 4 * 64),
            tf.keras.layers.Reshape((1, 4, 64)),
            tf.keras.layers.Conv2DTranspose(64, 5, strides=5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(64, 5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(32, 5, strides=5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(32, 5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(16, 3, strides=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(16, 3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid')
        ])

    def call(self, x):
        """
        前向传播
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VariationalAutoencoder(tf.keras.Model):
    """
    变分自编码器（VAE），输出均值和log方差
    """
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(75, 300, 1)),
            tf.keras.layers.Conv2D(16, 3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.MaxPooling2D(3),
            tf.keras.layers.Conv2D(32, 5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.MaxPooling2D(5),
            tf.keras.layers.Conv2D(64, 5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.MaxPooling2D(5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim * 2)  # 输出均值和log方差
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(1 * 4 * 64),
            tf.keras.layers.Reshape((1, 4, 64)),
            tf.keras.layers.Conv2DTranspose(64, 5, strides=5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(64, 5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(32, 5, strides=5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(32, 5, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(16, 3, strides=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(16, 3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid')
        ])

    def sample(self, mean, log_var):
        """
        重参数采样（reparameterization trick）
        """
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def call(self, x):
        """
        前向传播，返回重构图像
        """
        z_mean_log_var = self.encoder(x)
        z_mean, z_log_var = tf.split(z_mean_log_var, num_or_size_splits=2, axis=1)
        z = self.sample(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        # KL散度正则化损失
        beta = 0.1
        kl_loss = beta * (-0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        self.add_loss(tf.reduce_mean(kl_loss))
        return reconstructed
