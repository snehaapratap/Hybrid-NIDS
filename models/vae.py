from tensorflow import keras
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable
import tensorflow as tf
from tensorflow.keras import ops

@register_keras_serializable()
class VAE(Model):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder_dense_1 = layers.Dense(64, activation='relu')
        self.encoder_dense_2 = layers.Dense(32, activation='relu')
        self.z_mean_layer = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var_layer = layers.Dense(latent_dim, name='z_log_var')

        self.decoder_dense_1 = layers.Dense(32, activation='relu')
        self.decoder_dense_2 = layers.Dense(64, activation='relu')
        self.decoder_output = layers.Dense(input_dim, activation='sigmoid')

    def encode(self, inputs):
        x = self.encoder_dense_1(inputs)
        x = self.encoder_dense_2(x)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        return z_mean, z_log_var

    def decode(self, z):
        d = self.decoder_dense_1(z)
        d = self.decoder_dense_2(d)
        outputs = self.decoder_output(d)
        return outputs

    def sampling(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.sampling(z_mean, z_log_var)
        outputs = self.decode(z)
        return outputs

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(data)
            z = self.sampling(z_mean, z_log_var)
            reconstruction = self.decode(z)

            reconstruction_loss = ops.mean(ops.square(data - reconstruction), axis=-1) * self.input_dim
            kl_loss = -0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1)
            total_loss = ops.mean(reconstruction_loss + kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "reconstruction_loss": ops.mean(reconstruction_loss), "kl_loss": ops.mean(kl_loss)}

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop('input_dim')
        latent_dim = config.pop('latent_dim')
        return cls(input_dim, latent_dim, **config)
