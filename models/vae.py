from tensorflow import keras
from keras import layers
from keras.saving import register_keras_serializable

@register_keras_serializable()
class VAE(keras.Model):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
        ])
        self.decoder = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(input_dim, activation="sigmoid"),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
