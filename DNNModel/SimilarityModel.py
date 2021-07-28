import tensorflow.keras as K
import tensorflow as tf
class ConvBlock(K.layers.Layer):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.dense = K.layers.Conv1D(filters, 5, padding="same", use_bias=True)
        self.batch = K.layers.BatchNormalization()
        self.relu = K.layers.ReLU()
        self.max = K.layers.MaxPool1D(pool_size=3)

    def call(self, inputs, **kwargs):
        x = self.batch(self.dense(inputs))
        x = self.max(self.relu(x))
        return x
class SimilarityModel(K.models.Model):


    def __init__(self, filters=64, z_dim=64):
        super(SimilarityModel, self).__init__()

        self.conv_1 = ConvBlock(filters=filters)
        self.conv_2 = ConvBlock(filters=filters)

        #projector
        self.dense = K.layers.Dense(z_dim, activation=None, use_bias=True)
        self.normalize = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))
        self.flat = K.layers.Flatten()





    def call(self, inputs, training=None, mask=None):
        z = self.conv_1(inputs)
        z = self.conv_2(z)
        z = self.flat(z)
        z = self.normalize(self.dense(z))
        return z



if __name__ == '__main__':
    model = SimilarityModel()
    X = tf.random.uniform((3, 45, 42))
    z = model(X)
