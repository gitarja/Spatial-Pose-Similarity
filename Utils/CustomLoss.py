import tensorflow as tf
import tensorflow.keras as K
from tensorflow.python.framework import ops
import tensorflow_probability as tfp

def off_diagonal(x):
    n = tf.shape(x)[0]
    flattened = tf.reshape(x, [-1])[:-1]
    off_diagonals = tf.reshape(flattened, (n-1, n+1))[:, 1:]
    return tf.reshape(off_diagonals, [-1])
def normalize(x):
        return (x - tf.reduce_mean(x, 0)) / tf.math.reduce_std(x, 0)

def l2_dis(x1, x2):
    return tf.reduce_sum(tf.square(x1 - x2), 1)

def log_cosh_dis(x1, x2):
    return tf.reduce_sum(tf.math.log(tf.math.cosh(x1 - x2)), 1)

def get_inner_dist(x, s, i):
    loss = x[(i)*s:(i+1)*s]
    return

def squared_dist(A):
    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(A, 0)
    distances = tf.reduce_sum(tf.math.squared_difference(expanded_a, expanded_b), 2)
    return distances


class CentroidTriplet():

    def __init__(self, margin=0.5, soft=False, n_shots=5, mean=False):
        super(CentroidTriplet, self).__init__()
        self.margin = margin
        self.n_shots = n_shots
        self.soft = soft
        self.mean = mean


    def __call__(self, embed, n_class=5):
        embed = ops.convert_to_tensor_v2(embed,  name="embed")
        # comver tensor
        embed = tf.cast(embed, tf.float32)
        N, D = embed.shape
        res_embed = tf.reshape(embed, (n_class, self.n_shots, D))

        # centroids = tf.reduce_mean(res_embed, 1, keepdims=True)
        if self.mean:
            centroids = tf.reduce_mean(res_embed, 1, keepdims=True)
            dist_to_cens = tf.reshape(tf.reduce_sum(tf.square(tf.expand_dims(res_embed, 1) - centroids), -1),
                                      (n_class, n_class * self.n_shots))
        else:
            centroids = tfp.stats.percentile(res_embed, 50.0, 1, keepdims=True)
            dist_to_cens = tf.reshape(tf.reduce_sum(tf.square(tf.expand_dims(res_embed, 1) - centroids), -1),
                                      (n_class, n_class * self.n_shots))

        d = tf.reshape(tf.repeat(tf.eye(n_class), self.n_shots), (n_class, n_class * self.n_shots))
        inner_dist  = tf.reshape(tf.boolean_mask(dist_to_cens, d==1), (N, -1))
        extra_dist = tf.reduce_min(tf.reshape(tf.boolean_mask(dist_to_cens, d == 0), (n_class, n_class-1, self.n_shots)), axis=1)
        triplet_loss = tf.maximum(0., self.margin + inner_dist - tf.reshape(extra_dist, (N, -1)))


        if self.soft:
            triplet_loss = tf.square(triplet_loss)



        return triplet_loss



if __name__ == '__main__':
    import numpy as np

    # a = tf.transpose(tf.Variable([[1, 1, 1, 1, 2,2,2,2, 3,3,3,3]]))
    a =tf.random.normal((12, 1))
    loss = CentroidTriplet(n_shots=4)
    loss(a, n_class=3)

