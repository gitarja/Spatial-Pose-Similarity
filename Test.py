import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from DataGenerator import Dataset
from Conf import TENSOR_BOARD_PATH
import datetime
from DNNModel.SimilarityModel import SimilarityModel
from Utils.Libs import kNN
from Utils.CustomLoss import CentroidTriplet
import numpy as np
import random
import tensorflow_probability as tfp
import glob
from sklearn.metrics import euclidean_distances


def knn_class(q_logits, labels, ref_logits, ref_labels, shots=5):
    N, D = ref_logits.shape
    # ref_logits = tf.reduce_mean(tf.reshape(ref_logits, (len(labels), shots, D)), 1)
    # ref_logits = tfp.stats.percentile(tf.reshape(ref_logits, (len(labels), shots, D)), 50.0, axis=1)
    acc = kNN(q_logits, labels, ref_logits, ref_labels, ref_num=1)
    return acc




# dataset
test_dataset = Dataset(mode="test")
shots = 5


#checkpoint
models_path = "D:\\usr\\pras\\result\\Spatial-memory-pose\\*_double"
random.seed(2021)
num_classes = 4
data_test = [test_dataset.get_batches(shots=shots, num_classes=num_classes) for i in range(1000)]

for models in sorted(glob.glob(models_path)):
    checkpoint_path = models + "/model"
    print(models)

    correlation = False
    #model
    model = SimilarityModel(filters=32, z_dim=64)
    # check point

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model)

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=100)
    all_acc = []
    for j in range(1, 2, 1):
        checkpoint.restore(manager.checkpoints[-j])

        acc_avg = []
        for i in range(len(data_test)):
            query, labels, references, ref_labels = data_test[i]
            q_logits = model(query)
            ref_logits = model(references)

            acc = knn_class(q_logits, labels, ref_logits, ref_labels, shots)

            acc_avg.append(acc)
        all_acc.append((np.average(acc_avg)))

    print(np.max(all_acc))
