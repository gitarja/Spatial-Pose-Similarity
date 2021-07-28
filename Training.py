import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from DataGenerator import Dataset
from DNNModel.SimilarityModel import SimilarityModel
import numpy as np
import datetime
from Conf import TENSOR_BOARD_PATH
import argparse
from Utils.CustomLoss import CentroidTriplet

import random
import os

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
random.seed(2021)  # set seed
tf.random.set_seed(2021)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--margin', type=float, default=1.5)
    parser.add_argument('--soft', type=bool, default=False)
    parser.add_argument('--n_class', type=int, default=4)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--mean', type=bool, default=False)

    args = parser.parse_args()

    # set up GPUs
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)
    #
    # cross_tower_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=3)
    # strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_tower_ops)

    print(args.n_class)

    train_dataset = Dataset(mode="train")
    test_dataset = Dataset(mode="test")

    # training setting
    eval_interval = 1
    train_class = args.n_class

    batch_size = 256
    val_class = 4
    ref_num = 5
    val_loss_th = 1e+3
    shots = 50

    # training setting
    epochs = 300
    lr = 1e-3
    lr_siamese = 1e-3

    # early stopping
    early_th = 25
    early_idx = 0

    # siamese and discriminator hyperparameter values
    z_dim = args.z_dim

    # tensor board
    log_dir = TENSOR_BOARD_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_double"
    checkpoint_path = TENSOR_BOARD_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_double" + "/model"

    train_log_dir = log_dir + "/train"
    test_log_dir = log_dir + "/test"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # loss
    triplet_loss = CentroidTriplet(margin=args.margin, soft=args.soft, n_shots=shots, mean=args.mean)  # shots
    binary_loss = tf.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    model = SimilarityModel(filters=32, z_dim=z_dim)
    # check point
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model)

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=early_th)
    # optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr,
        decay_steps=50,
        decay_rate=0.7,
        staircase=True)
    siamese_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # metrics
    # train
    loss_train = tf.keras.metrics.Mean()
    # test
    loss_test = tf.keras.metrics.Mean()


    def compute_triplet_loss(embd, n_class):
        per_example_loss = triplet_loss(embd, n_class)
        return tf.reduce_mean(per_example_loss)


    def train_step(inputs):
        X = inputs
        with tf.GradientTape() as siamese_tape, tf.GradientTape():
            embeddings = model(X, training=True)

            embd_loss = compute_triplet_loss(embeddings, train_class)  # triplet loss
            loss = embd_loss

        # the gradients of the trainable variables with respect to the loss.
        siamese_grads = siamese_tape.gradient(loss, model.trainable_weights)
        siamese_optimizer.apply_gradients(zip(siamese_grads, model.trainable_weights))
        loss_train(loss)
        return loss


    def val_step(inputs):
        X = inputs
        embeddings = model(X, training=True)

        embd_loss = compute_triplet_loss(embeddings, val_class)  # triplet loss
        loss = embd_loss

        loss_test(loss)
        return loss


    def reset_metric():
        loss_train.reset_states()
        loss_test.reset_states()


    # `experimental_run_v2` replicates the provided computation and runs it
    # with the distributed input.

    # validation dataset
    val_inputs = test_dataset.get_mini_offline_batches(val_class,
                                                        shots=shots,

                                                        )

    for epoch in range(epochs):
        # dataset
        train_inputs = train_dataset.get_mini_offline_batches(train_class,
                                                              shots=shots,

                                                              )

        train_step(train_inputs)

        if (epoch + 1) % eval_interval == 0:

            val_step(val_inputs)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_train.result().numpy(), step=epoch)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', loss_test.result().numpy(), step=epoch)
            print("Training loss=%f, validation loss=%f" % (
                loss_train.result().numpy(), loss_test.result().numpy()))  # print train and val losses

            val_loss = loss_test.result().numpy()
            if (val_loss_th > val_loss):
                val_loss_th = val_loss
                manager.save()
                early_idx = 0

            else:
                early_idx += 1
            reset_metric()
