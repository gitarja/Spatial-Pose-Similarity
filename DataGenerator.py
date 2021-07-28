import pickle
import pandas as pd
import numpy as np
import random
from Conf import DATASET_PATH


class Dataset:

    def __init__(self, mode="train", val_frac=0.1):

        #load data
        if mode == "train":
            with open(DATASET_PATH + "train.pkl", 'rb') as f:
                ds = pickle.load(f)
        else:
            with open(DATASET_PATH + "test.pkl", 'rb') as f:
                ds = pickle.load(f)
        #load mean
        with open(DATASET_PATH + "mean.pkl", 'rb') as f:
            mean = pickle.load(f)
        with open(DATASET_PATH + "std.pkl", 'rb') as f:
            std = pickle.load(f)
        #load std

        self.data = {}
        def normalize(x):
            return (x - mean) /std
        for i in range(ds.shape[0]):
            for l in range(ds[i].shape[0]):
                label = str(i)
                if label not in self.data:
                    self.data[label] = []
                self.data[label].append(normalize(ds[i, l, :, :]))

        self.labels = list(self.data.keys())


    def get_mini_offline_batches(self, n_class, shots=2):
        anchor_positive = np.zeros(shape=(n_class * shots, 45, 42))
        labels = self.labels

        label_subsets = random.sample(labels, k=n_class)
        for i in range(len(label_subsets)):
            positive_to_split = random.sample(
                self.data[label_subsets[i]], k=shots)

            # set anchor and pair positives

            anchor_positive[i * shots:(i + 1) * shots] = positive_to_split

        return anchor_positive.astype(np.float32)

    def get_batches(self, shots, num_classes, outlier=False):

        temp_labels = np.zeros(shape=(num_classes))
        temp_images = np.zeros(shape=(num_classes, 45, 42))
        ref_labels = np.zeros(shape=(num_classes * shots))
        ref_images = np.zeros(shape=(num_classes * shots, 45, 42))


        if outlier == False:
            label_subsets = random.sample(self.labels, k=num_classes)
            for class_idx, class_obj in enumerate(label_subsets):
                temp_labels[class_idx] = class_idx
                ref_labels[class_idx * shots: (class_idx + 1) * shots] = class_idx

                # sample images

                images_to_split = random.sample(
                    self.data[label_subsets[class_idx]], k=shots + 1)
                temp_images[class_idx] = images_to_split[-1]
                ref_images[class_idx * shots: (class_idx + 1) * shots] = images_to_split[:-1]
        else:
            # generate support
            support_labels = random.sample(self.labels[:int(len(self.labels)/2)], k=num_classes)
            for class_idx, class_obj in enumerate(support_labels):
                ref_labels[class_idx * shots: (class_idx + 1) * shots] = class_idx
                ref_images[class_idx * shots: (class_idx + 1) * shots] = random.choices(
                    self.data[support_labels[class_idx]], k=shots)

            # generate query
            query_labels = random.sample(self.labels[int(len(self.labels) / 2):], k=num_classes)
            for class_idx, class_obj in enumerate(query_labels):
                temp_labels[class_idx ] = class_idx
                ref_images[class_idx] = random.choices(self.data[query_labels[class_idx]])

        return temp_images.astype(np.float32), temp_labels.astype(np.int32), ref_images.astype(
            np.float32), ref_labels.astype(np.float32)

if __name__ == '__main__':
   dataset = Dataset()
   X = dataset.get_mini_offline_batches(n_class=4, shots=5)

   print(X.shape)
#
# def get_samples(class_num = 1, shots = 5):
#     with open("dataset.pkl", "rb") as f:
#         dataset = pickle.load(f)
#     sample = []
#     classes = [0, 1, 2, 3]
#     c_classes = random.sample(classes, class_num)
#     print(c_classes)
#     for i in c_classes:
#         data = dataset[i]
#         tmp = []
#         for i in range(shots):
#             tmp.append(data[random.randrange(len(data))])
#         sample.append(tmp)
#     return np.array(sample).astype(np.float)
#
# # get_samples()
# print(get_samples(2).shape)