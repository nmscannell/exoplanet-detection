import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
#from imblearn.over_sampling import RandomOverSampler
from dataset import build_dataset
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def build_astro_sets(batch_size):
    files = 'data/train1, data/train2, data/train3, data/train4, data/train5, data/train6, data/train7, data/train8'
    train = build_dataset(files, batch_size, reverse_time_series_prob=0.5, shuffle_filenames=True)
    val = build_dataset('data/val', batch_size, reverse_time_series_prob=0.5, shuffle_filenames=True)
    test = build_dataset('data/test', 1)
    return train, val, test


def build_sets():
    # 5087 obs in train, 570 obs in test
    data = np.loadtxt('data/exoTrain.csv', skiprows=1, delimiter=',')
    x_train = data[:, 1:]
    y_train = data[:, 0, np.newaxis] - 1
    data = np.loadtxt('data/exoTest.csv', skiprows=1, delimiter=',')
    x_test = data[:, 1:]
    y_test = data[:, 0, np.newaxis] - 1

    # center
    x_train = (x_train - np.mean(x_train, axis=1).reshape(-1, 1))/np.std(x_train, axis=1).reshape(-1, 1)
    x_test = (x_test - np.mean(x_test, axis=1).reshape(-1, 1))/np.std(x_test, axis=1).reshape(-1, 1)
    '''
    os = RandomOverSampler(sampling_strategy='minority')
    x_train, y_train = os.fit_sample(x_train, y_train)
    '''

    for i in range(37):
        sample = x_train[i]
        for j in range(136):  # 5050 non-exoplanets, 37 exoplanets
            idx = np.random.randint(x_train.shape[1])
            sample = np.roll(sample, idx, axis=0)
            x_train = np.append(x_train, [sample], axis=0)
            y_train = np.append(y_train, [[1]], axis=0)

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2, random_state=42)

    # give samples shape [timesteps, features]
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    x_val = np.expand_dims(x_val, axis=2)

    print(x_train.shape)
    print(x_test.shape)
    print(x_val.shape)
    return x_train, y_train, x_val, y_val, x_test, y_test
