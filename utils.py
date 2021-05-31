import os
from scipy import io
import config
import random


def load_mat(dataset_index: int):
    train_kin = io.loadmat(config.TRAIN_DATA_DIRECTORY + os.sep + f"KinData{dataset_index}.mat")["KinData"]
    train_neural = io.loadmat(config.TRAIN_DATA_DIRECTORY + os.sep + f"NeuralData{dataset_index}.mat")["NeuralData"]
    test_kin = io.loadmat(config.TEST_DATA_DIRECTORY + os.sep + f"KinData{dataset_index}.mat")["KinData"]
    test_neural = io.loadmat(config.TEST_DATA_DIRECTORY + os.sep + f"NeuralData{dataset_index}.mat")["NeuralData"]
    return train_kin, train_neural, test_kin, test_neural


def random_load():
    num_dataset = len(os.listdir(config.TRAIN_DATA_DIRECTORY)) // 2
    return load_mat(random.randint(1, num_dataset))
