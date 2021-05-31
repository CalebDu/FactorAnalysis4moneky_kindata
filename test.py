from scipy import io
import matplotlib.pyplot as plt
import config
import os
import utils
import numpy as np


def main():
    train_kin, train_neural, test_kin, test_neural = utils.random_load()
    # plt.plot(np.array(range(0, 50)), train_neural[10][:50])

    print(train_neural[:10].dtype)
    # plt.plot(train_kin[0][:300], train_kin[1][:300])
    # plt.show()


if __name__ == '__main__':
    main()
    # mat = io.loadmat("2019070402_s96.mat")
    # plt.plot(range(1000), mat["DirectionNo"][0, :1000])
    # plt.show()
