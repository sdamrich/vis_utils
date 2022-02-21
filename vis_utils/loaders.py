import pandas as pd
import numpy as np
import os
import torchvision

# from https://github.com/stat-ml/ncvis
# use their get_pendigits.py and download_pendigits.sh to obtain the dataset
def load_pendigits(root_path):
    files = ["pendigits/optdigits.tes",
             "pendigits/optdigits.tra"]
    loaded = [None] * 2
    for f in files:
        df = pd.read_csv(os.path.join(root_path,f), header=None)
        for i in range(2):
            if i == 0:
                new = df.iloc[:, :-1].values
            else:
                new = df.iloc[:, -1].values
            if loaded[i] is None:
                loaded[i] = new
            else:
                loaded[i] = np.concatenate((loaded[i], new))
    return loaded


def load_mnist(root_path):
    root_path = os.path.join(root_path, "mnist")
    mnist_train = torchvision.datasets.MNIST(root=root_path, train=True,
                                             download=True, transform=None)
    x_train, y_train = mnist_train.data.float().numpy(), mnist_train.targets
    mnist_test = torchvision.datasets.MNIST(root=root_path, train=False,
                                            download=True, transform=None)
    x_test, y_test = mnist_test.data.float().numpy(), mnist_test.targets

    x = np.concatenate([x_train, x_test], axis=0)
    x = x.reshape(x.shape[0], -1)
    y = np.concatenate([y_train, y_test], axis=0)
    return x, y

