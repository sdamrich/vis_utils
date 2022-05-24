import pandas as pd
import numpy as np
import os
import torchvision
import scipy.sparse
from .utils import kNN_graph
from sklearn.decomposition import PCA
import urllib.request
import zipfile
from .treutlein_preprocess import preprocess

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


def load_small_mnist(root_path, seed=0, size=6000):
    root_path = os.path.join(root_path, f"mnist_seed_{seed}_size_{size}")

    try:
        x_small = np.load(os.path.join(root_path, "data.npy"))
        y_small = np.load(os.path.join(root_path, "labels.npy"))
    except FileNotFoundError:
        mnist_train = torchvision.datasets.MNIST(root=root_path, train=True,
                                                 download=True, transform=None)
        x_train, y_train = mnist_train.data.float().numpy(), mnist_train.targets
        mnist_test = torchvision.datasets.MNIST(root=root_path, train=False,
                                                download=True, transform=None)
        x_test, y_test = mnist_test.data.float().numpy(), mnist_test.targets

        x = np.concatenate([x_train, x_test], axis=0)
        x = x.reshape(x.shape[0], -1)
        y = np.concatenate([y_train, y_test], axis=0)

        ind = []
        np.random.seed(seed)
        for i in range(10):
            ind_by_class = np.random.choice(np.argwhere(y == i).flatten(),
                                            size=int(size/ 10),
                                            replace=False)
            ind.extend(ind_by_class)
        perm = np.random.permutation(10 * int(size/10))
        ind = (np.array(ind)[perm],)
        x_small = x[ind, :][0]
        y_small = y[ind]

        np.save(os.path.join(root_path, "data.npy"),
                x_small)
        np.save(os.path.join(root_path, "labels.npy"),
                y_small)
    return x_small, y_small


def load_cifar10(root_path):
    root_path = os.path.join(root_path, "cifar10")
    cifar10_train = torchvision.datasets.CIFAR10(root=root_path, train=True,
                                             download=True, transform=None)

    x_train, y_train = cifar10_train.data, cifar10_train.targets
    cifar10_test = torchvision.datasets.CIFAR10(root=root_path, train=False,
                                            download=True, transform=None)
    x_test, y_test = cifar10_test.data, cifar10_test.targets

    x = np.concatenate([x_train, x_test], axis=0)
    x = x.reshape(x.shape[0], -1)
    y = np.concatenate([y_train, y_test], axis=0)
    return x, y

def load_human(root_path):
    root_path = os.path.join(root_path, "human-409b2")
    try:
        x = np.load(os.path.join(root_path, "human-409b2.data.npy"))
        y = np.load(os.path.join(root_path, "human-409b2.labels.npy"))
    except FileNotFoundError:
        urls = ["https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.1.zip",
                "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.2.zip",
                "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.3.zip",
                "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.4.zip",
                "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.5.zip",
                "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.6.zip",
                "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.7.zip",]
        print("Downloading data")
        for url in urls:
            filename =  os.path.join(root_path, url.split("/")[-1])
            urllib.request.urlretrieve(url, filename)
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(os.path.join(root_path, "unzipped_files"))

        print("Preprocessing data")
        metafile = os.path.join(root_path,
                                "unzipped_files",
                                "metadata_human_cells.tsv")
        countfile = os.path.join(root_path,
                                 "unzipped_files",
                                 "human_cell_counts_consensus.mtx")
        line = "409b2"
        X, stage = preprocess(metafile, countfile, line)

        outputfile = "human-409b2"

        np.save(os.path.join(root_path, outputfile + ".data.npy"), X)
        np.save(os.path.join(root_path, outputfile + ".labels.npy"), stage)
        x = X
        y = stage
        print("Done")
    return x, y

# for translating labels to colors and time points
label_to_color = {
            "iPSCs": "navy",
            "EB": "royalblue",
            "Neuroectoderm": "skyblue",
            "Neuroepithelium": "lightgreen",
            "Organoid-1M": "gold",
            "Organoid-2M": "tomato",
            "Organoid-3M": "firebrick",
            "Organoid-4M": "maroon",
        }

time_to_color = {
        "  0 days": "navy",
        "  4 days": "royalblue",
        "10 days": "skyblue",
        "15 days": "lightgreen",
        "  1 month": "gold",
        "  2 months": "tomato",
        "  3 months": "firebrick",
        "  4 months": "maroon",
    }

color_to_time = {v: k for k,v in time_to_color.items()}
label_to_time = {k: color_to_time[label_to_color[k]] for k in label_to_color.keys()}

# complete loader:
def load_dataset(root_path, dataset, k=15):
    # load dataset
    if dataset == "pendigits":
        x, y = load_pendigits(root_path)

    elif dataset == "mnist":
        x, y = load_mnist(root_path)

    elif dataset.startswith("mnist_"):
        l = dataset.split("_")
        ind_seed = l.index("seed")
        seed = int(l[ind_seed + 1])
        ind_size = l.index("size")
        size = int(l[ind_size + 1])
        x, y = load_small_mnist(root_path, seed, size)
    elif dataset == "cifar10":
        x, y = load_cifar10(root_path)
    elif dataset == "human-409b2":
        x, y = load_human(root_path)
    else:
        raise NotImplementedError

    # get pca
    # load / compute and save 2D PCA for initialisation
    try:
        pca2 = np.load(os.path.join(root_path, dataset, "pca2.npy"))
    except FileNotFoundError:
        pca_projector = PCA(n_components=2)
        pca2 = pca_projector.fit_transform(np.array(x))
        np.save(os.path.join(root_path, dataset, "pca2.npy"), pca2)

    # get skkn graph
    knn_file_name = os.path.join(root_path,
                                 dataset,
                                 f"sknn_graph_k_{k}_metric_euclidean.npz")

    try:
        sknn_graph = scipy.sparse.load_npz(knn_file_name)
    except IOError:
        x_for_knn = x
        if dataset.startswith("mnist"):
            try:
                pca50 = np.load(os.path.join(root_path, dataset, "pca50.npy"))
            except FileNotFoundError:
                pca_50_projector = PCA(n_components=50, random_state=0)
                pca50 = pca_50_projector.fit_transform(x)
                np.save(os.path.join(root_path, dataset, "pca50.npy"), pca50)
            x_for_knn = pca50

        knn_graph = kNN_graph(x_for_knn.astype("float"),
                              k,
                              metric="euclidean").cpu().numpy().flatten()
        knn_graph = scipy.sparse.coo_matrix((np.ones(len(x) * k),
                                             (np.repeat(np.arange(x.shape[0]), k),
                                              knn_graph)),
                                            shape=(len(x), len(x)))
        sknn_graph = knn_graph.maximum(knn_graph.transpose()).tocoo()
        scipy.sparse.save_npz(knn_file_name, sknn_graph)

    return x, y, sknn_graph, pca2





