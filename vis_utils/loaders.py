import pandas as pd
import numpy as np
import os
import torchvision
import scipy.sparse
from .utils import kNN_graph
from sklearn.decomposition import PCA
import urllib.request
import zipfile
from .treutlein_preprocess import preprocess as treut_preprocess
from .zfish_preprocess import preprocess as zfish_preprocess

import urllib.request
import tarfile


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
    if not os.path.exists(root_path):
        os.mkdir(root_path)

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

def imbalance_dataset(x, y, props, seed=0):
    classes, counts = np.unique(y, return_counts= True)
    assert len(props) == len(np.unique(y))

    classes_idx = [ np.where(y == cls)[0] for cls in classes]

    new_counts = [int(prop * len(class_idx))
                        for prop, class_idx in zip(props, classes_idx) ]

    np.random.seed(seed)
    new_classes_idx = []
    for i in range(len(classes)):
        subsample_idx = np.random.permutation(counts[i])[:new_counts[i]]
        new_classes_idx.append(classes_idx[i][subsample_idx])
    full_subsample_idx = np.concatenate(new_classes_idx)

    return x[full_subsample_idx], y[full_subsample_idx]





def load_imba_mnist(root_path, props, seed=0):
    x, y = load_mnist(root_path)
    return imbalance_dataset(x, y, props, seed)



def load_cifar10(root_path):
    root_path = os.path.join(root_path, "cifar10")
    if not os.path.exists(root_path):
        os.mkdir(root_path)
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
    if not os.path.exists(root_path):
        os.mkdir(root_path)

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
        X, stage = treut_preprocess(metafile, countfile, line)

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


def load_zebrafish(root_path):
    root_path = os.path.join(root_path, "zebrafish")
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    try:
        x = np.load(os.path.join(root_path, "zfish.data.npy"))
        y = np.load(os.path.join(root_path, "zfish.labels.npy"))
    except FileNotFoundError:
        # download
        print("Downloading zebrafish data...")
        url = "https://kleintools.hms.harvard.edu/paper_websites/wagner_zebrafish_timecourse2018/WagnerScience2018.h5ad"
        file_name = "WagnerScience2018.h5ad"
        file_path = os.path.join(root_path, file_name)
        urllib.request.urlretrieve(url, file_path)

        print("Preprocessing zebrafish data...")
        # preprocess
        X, stage, alt_c = zfish_preprocess(file_path)
        np.save(os.path.join(root_path, "zfish.data.npy"), X)
        np.save(os.path.join(root_path, "zfish.labels.npy"), stage)
        np.save(os.path.join(root_path, "zfish.altlabels.npy"), alt_c)
        print("...done.")

        x = X
        y = stage
    return x, y

zebra_label_to_color = {
    "4hpf": "navy",
    "6hpf": "royalblue",
    "8hpf": "skyblue",
    "10hpf": "lightgreen",
    "14hpf": "gold",
    "18hpf": "tomato",
    "24hpf": "firebrick",
    "unused": "maroon",
}

zebra_color_to_label = {val: key for key, val in zebra_label_to_color.items()}

def load_c_elegans(root_path):
    data_dir = os.path.join(root_path, "c_elegans")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(os.path.join(data_dir,
                                       "packer_c-elegans",
                                       "c-elegans_qc_final.txt")):
        # download the C. elegans data
        url = "http://cb.csail.mit.edu/cb/densvis/datasets/packer_c-elegans_data.tar.gz"
        file_name = os.path.join(data_dir, "packer_c-elegans_data.tar.gz")

        urllib.request.urlretrieve(url, file_name)

        # extract the data
        tar = tarfile.open(file_name, "r:gz")
        tar.extractall(path=data_dir)
        tar.close()


    x = pd.read_csv(os.path.join(data_dir,
                                 "packer_c-elegans",
                                 "c-elegans_qc_final.txt"),
                         sep='\t',
                         header=None)
    x = np.array(x)
    meta = pd.read_csv(os.path.join(data_dir,
                                    "packer_c-elegans",
                                    "c-elegans_qc_final_metadata.txt"),
                       sep=',',
                       header=0)

    cell_types = meta["cell.type"].to_numpy().astype(str)

    y = np.zeros(len(cell_types)).astype(int)
    #name_to_label = {}
    for i, phase in enumerate(np.unique(cell_types)):
        #name_to_label[phase] = i
        y[cell_types == phase] = i
    return x, y

def load_k49(root_dataset):
    data_dir = os.path.join(root_dataset, "k49")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(os.path.join(data_dir, "k49-train-imgs.npz")):
        urls = ["http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz",
                "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz",
                "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz",
                "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz"]
        for url in urls:
            name = url.split("/")[-1]
            file_name = os.path.join(data_dir, name)
            urllib.request.urlretrieve(url, file_name)

    with np.load(os.path.join(data_dir, "k49-train-imgs.npz")) as data:
        x_train = data["arr_0"]
    with np.load(os.path.join(data_dir, "k49-test-imgs.npz")) as data:
        x_test = data["arr_0"]
    x = np.concatenate([x_train, x_test])
    x = x.reshape(len(x), -1)

    with np.load(os.path.join(data_dir, "k49-train-labels.npz")) as data:
        y_train = data["arr_0"]

    with np.load(os.path.join(data_dir, "k49-test-labels.npz")) as data:
        y_test = data["arr_0"]
    y = np.concatenate([y_train, y_test])

    return x, y



# complete loader:
def load_dataset(root_path, dataset, k=15):
    # load dataset
    if not os.path.exists(os.path.join(root_path, dataset)):
        os.mkdir(os.path.join(root_path, dataset))
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

    elif dataset.startswith("imba_mnist"):
        l = dataset.split("_")
        ind_seed = l.index("seed")
        seed = int(l[ind_seed + 1])
        mode = l[2]
        if mode == "lin":
            x, y = load_imba_mnist(root_path,
                                   props = 1.0 - np.arange(10) / 10,
                                   seed=seed)
        elif mode == "odd":
            x, y = load_imba_mnist(root_path,
                                   props = 5*[1.0, 0.1],
                                   seed = seed)
        else:
            raise NotImplementedError(f"Subsampling mode {mode} is not implemented.")
    elif dataset == "cifar10":
        x, y = load_cifar10(root_path)
    elif dataset == "human-409b2":
        x, y = load_human(root_path)
    elif dataset == "zebrafish":
        x, y = load_zebrafish(root_path)
    elif dataset == "c_elegans":
        x, y = load_c_elegans(root_path)
    elif dataset == "k49":
        x, y = load_k49(root_path)
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
        if dataset.startswith("mnist") or dataset == "k49" or dataset == "cifar10":
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





