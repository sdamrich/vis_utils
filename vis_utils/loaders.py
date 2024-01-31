import pandas as pd
import numpy as np
import os
import torchvision
import scipy.sparse
from .utils import kNN_graph, save_dict, load_dict
from sklearn.decomposition import PCA
import zipfile
from .treutlein_preprocess import preprocess as treut_preprocess
from .zfish_preprocess import preprocess as zfish_preprocess
from .rnaseqTools import geneSelection, sparseload
import urllib.request
import requests
import tarfile
import h5py
import scanpy as sc
import subprocess



def categorical2numeric(y, return_unique=False):
    y_unique = np.unique(y)
    y_numeric = np.zeros_like(y)
    for i, y_u in enumerate(y_unique):
        y_numeric[y==y_u] = i
    if return_unique:
        return y_numeric, y_unique
    else:
        return y_numeric


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
        #urls = ["https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.1.zip",
        #        "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.2.zip",
        #        "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.3.zip",
        #        "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.4.zip",
        #        "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.5.zip",
        #        "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.6.zip",
        #        "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.7.zip",]
        metafile = os.path.join(root_path,
                                #"unzipped_files",
                                "metadata_human_cells.tsv")
        countfile = os.path.join(root_path,
                                 #"unzipped_files",
                                 "human_cell_counts_consensus.mtx")
        urls = []
        if not os.path.exists(metafile):
            urls.append("http://ftp.ebi.ac.uk/biostudies/nfs/E-MTAB-/552/E-MTAB-7552/Files/metadata_human_cells.tsv")
        if not os.path.exists(countfile):
            urls.append("http://ftp.ebi.ac.uk/biostudies/nfs/E-MTAB-/552/E-MTAB-7552/Files/human_cell_counts_consensus.mtx")

        if len(urls) > 0:
            print("Downloading data")
        for url in urls:
            filename = os.path.join(root_path, url.split("/")[-1])
            #urllib.request.urlretrieve(url, filename)
            download_file(url, filename)
        #    print(filename)
        #    #with zipfile.ZipFile(filename, "r") as zip_ref:
        #    #    zip_ref.extractall(os.path.join(root_path, "unzipped_files"))
        #    with tarfile.open(filename, "r:gz") as f:
        #        f.extractall()
        #    assert False

        print("Preprocessing data")
        line = "409b2"
        X, stage = treut_preprocess(metafile, countfile, line)

        outputfile = "human-409b2"

        np.save(os.path.join(root_path, outputfile + ".data.npy"), X)
        np.save(os.path.join(root_path, outputfile + ".labels.npy"), stage)
        x = X
        y = stage
        print("Done")

    d = {"label_colors": {
        "iPSCs": "navy",
        "EB": "royalblue",
        "Neuroectoderm": "skyblue",
        "Neuroepithelium": "lightgreen",
        "Organoid-1M": "gold",
        "Organoid-2M": "tomato",
        "Organoid-3M": "firebrick",
        "Organoid-4M": "maroon",
    }, "time_colors": {
        "  0 days": "navy",
        "  4 days": "royalblue",
        "10 days": "skyblue",
        "15 days": "lightgreen",
        "  1 month": "gold",
        "  2 months": "tomato",
        "  3 months": "firebrick",
        "  4 months": "maroon",
    }}

    return x, y, d



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

def load_tasic(root_dataset):
    data_dir = os.path.join(root_dataset, "tasic")

    try:
        x = np.load(os.path.join(data_dir, "pca50.npy"))
        y = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "tasic2018.pkl"))
    except FileNotFoundError:
        urls = ["http://celltypes.brain-map.org/api/v2/well_known_file_download/694413985",
                "http://celltypes.brain-map.org/api/v2/well_known_file_download/694413179",
                "https://raw.githubusercontent.com/berenslab/mini-atlas/master/data/raw/allen/tasic2018/sample_heatmap_plot_data.csv"
                ]

        file_names = ["mouse_VISp_gene_expression_matrices_2018-06-14.zip",
                      "mouse_ALM_gene_expression_matrices_2018-06-14.zip",
                      "sample_heatmap_plot_data.csv"
                      ]

        for file_name, url in zip(file_names, urls):
            file_name = os.path.join(data_dir, file_name)
            if not os.path.exists(file_name):
                urllib.request.urlretrieve(url, file_name)

            if file_name.endswith(".zip"):
                with zipfile.ZipFile(file_name, "r") as zip_ref:
                    zip_ref.extractall(os.path.join(data_dir, file_name.split("/")[-1].strip(".zip")))

        # from https://github.com/berenslab/rna-seq-tsne/blob/master/tasic-et-al.ipynb
        file_name_VISp = os.path.join(data_dir, "mouse_VISp_gene_expression_matrices_2018-06-14", "mouse_VISp_2018-06-14_exon-matrix.csv")
        counts1, genes1, cells1 = sparseload(file_name_VISp)

        file_name_ALM = os.path.join(data_dir, "mouse_ALM_gene_expression_matrices_2018-06-14", "mouse_ALM_2018-06-14_exon-matrix.csv")
        counts2, genes2, cells2 = sparseload(file_name_ALM)

        counts = scipy.sparse.vstack((counts1, counts2), format='csc')

        cells = np.concatenate((cells1, cells2))

        if np.all(genes1==genes2):
            genes = np.copy(genes1)

        genesDF = pd.read_csv(os.path.join(data_dir, "mouse_VISp_gene_expression_matrices_2018-06-14",
                                           "mouse_VISp_2018-06-14_genes-rows.csv"))
        ids = genesDF['gene_entrez_id'].tolist()
        symbols = genesDF['gene_symbol'].tolist()
        id2symbol = dict(zip(ids, symbols))
        genes = np.array([id2symbol[g] for g in genes])

        clusterInfo = pd.read_csv(os.path.join(data_dir, "sample_heatmap_plot_data.csv"))
        goodCells = clusterInfo['sample_name'].values
        ids = clusterInfo['cluster_id'].values
        labels = clusterInfo['cluster_label'].values
        colors = clusterInfo['cluster_color'].values

        clusterNames = np.array([labels[ids == i + 1][0] for i in range(np.max(ids))])
        clusterColors = np.array([colors[ids == i + 1][0] for i in range(np.max(ids))])
        clusters = np.copy(ids)

        ind = np.array([np.where(cells == c)[0][0] for c in goodCells])
        counts = counts[ind, :]

        areas = (ind < cells1.size).astype(int)

        clusters = clusters - 1

        tasic2018 = {'counts': counts, 'genes': genes, 'clusters': clusters, 'areas': areas,
                     'clusterColors': clusterColors, 'clusterNames': clusterNames}

        save_dict(tasic2018, os.path.join(data_dir, "tasic2018.pkl"))

        markerGenes = ['Snap25', 'Gad1', 'Slc17a7', 'Pvalb', 'Sst', 'Vip', 'Aqp4',
                       'Mog', 'Itgam', 'Pdgfra', 'Flt1', 'Bgn', 'Rorb', 'Foxp2']

        importantGenesTasic2018 = geneSelection(
            tasic2018['counts'], n=3000, threshold=32,
            markers=markerGenes, genes=tasic2018['genes'], plot=False)

        librarySizes = np.sum(tasic2018['counts'], axis=1)
        X = np.log2(tasic2018['counts'][:, importantGenesTasic2018] / librarySizes * 1e+6 + 1)
        X = np.array(X)
        X = X - X.mean(axis=0)
        U, s, V = np.linalg.svd(X, full_matrices=False)
        U[:, np.sum(V, axis=1) < 0] *= -1
        X = np.dot(U, np.diag(s))
        X = X[:, np.argsort(s)[::-1]][:, :50]

        x = X
        y = tasic2018["clusters"]
        d = tasic2018

        np.save(os.path.join(data_dir, "pca50.npy"), x)
        np.save(os.path.join(data_dir, "labels.npy"), y)

    return x, y, d


def load_tasic3000(root_dataset):
    data_dir = os.path.join(root_dataset, "tasic")

    try:
        x = np.load(os.path.join(data_dir, "log_imp_genes.npy"))
        y = np.load(os.path.join(data_dir, "labels.npy"))
        d = load_dict(os.path.join(data_dir, "tasic2018.pkl"))
    except FileNotFoundError:
        urls = ["http://celltypes.brain-map.org/api/v2/well_known_file_download/694413985",
                "http://celltypes.brain-map.org/api/v2/well_known_file_download/694413179",
                "https://raw.githubusercontent.com/berenslab/mini-atlas/master/data/raw/allen/tasic2018/sample_heatmap_plot_data.csv"
                ]

        file_names = ["mouse_VISp_gene_expression_matrices_2018-06-14.zip",
                      "mouse_ALM_gene_expression_matrices_2018-06-14.zip",
                      "sample_heatmap_plot_data.csv"
                      ]

        for file_name, url in zip(file_names, urls):
            file_name = os.path.join(data_dir, file_name)
            if not os.path.exists(file_name):
                urllib.request.urlretrieve(url, file_name)

            if file_name.endswith(".zip"):
                with zipfile.ZipFile(file_name, "r") as zip_ref:
                    zip_ref.extractall(os.path.join(data_dir, file_name.split("/")[-1].strip(".zip")))

        # from https://github.com/berenslab/rna-seq-tsne/blob/master/tasic-et-al.ipynb
        file_name_VISp = os.path.join(data_dir, "mouse_VISp_gene_expression_matrices_2018-06-14", "mouse_VISp_2018-06-14_exon-matrix.csv")
        counts1, genes1, cells1 = sparseload(file_name_VISp)

        file_name_ALM = os.path.join(data_dir, "mouse_ALM_gene_expression_matrices_2018-06-14", "mouse_ALM_2018-06-14_exon-matrix.csv")
        counts2, genes2, cells2 = sparseload(file_name_ALM)

        counts = scipy.sparse.vstack((counts1, counts2), format='csc')

        cells = np.concatenate((cells1, cells2))

        if np.all(genes1==genes2):
            genes = np.copy(genes1)

        genesDF = pd.read_csv(os.path.join(data_dir, "mouse_VISp_gene_expression_matrices_2018-06-14",
                                           "mouse_VISp_2018-06-14_genes-rows.csv"))
        ids = genesDF['gene_entrez_id'].tolist()
        symbols = genesDF['gene_symbol'].tolist()
        id2symbol = dict(zip(ids, symbols))
        genes = np.array([id2symbol[g] for g in genes])

        clusterInfo = pd.read_csv(os.path.join(data_dir, "sample_heatmap_plot_data.csv"))
        goodCells = clusterInfo['sample_name'].values
        ids = clusterInfo['cluster_id'].values
        labels = clusterInfo['cluster_label'].values
        colors = clusterInfo['cluster_color'].values

        clusterNames = np.array([labels[ids == i + 1][0] for i in range(np.max(ids))])
        clusterColors = np.array([colors[ids == i + 1][0] for i in range(np.max(ids))])
        clusters = np.copy(ids)

        ind = np.array([np.where(cells == c)[0][0] for c in goodCells])
        counts = counts[ind, :]

        areas = (ind < cells1.size).astype(int)

        clusters = clusters - 1

        tasic2018 = {'counts': counts, 'genes': genes, 'clusters': clusters, 'areas': areas,
                     'clusterColors': clusterColors, 'clusterNames': clusterNames}

        save_dict(tasic2018, os.path.join(data_dir, "tasic2018.pkl"))

        markerGenes = ['Snap25', 'Gad1', 'Slc17a7', 'Pvalb', 'Sst', 'Vip', 'Aqp4',
                       'Mog', 'Itgam', 'Pdgfra', 'Flt1', 'Bgn', 'Rorb', 'Foxp2']

        importantGenesTasic2018 = geneSelection(
            tasic2018['counts'], n=3000, threshold=32,
            markers=markerGenes, genes=tasic2018['genes'], plot=False)

        librarySizes = np.sum(tasic2018['counts'], axis=1)
        X = np.log2(tasic2018['counts'][:, importantGenesTasic2018] / librarySizes * 1e+6 + 1)
        X = np.array(X)

        y = tasic2018["clusters"]
        d = tasic2018

        np.save(os.path.join(data_dir, "log_imp_genes.npy"), X)
        np.save(os.path.join(data_dir, "labels.npy"), y)

    return x, y, d


def load_mca_ss2(root_path):
    data_dir = os.path.join(root_path, "mca_ss2")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    try:
        x = np.load(os.path.join(data_dir, "mca_ss2.data.npy"))
        y = np.load(os.path.join(data_dir, "mca_ss2.labels.npy"))
        d = load_dict(os.path.join(data_dir, "meta_mca_ss2.pkl"))
    except FileNotFoundError:
        print("Downloading MCA Smartseq2 data...", end="", flush=True)
        urls = ["https://github.com/vhowick/MalariaCellAtlas/raw/v1.0/Expression_Matrices/Smartseq2/SS2_tmmlogcounts.csv.zip",
                "https://raw.githubusercontent.com/vhowick/MalariaCellAtlas/v1.0/Expression_Matrices/Smartseq2/SS2_pheno.csv"]
        for url in urls:
            file_name =  os.path.join(data_dir, url.split("/")[-1])
            if not os.path.exists(file_name):
                urllib.request.urlretrieve(url, file_name)
            if file_name.endswith(".zip"):
                with zipfile.ZipFile(file_name, "r") as zip_ref:
                    zip_ref.extractall(data_dir)

        file_name_pp = os.path.join(data_dir, "SS2_tmmlogcounts.csv")
        x = pd.read_csv(file_name_pp, encoding='latin-1', header=0, index_col=0).to_numpy().T

        file_name_pheno = os.path.join(data_dir, "SS2_pheno.csv")
        clusters = pd.read_csv(file_name_pheno, encoding='latin-1')["ShortenedLifeStage2"].to_numpy()

        # numeric labels
        cluster_names  = np.unique(clusters)
        cluster_names_to_labels = {cluster_names[i]: i for i in range(len(cluster_names))}
        y = np.array([cluster_names_to_labels[clusters[i]] for i in range(len(clusters))])

        # dicts for plotting
        cluster_names_to_print_names = {
            "bbSpz" :"Injected sporozoite",
            "EEF":"Liver stage",
            "Merozoite":"Merizoite",
            "oocyst":"Oocyst",
            "ook" :"Bolus okinete",
            "ooSpz" :"????",
            "Ring":"Ring",
            "sgSpz":"Gland sporozoite",
            "Schizont":"Schizont",
            "Male":"Male gametocyte",
            "Female":"Female gametocyte",
            "ookoo" :"Okinete/oocyst",
            "Trophozoite":"Trophozoite"
        }

        cluster_names_to_colors = {
            "bbSpz" : "#000080",
            "EEF":"#ff8c00",
            "Merozoite":"#ffb6c1",
            "oocyst":"#4682b4",
            "ook" : "#00868b",
            "ooSpz" :"#87cefa",
            "Ring":"#ff69b4",
            "sgSpz":"#4169e1",
            "Schizont":"#d02090",
            "Male":"#a020f0",
            "Female":"#551a8b",
            "ookoo":"#48d1cc",
            "Trophozoite":"#ee82ee"

        }

        d = {"cluster_names": {i: cluster_names[i] for i in range(len(cluster_names))},
             "cluster_print_names": {i: cluster_names_to_print_names[cluster_names[i]] for i in range(len(cluster_names))},
             "cluster_colors": {i: cluster_names_to_colors[cluster_names[i]] for i in range(len(cluster_names))}}

        outputfile = "mca_ss2"
        np.save(os.path.join(data_dir, outputfile + ".data.npy"), x)
        np.save(os.path.join(data_dir, outputfile + ".labels.npy"), y)
        save_dict(d, os.path.join(data_dir, "meta_mca_ss2.pkl"))

        print("done")
    return x, y, d


def load_mca_ss2_idc(root_path):
    x, y, d = load_mca_ss2(root_path)

    # labels in IDC
    labels_idc = ["Merozoite", "Ring", "Trophozoite", "Schizont"]

    # filter data
    idx = np.array([d["cluster_names"][cluster_id] in labels_idc for cluster_id in y])

    x = x[idx]
    y = y[idx]
    for i in np.unique(y):
        if d["cluster_names"][i] not in labels_idc:
            del d["cluster_names"][i]
            del d["cluster_print_names"][i]
            del d["cluster_colors"][i]
    return x, y, d


dataset2url_cc = {"neurosphere": "https://zenodo.org/record/5519841/files/neurosphere.qs",
               "hippocampus": "https://zenodo.org/record/5519841/files/hipp.qs",
               "HeLa2": "https://zenodo.org/record/5519841/files/HeLa2.qs",
               "pancreas": "https://zenodo.org/record/5519841/files/endo.qs",
               "pallium": "https://storage.googleapis.com/linnarsson-lab-tmp/Cortex_EMX1_louvain3_passedQC_PostM_rev1.h5ad"}


def download_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully from {url}.")
    else:
        print(f"Failed to download file from {url}. Status code: {response.status_code}")


def download_cc_file(dataset, root_path):
    data_dir = os.path.join(root_path, dataset)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    download_file(dataset2url_cc[dataset], os.path.join(data_dir, f"{dataset}.qs"))


def load_cc_dataset(root_path, dataset, representation="tricycleEmbedding"):
    data_dir = os.path.join(root_path, dataset)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    try:
        with h5py.File(os.path.join(data_dir, dataset+".h5"), "r") as f:
            assert representation in f.keys(), f"representation {representation} not found, only {f.keys()} available"
            x = f[representation][:].T
            stages = np.array([s.decode("utf-8") for s in f["CCStage"][:]])
            if "cell_type" in f.keys():
                cell_types = np.array([s.decode("utf-8") for s in f["cell_type"][:]])
                cell_types_exist = True
            else:
                cell_types_exist = False
            theta = f["tricyclePosition"][:]
    except:
        # download dataset
        qs_file = os.path.join(root_path, dataset, f"{dataset}.qs")
        if not os.path.exists(qs_file):
            download_cc_file(dataset=dataset, root_path=root_path)

        # pancreas dataset does not have the tricycle and GOPCA embedding precomputed, so we need to compute it ourselves
        if dataset == "pancreas":
            # clone required git repos
            tricycle_path = f"{os.path.dirname(os.path.realpath(__file__))}/tricycle"
            if not os.path.exists(tricycle_path):
                subprocess.run(["git", "clone", "https://github.com/hansenlab/tricycle.git", tricycle_path])
            tricycle_fig_path = f"{os.path.dirname(os.path.realpath(__file__))}/tricycle_paper_figs"
            if not os.path.exists(tricycle_fig_path):
                subprocess.run(["git", "clone", "https://github.com/hansenlab/tricycle_paper_figs.git", tricycle_fig_path])

            # compute the GOPCA and tricycle embedding representations
            subprocess.run(["conda", "run", "-n", "oneR",
                            "Rscript", f"{os.path.dirname(os.path.realpath(__file__))}/pancreas.R", str(root_path), str(os.path.dirname(os.path.realpath(__file__)))])

        # copy data from .qs to .h5
        # note that this does not work if this function is called from a jupyter notebook or a tmux environment
        subprocess.run(["conda", "run", "-n", "oneR",
                        "Rscript", f"{os.path.dirname(os.path.realpath(__file__))}/cc_dataset.R", str(root_path), dataset])

        # now that the data is available in .h5, load it
        with h5py.File(os.path.join(data_dir, dataset+".h5"), "r") as f:
            assert representation in f.keys(), f"representation {representation} not found, only {f.keys()} available"
            x = f[representation][:].T
            stages = np.array([s.decode("utf-8") for s in f["CCStage"][:]])
            if "cell_type" in f.keys():
                cell_types = np.array([s.decode("utf-8") for s in f["cell_type"][:]])
                cell_types_exist = True
            else:
                cell_types_exist = False
            theta = f["tricyclePosition"][:]
        #return load_cc_dataset(root_path=root_path, dataset=dataset, representation=representation)

    cc_colors = {"G1.S": '#B2627C', "S": '#F29360', "G2": '#FCEA64', "G2.M": '#86BBD8', "M.G1": '#8159ba', "NA": "gray"}

    unique_stages = np.unique(stages)

    stage_to_y = {unique_stages[i]: i for i in range(len(unique_stages))}
    y = np.array([stage_to_y[stage] for stage in stages])
    y_to_stage = {v: k for k, v in stage_to_y.items()}

    reordered_stages = np.array([y_to_stage[i] for i in np.unique(y)])
    reordered_colors = np.array([cc_colors[y_to_stage[i]] for i in np.unique(y)])

    d = {"stage_names": reordered_stages,
         "colors": reordered_colors,
         "theta": theta,}
    if cell_types_exist:
        d["cell_types"] = cell_types

    return x, y, d


def load_small_cc_dataset(root_path, dataset, representation="tri", seed=0):
    x, y, d = load_cc_dataset(root_path, dataset, representation)
    np.random.seed(seed)
    idx = np.random.choice(len(x), replace=False, size=1000)
    d["theta"] = d["theta"][idx]
    d["cell_types"] = d["cell_types"][idx]
    return x[idx], y[idx], d


def load_pallium_scVI(root_path):
    try:
        with h5py.File(os.path.join(root_path, "pallium_scVI", "pallium_scVI.h5"), "r") as f:
            x = f["scVI"][:]
            y = f["CellClass"][:]
            d = {
                "Subset": f["Subset"][:],
                "CellCycle": f["CellCycle"][:],
                "UMAP": f["UMAP"][:],
                "Cycling": f["Cycling"][:]
                 }
    except FileNotFoundError:
        adata = sc.read(os.path.join(root_path, "pallium", "pallium.h5ad"))

        if not os.path.exists(os.path.join(root_path, "pallium_scVI")):
            os.mkdir(os.path.join(root_path, "pallium_scVI"))
        with h5py.File(os.path.join(root_path, "pallium_scVI", "pallium_scVI.h5"), "w") as file:
            file.create_dataset("scVI", data=np.array(adata.obsm["X_scVI"]))
            file.create_dataset("Subset", data=np.array(adata.obs["Subset"]))
            file.create_dataset("CellClass", data=np.array(adata.obs["CellClass"]))
            file.create_dataset("CellCycle", data=np.array(adata.obs["CellCycle"]))
            file.create_dataset("Cycling", data=np.array(adata.obs["Cycling"]))
            file.create_dataset("UMAP", data=np.array(adata.obsm["X_umap"]))
        x, y, d = load_pallium_scVI(root_path)

    return x, y, d


def load_pallium_scVI_10pcw(root_path):
    try:
        with h5py.File(os.path.join(root_path, "pallium_scVI_10pcw", "pallium_scVI_10pcw.h5"), "r") as f:
            x = f["scVI"][:]
            y = f["CellClass"][:]
            d = {
                "Subset": f["Subset"][:],
                "CellCycle": f["CellCycle"][:],
                "UMAP": f["UMAP"][:],
                "Cycling": f["Cycling"][:]
                 }
    except FileNotFoundError:
        x, y, d = load_pallium_scVI(root_path)

        mask = d["Subset"] == b"10wk"
        new_dataset = "pallium_scVI_10pcw"

        if not os.path.exists(os.path.join(root_path, new_dataset)):
            os.mkdir(os.path.join(root_path, new_dataset))
        with h5py.File(os.path.join(root_path, new_dataset, new_dataset + ".h5"), "w") as file:
            file.create_dataset("scVI", data=x[mask])
            file.create_dataset("CellClass", data=y[mask])
            file.create_dataset("Subset", data=d["Subset"][mask])
            file.create_dataset("CellCycle", data=d["CellCycle"][mask])
            file.create_dataset("Cycling", data=d["Cycling"][mask])
            file.create_dataset("UMAP", data=d["UMAP"][mask])

        # reload, now the try block should work out
        x, y, d = load_pallium_scVI_10pcw(root_path)
    return x, y, d


def load_pallium_scVI_10pcw_cycling(root_path):
    try:
        with h5py.File(os.path.join(root_path, "pallium_scVI_10pcw_cycling", "pallium_scVI_10pcw_cycling.h5"), "r") as f:
            x = f["scVI"][:]
            y = f["CellClass"][:]
            d = {
                "Subset": f["Subset"][:],
                "CellCycle": f["CellCycle"][:],
                "UMAP": f["UMAP"][:],
                "Cycling": f["Cycling"][:]
                 }
    except FileNotFoundError:

        x, y, d = load_pallium_scVI_10pcw(root_path)

        mask = d["Cycling"]

        new_dataset = "pallium_scVI_10pcw_cycling"
        if not os.path.exists(os.path.join(root_path, new_dataset)):
            os.mkdir(os.path.join(root_path, new_dataset))
        with h5py.File(os.path.join(root_path, new_dataset, new_dataset + ".h5"), "w") as file:
            file.create_dataset("scVI", data=x[mask])
            file.create_dataset("CellClass", data=y[mask])
            file.create_dataset("Subset", data=d["Subset"][mask])
            file.create_dataset("CellCycle", data=d["CellCycle"][mask])
            file.create_dataset("Cycling", data=d["Cycling"][mask])
            file.create_dataset("UMAP", data=d["UMAP"][mask])

        # reload, now the try block should work out
        x, y, d = load_pallium_scVI_10pcw_cycling(root_path)

    return x, y, d


def load_pallium_scVI_IPC(root_path):
    try:
        with h5py.File(os.path.join(root_path, "pallium_scVI_IPC", "pallium_scVI_IPC.h5"), "r") as f:
            x = f["scVI"][:]
            y = f["CellClass"][:]
            d = {
                "Subset": f["Subset"][:],
                "CellCycle": f["CellCycle"][:],
                "UMAP": f["UMAP"][:],
                "Cycling": f["Cycling"][:]
                 }
    except FileNotFoundError:
        x, y, d = load_pallium_scVI(root_path)

        mask = y == b'Neuronal IPC'
        new_dataset = "pallium_scVI_IPC"
        if not os.path.exists(os.path.join(root_path, new_dataset)):
            os.mkdir(os.path.join(root_path, new_dataset))
        with h5py.File(os.path.join(root_path, new_dataset, new_dataset + ".h5"), "w") as file:
            file.create_dataset("scVI", data=x[mask])
            file.create_dataset("Subset", data=d["Subset"][mask])
            file.create_dataset("CellClass", data=y[mask])
            file.create_dataset("CellCycle", data=d["CellCycle"][mask])
            file.create_dataset("Cycling", data=d["Cycling"][mask])
            file.create_dataset("UMAP", data=d["UMAP"][mask])
        x, y, d = load_pallium_scVI_IPC(root_path)

    return x, y, d



def load_subsampled_pallium(root_path, dataset, seed=0, n=1000):
    x, y, _, _, d = load_dataset(root_path, dataset)
    np.random.seed(seed)
    mask = np.random.choice(x.shape[0], n, replace=False)
    return x[mask], y[mask], {key: value[mask] for key, value in d.items()}


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


def load_toy(root_path):
    data_dir = os.path.join(root_path, "toy")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([0.5, np.sqrt(3)/2])

    x= np.stack([p1, p2, p3])
    y = np.arange(3, dtype=int)

    return x, y


# complete loader:
def load_dataset(root_path, dataset, k=15, seed=None):
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
        x, y, d = load_human(root_path)
    elif dataset == "zebrafish":
        x, y = load_zebrafish(root_path)
    elif dataset == "c_elegans":
        x, y = load_c_elegans(root_path)
    elif dataset == "k49":
        x, y = load_k49(root_path)
    elif dataset == "toy":
        x, y = load_toy(root_path)
    elif dataset == "tasic":
        x, y, d = load_tasic(root_path)
    elif dataset == "tasic3000":
        x, y, d = load_tasic3000(root_path)
    elif dataset == "mca_ss2":
        x, y,d = load_mca_ss2(root_path)
    elif dataset == "mca_ss2_idc":
        x, y, d = load_mca_ss2_idc(root_path)
    elif dataset == "neurosphere_pca":
        x, y, d = load_cc_dataset(root_path, "neurosphere", representation="PCA30D")
    elif dataset == "neurosphere_gopca":
        x, y, d = load_cc_dataset(root_path, "neurosphere", representation="GOPCA20D")
    elif dataset == "neurosphere_tricycle":
        x, y, d = load_cc_dataset(root_path, "neurosphere", representation="tricycleEmbedding")
    elif dataset == "neurosphere_pca_small":
        x, y, d = load_small_cc_dataset(root_path, "neurosphere", representation="PCA30D", seed=seed)
    elif dataset == "neurosphere_gopca_small":
        x, y, d = load_small_cc_dataset(root_path, "neurosphere", representation="GOPCA20D", seed=seed)
    elif dataset == "neurosphere_tricycle_small":
        x, y, d = load_small_cc_dataset(root_path, "neurosphere", representation="tricycleEmbedding", seed=seed)
    elif dataset == "hippocampus_pca":
        x, y, d = load_cc_dataset(root_path, "hippocampus", representation="PCA30D")
    elif dataset == "hippocampus_gopca":
        x, y, d = load_cc_dataset(root_path, "hippocampus", representation="GOPCA20D")
    elif dataset == "hippocampus_tricycle":
        x, y, d = load_cc_dataset(root_path, "hippocampus", representation="tricycleEmbedding")
    elif dataset == "hippocampus_pca_small":
        x, y, d = load_small_cc_dataset(root_path, "hippocampus", representation="PCA30D", seed=seed)
    elif dataset == "hippocampus_gopca_small":
        x, y, d = load_small_cc_dataset(root_path, "hippocampus", representation="GOPCA20D", seed=seed)
    elif dataset == "hippocampus_tricycle_small":
        x, y, d = load_small_cc_dataset(root_path, "hippocampus", representation="tricycleEmbedding", seed=seed)
    elif dataset == "HeLa2_pca":
        x, y, d = load_cc_dataset(root_path, "HeLa2", representation="PCA30D")
    elif dataset == "HeLa2_gopca":
        x, y, d = load_cc_dataset(root_path, "HeLa2", representation="GOPCA20D")
    elif dataset == "HeLa2_tricycle":
        x, y, d = load_cc_dataset(root_path, "HeLa2", representation="tricycleEmbedding")
    elif dataset == "pancreas_pca":
        x, y, d = load_cc_dataset(root_path, "pancreas", representation="PCA30D")
    elif dataset == "pancreas_gopca":
        x, y, d = load_cc_dataset(root_path, "pancreas", representation="GOPCA20D")
    elif dataset == "pancreas_tricycle":
        x, y, d = load_cc_dataset(root_path, "pancreas", representation="tricycleEmbedding")
    elif dataset == "pallium_scVI":
        x, y, d = load_pallium_scVI(root_path)
    elif dataset == "pallium_scVI_small":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI", seed=seed, n=1000)
    elif dataset == "pallium_scVI_medium":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI", seed=seed, n=5000)
    elif dataset == "pallium_scVI_10pcw":
        x, y, d = load_pallium_scVI_10pcw(root_path)
    elif dataset == "pallium_scVI_10pcw_small":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI_10pcw", seed=seed, n=1000)
    elif dataset == "pallium_scVI_10pcw_medium":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI_10pcw", seed=seed, n=5000)
    elif dataset == "pallium_scVI_10pcw_cycling":
        x, y, d = load_pallium_scVI_10pcw_cycling(root_path)
    elif dataset == "pallium_scVI_10pcw_cycling_small":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI_10pcw_cycling", seed=seed, n=1000)
    elif dataset == "pallium_scVI_10pcw_cycling_medium":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI_10pcw_cycling", seed=seed, n=5000)
    elif dataset == "pallium_scVI_IPC":
        x, y, d = load_pallium_scVI_IPC(root_path)
    elif dataset == "pallium_scVI_IPC_small":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI_IPC", seed=seed, n=1000)
    elif dataset == "pallium_scVI_IPC_medium":
        x, y, d = load_subsampled_pallium(root_path, "pallium_scVI_IPC", seed=seed, n=5000)
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

    if 'd' in locals():
        return x, y, sknn_graph, pca2, d
    else:
        return x, y, sknn_graph, pca2





