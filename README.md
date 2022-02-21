# vis_utils
Some utitlity functions for visualisation methods such as t-SNE, NCVis and UMAP. 
In particular, it implements the loss functions of the above in closed form using [pykeops](https://github.com/getkeops/keops/tree/master/pykeops).


## Installation
Clone the repo
```
git clone https://github.com/sdamrich/vis_utils
```
Create conda environment from file
```
cd vis_utils
conda env create -f environment.yml
```

Optionally: Install [openTSNE](https://github.com/sdamrich/openTSNE) from source if you want to use the t-SNE wrapper.

```
conda activate vis_utils
python setup.py install
```


