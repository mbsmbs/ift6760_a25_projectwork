# Conf-GFlowNet

This repository implements the experiments described in [Towards equilibrium molecular conformation generation with GFlowNets](https://arxiv.org/abs/2310.14782) by Volokhova & Koziarski et al.

---

## IFT6760B A25 Project Work (our fork)

This repository is a **fork** of `GFNOrg/conf-gfn` used for the IFT6760 A25
project on sampling full intrinsic coordinates (torsions, bond angles and
bond lengths) with GFlowNets.

## Installation

The commands below are the original installation instructions from the
Conf-GFlowNet repository. They assume a **Linux** machine with an
**NVIDIA GPU (CUDA 11.7)**:

```bash
conda create -n confgfn python=3.8
source activate confgfn

conda install mamba -n base -c conda-forge

mamba install xtb -c conda-forge
mamba install tblite -c conda-forge
mamba install tblite-python -c conda-forge

# Update pip
python -m pip install --upgrade pip
# Install PyTorch family
python -m pip install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
# Install DGL (see https://www.dgl.ai/pages/start.html)
python -m pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html
# Requirements to run
python -m pip install numpy pandas hydra-core tqdm torchtyping six xtb scikit-learn torchani==2.2.3 rdkit wurlitzer wandb matplotlib dgllife ultranest
python -m pip install -U --no-deps pytorch3d==0.3.0
```

### Recommended setup

To reproduce our experiments, create a new environment and install the dependencies specified in requirements.txt

For other environments (e.g. macOS, Windows, or Linux/CPU-only), a simpler setup that we use in this fork is:

```bash
# Create and activate the environment
conda create -n confgfn python=3.8
conda activate confgfn

# Install the exact dependencies used in this fork
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Training

Example command to train a Conf-GFlowNet on the specific molecule `CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O` and TorchANI as the energy proxy estimator for the reward:

```bash
python main.py +experiments=ai4mat23/mlp_torchani device=cpu 'env.smiles="CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"' proxy=conformers/torchani logger.do.online=True user.logdir.root=logs
```

Where:

- `+experiments=ai4mat23/mlp_torchani` points to a config file with hyperparameters defined (see [here](https://github.com/GFNOrg/conf-gfn/blob/main/config/experiments/ai4mat23/mlp_torchani.yaml)).
- `device=cpu` specifies the device (`cpu` or `cuda`).
- `'env.smiles="CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"'` specifies the SMILES of a molecule. Alternatively, you can use `env.smiles=ID`, e.g. `env.smiles=0`, to run on one of the [predefined molecules](https://github.com/GFNOrg/conf-gfn/blob/main/gflownet/envs/conformers/conformer.py) used in the experiments described in the paper.
- `proxy=conformers/torchani` denotes the proxy model: either `conformers/tblite` for GFN2-xTB, `conformers/xtb` for GFN-FF, or `conformers/torchani` for TorchANI.
- `logger.do.online=True` whether to log the results to wandb.
- `user.logdir.root=logs` points to a directory in which log files will be stored.

## Citation

```bibtex
@article{volokhova2023towards,
  title={Towards equilibrium molecular conformation generation with GFlowNets},
  author={Volokhova, Alexandra and Koziarski, Micha{\l} and Hern{\'a}ndez-Garc{\'\i}a, Alex and Liu, Cheng-Hao and Miret, Santiago and Lemos, Pablo and Thiede, Luca and Yan, Zichao and Aspuru-Guzik, Al{\'a}n and Bengio, Yoshua},
  journal={arXiv preprint arXiv:2310.14782},
  year={2023}
}
```

## Acknowledgment

This repository is based on (and would not be possible without) [github.com/alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet/), a library for all of your GFlowNet needs.
