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

Example command to train a Conf-GFlowNet on ibuprofen `CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O` and TorchANI as the energy proxy estimator for the reward:

```bash
python main.py     env=conformers/conformer     policy=heterogeneous     proxy=molecule     gflownet=trajectorybalance     env.buffer.test.n=10     gflownet.optimizer.batch_size.forward=16  env.flex_bond_lengths=None env.flex_bond_angles=None env.n_torsion_angles= -1 
```

Where:

- env=conformers/conformer: Specifies the environment configuration for **conformer generation**.
- policy=heterogeneous: Defines the GFlowNet policy to handle a **diverse/mixed set of actions** (e.g., different types of structural modifications).
- proxy=molecule: Denotes the proxy model used for the reward, suggesting the reward is derived from the molecular structure or properties. Other options from original implementation: conformers/tblite` for GFN2-xTB, `conformers/xtb` for GFN-FF, or `conformers/torchani` for TorchANI.
- gflownet=trajectorybalance: Sets the GFlowNet objective to the Trajectory Balance algorithm.
- env.buffer.test.n = 10: Sets the number of testing samples n  to be stored in the environment's test buffer to 10.
- gflownet.optimizer.batch_size.forward=16: Sets the batch size for the forward pass of the GFlowNet optimizer to 16.
- env.flex_bond_lengths=None: Enables full flexibility for those geometry parameters by triggering the automatic discovery of all flexible bonds. Other option: empty list indicating no specific bond lengths are allowed to be flexible/modified during the conformational search.
- env.flex_bond_angles=None: Enables full flexibility for those geometry parameters by triggering the automatic discovery of all flexible angles. Other option: empty list indicating no specific bond lengths are allowed to be flexible/modified during the conformational search.
- env.n_torsion_angles=-1: Sets the number of torsion angles to be controlled to -1, which usually means all rotatable bonds are included in the search space.

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
