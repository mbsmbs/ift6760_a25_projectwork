# Conf-GFlowNet (Our starting point)

This repository implements the experiments described in [Towards equilibrium molecular conformation generation with GFlowNets](https://arxiv.org/abs/2310.14782) by Volokhova & Koziarski et al.

---

## IFT6760B A25 Project Work (our fork)

This repository is a **fork** of `GFNOrg/conf-gfn` used for the IFT6760 A25
project on sampling full intrinsic coordinates (torsions, bond angles and
bond lengths) with GFlowNets.

Our work is organized in two stages:

- **Stage 1 (used in the report):** extended `Conf-GFlowNet` environment where a subset of
  bond lengths and bond angles (BLA) are controllable alongside torsions. We train and
  compare a torsion-only baseline and a torsions+ BLA model on ibuprofen using a
  TorchANI-based reward.
- **Stage 2 (prototype):** ongoing re-implementation with a heterogeneous policy
  (separate heads for torsions vs. geometric coordinates) and Jacobian-aware rewards.
  This code is experimental and not used for the main quantitative results in the paper.

---

## Installation

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

### Original Conf-GFlowNet CUDA setup (optional)

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

## Training (our experiments)

All commands below are run from the repository root.

### Stage 1 – torsion-only baseline

```bash
python main.py +experiment=<torsion_only_experiment> device=cpu \
  env.smiles="CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O" \
  proxy=conformers/torchani user.logdir.root=logs/hybrid_extended
```

### Stage 1 – extended torsions + BLA model

```bash
python main.py +experiment=<torsions_bla_experiment> device=cpu \
  env.smiles="CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O" \
  proxy=conformers/torchani user.logdir.root=logs/hybrid_extended
```

Replace <torsion_only_experiment> and <torsions_bla_experiment> with the Hydra
config names you used for the baseline and extended runs (e.g. the configs under
conf/hybrid_extended/). These commands reproduce the logs in logs/hybrid_extended/
used to generate the energy–RMSD comparison plots in the report.

### Comparing energy and RMSD (Figure 1 in the report)

```bash
python project_debug/compare_rmsd_and_energy.py \
  base_dir=logs/hybrid_extended/<baseline_run_id> \
  ext_dir=logs/hybrid_extended/<extended_run_id>
```

This script creates energy_rmsd_comparison.png, which is the figure reported in the
Results section.

### Stage 2 - full intrinsic coordinates (still processing)

The second-stage full–intrinsic-coordinate implementation (heterogeneous policy,
Jacobian-aware reward) is located in:
[Link to the development branch](https://github.com/mbsmbs/ift6760_a25_projectwork/tree/fully_flexible?tab=readme-ov-file)

## Acknowledgment

This repository is based on (and would not be possible without) [github.com/alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet/), a library for all of your GFlowNet needs.
