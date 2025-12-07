#!/bin/bash
#SBATCH --account=aip-irina
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=8G         # memory per node
#SBATCH --time=23:30:00
#SBATCH --cpus-per-task=16


JOB_NAME="molconf"
PATH_BASE="/project/aip-irina/julkuhn/ift6760_a25_projectwork"
OUT_PATH="${PATH_BASE}/output"

DATE_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
JOB_NAME="${JOB_NAME}_${DATE_TIME}"

#SBATCH -o ${OUT_PATH}/logs/molconf_%j.out
#SBATCH -e ${OUT_PATH}/logs/molconf_%j.err


# venv aktivieren
source /project/aip-irina/julkuhn/ift6760_a25_projectwork/.venv/bin/activate
echo "Python:" $(which python)

# eegdash strikt offline + Datenbasis
export EEGDASH_OFFLINE=1
export EEGDASH_DISABLE_CLOUD=1
export EEGDASH_ALLOW_S3=0
export EEGDASH_DATA_DIR=/project/aip-irina/julkuhn/ift6760_a25_projectwork/data

# lokales TMP (vermeidet /home)
export TMPDIR=/project/aip-irina/julkuhn/ift6760_a25_projectwork/tmp
mkdir -p "$TMPDIR"

# Ensure logs directory exists
mkdir -p /project/aip-irina/julkuhn/ift6760_a25_projectwork/logs/gflownet

export AWS_DEFAULT_REGION=us-east-2
export AWS_RETRY_MODE=standard
export AWS_MAX_ATTEMPTS=10
export NUM_DATALOADER_WORKERS=1

# zur Kontrolle ins Log schreiben:
echo "EEGDASH_OFFLINE=$EEGDASH_OFFLINE"
echo "EEGDASH_DATA_DIR=$EEGDASH_DATA_DIR"
echo "TMPDIR=$TMPDIR"

#srun python -m test_startkit_challenge_1 
# srun python -m run_itransformer_ch1
# Override user config to ensure correct log directory

module load python/3.10 scipy-stack
virtualenv --system-site-packages $SCRATCH/conf-gfn-env
source $SCRATCH/conf-gfn-env/bin/activate
#pip install --no-index torch hydra-core omegaconf rdkit dgl

# 1. Export the variable separately
export HYDRA_FULL_ERROR=1

# 2. Run the command (remove the variable from this line)
srun HYDRA_FULL_ERROR=1 python main.py \
    env=conformers/conformer \
    policy=heterogeneous \
    proxy=molecule \
    gflownet=trajectorybalance \
    env.buffer.test.n=1000 \
    gflownet.optimizer.batch_size.forward=16 \
    env.length_traj=5 \
    env.length_scale=0.01 \
    env.angle_scale=0.05 \
    logger.do.online=False \
    logger.do.checkpoint=True