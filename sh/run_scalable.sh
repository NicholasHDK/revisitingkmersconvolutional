#!/bin/bash
#SBATCH --job-name=scalable_non_distributed
#SBATCH --output=%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=0-12:00:00


nvidia-smi
set -eu

# Critical for PyTorch distributed
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "port" ${MASTER_PORT} "==" ${MASTER_ADDR} "==" ${SLURM_JOB_NODELIST} "--" 

export PYTHONUNBUFFERED=1
#export NCCL_DEBUG=INFO # For printing the debug related to NCCL
export NCCL_SOCKET_IFNAME=bond0
#export NCCL_P2P_LEVEL=PHB

# Define the global variables
NAME=scalable
BASEFOLDER=$HOME/revisitingkmersconvolutional
PYTHON="srun singularity exec --nv ${HOME}/project_base.sif python3"
SCRIPT_PATH=${BASEFOLDER}/src/scalable.py

# Model Parameters
INPUT_PATH=$HOME/dnabert-s_data/train_100k.csv
LOSS_NAME="bern"
POSTFIX="_test10"
K=4
OUT_DIM=256
NEGSAMPLEPERPOS=200
MAXSEQNUM=100
EPOCHNUM=100 #100 #300
LR=0.001
BATCH_SIZE=10000 #100000 #1000 #10000
SAVE_EVERY=5 #50
DISTRIBUTED=1 #0 #1
DEVICE=gpu #gpu #None #gpu
SEED=1
START_EPOCH=0
NUM_FILTERS=136
WORKERS_NUM=8
MAXSEQNUM=100000
# Define the output path
NAME=${NAME}k=$K

OUTPUT_PATH=${BASEFOLDER}/models/${NAME}.model
LOG_DIR=${BASEFOLDER}/logs/${NAME}

# Define the command
CMD="$PYTHON ${SCRIPT_PATH} --input $INPUT_PATH --output_path ${OUTPUT_PATH} --k ${K} --out_dim ${OUT_DIM}"
CMD="${CMD} --neg_sample_per_pos ${NEGSAMPLEPERPOS} --max_seq_num ${MAXSEQNUM}"
CMD="${CMD} --epoch $EPOCHNUM --lr $LR --batch_size ${BATCH_SIZE} --save_every ${SAVE_EVERY} --distributed ${DISTRIBUTED}"
CMD="${CMD} --device ${DEVICE} --loss_name ${LOSS_NAME} --seed ${SEED} --workers ${WORKERS_NUM}"
# Run the command
$CMD

