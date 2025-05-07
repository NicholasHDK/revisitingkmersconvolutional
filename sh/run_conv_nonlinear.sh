#!/usr/bin/bash -l
#SBATCH --job-name=RUN_CONV_NONLINEAR
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH  --partition=batch

nvidia-smi

# Set up environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 50000))

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# NCCL Debugging and Communication Setup
#export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0

# Define the global variables
MODELNAME=conv_nonlinear
BASEFOLDER=${HOME}/revisitingkmers
PYTHON="singularity exec --nv ${HOME}/project_base.sif python3"
SCRIPT_PATH=${BASEFOLDER}/src/conv_nonlinear.py

# Model Parameters
INPUT_PATH=$HOME/train_100k.csv
POSTFIX=""
K=8
DIM=256
EPOCHNUM=300
LR=0.001
NEGSAMPLEPERPOS=200
BATCH_SIZE=4
MAXREADNUM=100000
SEED=26042024
CHECKPOINT=1

# Define the output path
OUTPUT_PATH=${BASEFOLDER}/models/${MODELNAME}_train_100k_k=${K}_d=${DIM}_negsampleperpos=${NEGSAMPLEPERPOS}
OUTPUT_PATH=${OUTPUT_PATH}_epoch=${EPOCHNUM}_LR=${LR}_batch=${BATCH_SIZE}_maxread=${MAXREADNUM}_seed=${SEED}${POSTFIX}.log

# Define the command for running distributed training with torchrun
CMD="$PYTHON -m torch.distributed.run --nproc_per_node=4 --standalone"
CMD="${CMD} ${SCRIPT_PATH} --input $INPUT_PATH --k ${K} --epoch $EPOCHNUM --lr $LR --inputs 256"
CMD="${CMD} --neg_sample_per_pos ${NEGSAMPLEPERPOS} --max_read_num ${MAXREADNUM}"
CMD="${CMD} --batch_size ${BATCH_SIZE} --device cuda --output ${OUTPUT_PATH} --seed ${SEED} --checkpoint ${CHECKPOINT}"

# Start GPU monitoring in background
while true; do
    echo "==== $(date) ====" >> gpu_usage.log
    nvidia-smi >> gpu_usage.log
    sleep 100
done &
MONITOR_PID=$!
#Run the command
$CMD 

kill $MONITOR_PID
echo "Job finished at $(date)"
