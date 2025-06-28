#!/usr/bin/bash -l
#SBATCH --job-name=EVAL_F2048_EPOCH
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=0-12:00:00
#SBATCH --array=1-9

# Define the global variables
BASEFOLDER=${HOME}/revisitingkmersconvolutional
PYTHON="singularity exec --nv ${BASEFOLDER}/../project_base.sif python3"
SCRIPT_PATH=${BASEFOLDER}/evaluation/binning.py
RESULTS_FOLDER=${BASEFOLDER}/results

export PYTHONPATH=${PYTHONPATH}:${BASEFOLDER}

# Check if a folder exists or not
if ! [ -d ${RESULTS_FOLDER} ]; then
mkdir ${RESULTS_FOLDER}
fi
if ! [ -d ${RESULTS_FOLDER}/reference ]; then
mkdir ${RESULTS_FOLDER}/reference
fi
if ! [ -d ${RESULTS_FOLDER}/marine ]; then
mkdir ${RESULTS_FOLDER}/marine
fi
if ! [ -d ${RESULTS_FOLDER}/plant ]; then
mkdir ${RESULTS_FOLDER}/plant
fi


# Model Parameters
POSTFIX=""
K=6
DIM=256
EPOCHNUM=300
LR=0.001
NEGSAMPLEPERPOS=200
BATCH_SIZE=8
MAXREADNUM=100000
MODELNAME="scalable"
EPOCH=$((${SLURM_ARRAY_TASK_ID}*10))
# Define the model name


MODELNAME=k${K}.model.epoch_${EPOCH}.checkpoint

# Define th parameters
SPECIES_LIST=("reference" "plant" "marine")
MODELLIST=conv_nonlinear
DATA_DIR=${BASEFOLDER}/../dnabert-s_data
MODEL_PATH=${BASEFOLDER}/models/F2048/${MODELNAME}

for SPECIES in ${SPECIES_LIST[@]}
do
# Define output path
OUTPUT_PATH="${BASEFOLDER}/results/${SPECIES}/${MODELNAME}.txt"

# Define the command
CMD="${PYTHON} ${SCRIPT_PATH} --data_dir ${DATA_DIR} --model_list ${MODELLIST}"
CMD=$CMD" --species ${SPECIES} --test_model_dir ${MODEL_PATH}"
CMD=$CMD" --output ${OUTPUT_PATH} --metric l2"

$CMD
done



