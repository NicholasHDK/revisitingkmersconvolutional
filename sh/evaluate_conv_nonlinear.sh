#!/usr/bin/bash -l
#SBATCH --job-name=EVALUATION
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=0-12:00:00

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
K=4
DIM=256
EPOCHNUM=300
LR=0.001
NEGSAMPLEPERPOS=200
BATCH_SIZE=8
MAXREADNUM=100000
MODELNAME="conv_nonlinear"

# Define the model name


MODELNAME=scalablek=5_n_filters=1024_all_params.model.epoch_30.checkpoint

# Define the evaluation parameters
SPECIES_LIST=("reference" "plant" "marine")
MODELLIST=conv_nonlinear
DATA_DIR=${BASEFOLDER}/../dnabert-s_data/
MODEL_PATH=${BASEFOLDER}/models/${MODELNAME}

for SPECIES in ${SPECIES_LIST[@]}
do
# Define output path
OUTPUT_PATH="${BASEFOLDER}/results/${SPECIES}/${MODELNAME}.txt"

# Define the command
CMD="${PYTHON} ${SCRIPT_PATH} --data_dir ${DATA_DIR} --model_list ${MODELLIST}"
CMD=$CMD" --species ${SPECIES} --test_model_dir ${MODEL_PATH}"
CMD=$CMD" --output ${OUTPUT_PATH} --metric l2"

echo k${K}
$CMD
done



