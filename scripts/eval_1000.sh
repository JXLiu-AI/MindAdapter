#!/bin/bash
# Run eval_1000 for a given model_name and store logs
set -e
source /data/shared/miniconda/etc/profile.d/conda.sh
conda activate MindAligner

MODEL_NAME=${1:-5->7_64shot}
LOG_DIR=logs_eval_1000
mkdir -p ${LOG_DIR}
LOG_FILE=${LOG_DIR}/${MODEL_NAME//\/>/_}.log

echo "Running eval_1000 for ${MODEL_NAME}, logging to ${LOG_FILE}"
python eval_1000.py --model_name "${MODEL_NAME}" > "${LOG_FILE}" 2>&1 || (echo "eval failed, see ${LOG_FILE}" && exit 1)

echo "Done. See ${LOG_FILE} for details."