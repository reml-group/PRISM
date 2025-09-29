#!/bin/bash


GPUS_TO_USE=(0 1) 


MAIN_SCRIPT="main_ablation.py"


ABLATION_MODE="none"



NUM_GPUS=${#GPUS_TO_USE[@]}
PYTHON_EXE="python"
RUN_ID="run_${ABLATION_MODE}_$(date +%Y%m%d_%H%M%S)_$$"


echo "=========================================================================="
echo "Starting Experiment Run"
echo "=========================================================================="
echo "Run ID: ${RUN_ID}"
echo "Main Script: ${MAIN_SCRIPT}"
echo "Ablation Mode: ${ABLATION_MODE}"
echo "Number of Workers: ${NUM_GPUS} on GPUs: ${GPUS_TO_USE[*]}"
echo "--------------------------------------------------------------------------"
echo "Detailed logs will be in: ./logs/${RUN_ID}/worker_stdout_logs/"
echo "Final JSONL results will be in: ./logs/${RUN_ID}/"
echo "--------------------------------------------------------------------------"


cleanup() {
    echo ""
    echo "Interrupt received, attempting to kill child processes..."
    pkill -P $$
    echo "Cleanup attempt complete."
    exit 1
}
trap cleanup SIGINT SIGTERM

for (( i=0; i<${NUM_GPUS}; i++ )); do
    WORKER_ID=${i}
    GPU_ID=${GPUS_TO_USE[i]}
    
    echo "Launching Worker ${WORKER_ID} on GPU ${GPU_ID}..."

    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_EXE} ${MAIN_SCRIPT} \
        --worker_id ${WORKER_ID} \
        --num_workers ${NUM_GPUS} \
        --run_id "${RUN_ID}" \
        --ablation_mode ${ABLATION_MODE} &
done

echo "--------------------------------------------------------------------------"
echo "All workers launched. Waiting for completion..."
wait
echo ""
echo "=========================================================================="
echo "All workers have completed their tasks for Run ID: ${RUN_ID}"
echo "=========================================================================="
