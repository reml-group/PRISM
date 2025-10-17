#!/bin/bash


GPUS_TO_USE=(0 1 2 3) 


NUM_GPUS=${#GPUS_TO_USE[@]} 
PYTHON_EXE="python"
MAIN_SCRIPT="main_experiment.py"


RUN_ID="run_$(date +%Y%m%d_%H%M%S)_$$"

echo "Starting experiment with ${NUM_GPUS} workers on specified GPUs: ${GPUS_TO_USE[*]}"
echo "RUN_ID for this execution: ${RUN_ID}"
echo "Detailed logs will be in: ./logs/${RUN_ID}/worker_stdout_logs/"
echo "Final JSONL results will be in: ./logs/${RUN_ID}/"
echo "--------------------------------------------------------------------------"

cleanup() {
    echo "" 
    echo "Interrupt received, attempting to kill child processes..."
    pkill -P $$
    echo "Cleanup attempt complete. Some processes might need manual termination."
    exit 1
}

trap cleanup SIGINT SIGTERM


#  Loop through the number of GPUs to launch a worker for each one.
for (( i=0; i<${NUM_GPUS}; i++ ))
do
    WORKER_ID=${i}
    
    GPU_ID=${GPUS_TO_USE[i]}
    
    echo "Launching Worker ${WORKER_ID} on GPU ${GPU_ID}..."

    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_EXE} ${MAIN_SCRIPT} \
        --worker_id ${WORKER_ID} \
        --num_workers ${NUM_GPUS} \
        --run_id "${RUN_ID}" &

done

echo "--------------------------------------------------------------------------"
echo "All workers launched. Waiting for completion..."
#  'wait' command waits for all background jobs (the workers) to finish before the script exits.
wait
echo "All workers have completed their tasks."
echo "Experiment ${RUN_ID} finished."
