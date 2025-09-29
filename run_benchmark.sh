#!/bin/bash

# --- GPU Configuration ---
#  Specify the GPU IDs to be used for the parallel benchmark run.
GPUS_TO_USE=(0 3)


# --- Script Configuration ---
#  The main Python script that will be executed to run the benchmark for each strategy.
MAIN_SCRIPT="benchmark_strategies.py"

#  Get the total number of GPUs to determine the number of parallel workers.
NUM_GPUS=${#GPUS_TO_USE[@]}
#  Python executable command.
PYTHON_EXE="python"
#  Create a unique identifier for this benchmark run using date, time, and process ID.
RUN_ID="benchmark_$(date +%Y%m%d_%H%M%S)_$$"

echo "Starting BENCHMARKING run with ${NUM_GPUS} workers on GPUs: ${GPUS_TO_USE[*]}"
echo "RUN_ID: ${RUN_ID}"


#  This function is triggered on interrupt (Ctrl+C) or termination signals to clean up child processes.
cleanup() {
    echo ""
    echo "Interrupt received, killing child processes..."
    pkill -P $$
    exit 1
}
#  Trap SIGINT (Ctrl+C) and SIGTERM signals and call the cleanup function.
trap cleanup SIGINT SIGTERM

#  Loop through the number of GPUs to launch a worker for each one.
for (( i=0; i<${NUM_GPUS}; i++ )); do
    #  Assign a unique worker ID to each process.
    WORKER_ID=${i}
    #  Get the specific GPU ID for the current worker from the GPUS_TO_USE array.
    GPU_ID=${GPUS_TO_USE[i]}
    echo "Launching Benchmark Worker ${WORKER_ID} on GPU ${GPU_ID}..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_EXE} ${MAIN_SCRIPT} \
        --worker_id ${WORKER_ID} \
        --num_workers ${NUM_GPUS} \
        --run_id "${RUN_ID}" &
done

wait
echo "Benchmark run completed for RUN_ID: ${RUN_ID}"