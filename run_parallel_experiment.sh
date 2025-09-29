#!/bin/bash

# --- GPU Configuration ---
#  Specify the GPU IDs to be used for the parallel experiment.
# Example: (0 1) will use GPU 0 and GPU 1.
GPUS_TO_USE=(0 1 2 3) 

# --- Script Configuration ---
#  Get the total number of GPUs to determine the number of parallel workers.
NUM_GPUS=${#GPUS_TO_USE[@]} 
#  Python executable command.
PYTHON_EXE="python"
#  The main Python script to be executed by each worker.
MAIN_SCRIPT="main_experiment.py"

# --- Run Identifier ---
#  Create a unique identifier for this experiment run based on date, time, and process ID.
RUN_ID="run_$(date +%Y%m%d_%H%M%S)_$$"

echo "Starting experiment with ${NUM_GPUS} workers on specified GPUs: ${GPUS_TO_USE[*]}"
echo "RUN_ID for this execution: ${RUN_ID}"
echo "Detailed logs will be in: ./logs/${RUN_ID}/worker_stdout_logs/"
echo "Final JSONL results will be in: ./logs/${RUN_ID}/"
echo "--------------------------------------------------------------------------"

# --- Cleanup Function ---
#  This function is triggered on interrupt (Ctrl+C) or termination signals to clean up child processes.
cleanup() {
    echo "" 
    echo "Interrupt received, attempting to kill child processes..."
    #  Kill all processes that are children of the current script's process.
    pkill -P $$
    echo "Cleanup attempt complete. Some processes might need manual termination."
    exit 1
}
#  Trap SIGINT (Ctrl+C) and SIGTERM signals and call the cleanup function.
trap cleanup SIGINT SIGTERM

# --- Main Execution Loop ---
#  Loop through the number of GPUs to launch a worker for each one.
for (( i=0; i<${NUM_GPUS}; i++ ))
do
    #  Assign a unique worker ID to each process.
    WORKER_ID=${i}
    
    #  Get the specific GPU ID for the current worker from the GPUS_TO_USE array.
    GPU_ID=${GPUS_TO_USE[i]}
    
    echo "Launching Worker ${WORKER_ID} on GPU ${GPU_ID}..."

    # --- Launch Worker Process ---
    #  Set the CUDA_VISIBLE_DEVICES to the assigned GPU ID, then execute the main Python script.
    # Pass worker_id, num_workers, and run_id as arguments to the script.
    # The '&' at the end runs the command in the background, allowing the loop to continue and launch other workers.
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