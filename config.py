# sarma_math_solver/config.py

# --- LLM Configuration ---
LLM_PROVIDER = "vllm"
# --- Path to your local LLM model ---
# Path to your actual language model. Please modify this.
MODEL_NAME_OR_PATH = "/nfsdat/home/shqislm/yinziang/model/qwen/Qwen2.5-Math-7B-Instruct/" 
#MODEL_NAME_OR_PATH = "/nfsdat/home/jmaslm/yinziang/model/Qwen2.5-Math-1.5B/"
# --- Path to the tokenizer, usually same as the model path ---
TOKENIZER_PATH = MODEL_NAME_OR_PATH 

# --- For vLLM, some models might require trust_remote_code=True ---
# Set to True if the model requires trusting remote code (e.g., for Qwen models).
VLLM_TRUST_REMOTE_CODE = True 
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.55

# --- Sampling Configuration for Pass@k ---
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0 # ToRA project often sets this to 0 for deterministic output
DEFAULT_TOP_P = 1.0 
# --- Number of samples to generate for each problem ---
N_SAMPLING = 5
# --- Temperature for sampling when N_SAMPLING > 1 ---
# If N_SAMPLING is 1 (e.g., for pass@1 or debugging), can be set to 0 for deterministic output.
SAMPLING_TEMPERATURE = 0.7 
# When temperature is 0, top_p must be 1.0
SAMPLING_TOP_P = 1.0

# --- Dataset Configuration ---
DATA_DIR = "/nfsdat/home/shqislm/yinziang/sarma_math_solver3_mode/" # Relative to the project root
# --- Path to the test dataset file ---
TEST_DATA_PATH = "/nfsdat/home/shqislm/yinziang/sarma_math_solver3_mode/math500_idx.jsonl" 
# --- Path to the scored/recorded output file from previous runs ---
SCORED_DATA_PATH = "/nfsdat/home/shqislm/yinziang/sarma_math_solver3_mode/math500_test_record2.jsonl"


# --- Execution Mode Selection ---
# Select the execution mode. Options: "sarma" (PRISM), "cot", "pal", "tora", "multi".
EXECUTION_MODE = "cot" 


# --- Prompt Configuration ---
# Directory where prompt templates are stored.
PROMPT_DIR = "./prompts/" # Relative to the project root

# --- Experiment Configuration (for PRISM routing) ---
# Confidence threshold for Confident Routing.
TAU_C = 0.35 # Example value, please adjust based on your statistical analysis
# Ambiguity margin for Deliberative Routing.
TAU_A = 0.1 # Example value, please adjust based on your statistical analysis

# --- Output Configuration ---
# Directory to save logs and results.
LOG_DIR = "./logs/"

# --- Python Executor Configuration ---
# Timeout for executing Python code snippets (in seconds).
PYTHON_EXEC_TIMEOUT = 20 

# --- Strategy Names (for consistency) ---
COT_STRATEGY_NAME = "cot"
PAL_STRATEGY_NAME = "pal"
TORA_STRATEGY_NAME = "tora"
MULTI_STRATEGY_NAME = "multi"

# --- Stop Sequences for LLM ---
# These are common stop tokens; specific strategies might add more.
# Common stop sequences to halt LLM generation.
COMMON_STOP_SEQUENCES = ["</s>", "<|endoftext|>", "<|im_end|>", "```output"]