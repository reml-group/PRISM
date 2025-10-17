# sarma_math_solver/config.py


LLM_PROVIDER = "vllm"
# Path to your actual language model. Please modify this.
MODEL_NAME_OR_PATH = "path" 
# --- Path to the tokenizer, usually same as the model path ---
TOKENIZER_PATH = MODEL_NAME_OR_PATH 


VLLM_TRUST_REMOTE_CODE = True 
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.55

# --- Sampling Configuration for Pass@k ---
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0 
DEFAULT_TOP_P = 1.0 
N_SAMPLING = 5
SAMPLING_TEMPERATURE = 0.7 
SAMPLING_TOP_P = 1.0

# --- Dataset Configuration ---
DATA_DIR = "path" 
TEST_DATA_PATH = "path" 
SCORED_DATA_PATH = "path"


# --- Execution Mode Selection ---
# Select the execution mode. Options: "sarma" (PRISM), "cot", "pal", "tora", "multi".
EXECUTION_MODE = "cot" 


PROMPT_DIR = "./prompts/" 

# Confidence threshold for Confident Routing.
TAU_C = 0.35 
# Ambiguity margin for Deliberative Routing.
TAU_A = 0.1


LOG_DIR = "./logs/"


PYTHON_EXEC_TIMEOUT = 20 

COT_STRATEGY_NAME = "cot"
PAL_STRATEGY_NAME = "pal"
TORA_STRATEGY_NAME = "tora"
MULTI_STRATEGY_NAME = "multi"


COMMON_STOP_SEQUENCES = ["</s>", "<|endoftext|>", "<|im_end|>", "```output"]


