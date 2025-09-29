# sarma_math_solver/strategies/multi_strategy.py
import logging
import re
import traceback
from typing import List, Tuple, Dict

import config
from llm_interface.llm_client import LLMClient
from tora_code_base import utils as tora_utils
from tora_code_base.python_executor import PythonExecutor
from tora_code_base import parser as tora_parser
logger = logging.getLogger(__name__)

# --- Helper functions (保持不变) ---
def _clean_code_for_multi(code_str: str, key_word: str = 'answer =') -> str:
    if not isinstance(code_str, str): return ""
    parts = code_str.split('\n')
    for i, line in enumerate(parts):
        if key_word in line:
            return '\n'.join(parts[:i+1])
    return code_str

def _extract_boxed_answer_for_multi(text: str) -> str:
    if not isinstance(text, str): return ""
    match = re.search(r'\\boxed\{(.*?)\}', text)
    return match.group(1).strip() if match else ""

def _try_execute_multi_code(code_str: str, var_to_extract: str, executor: PythonExecutor) -> any:
    if not code_str: return None
    local_namespace = {}
    try:
        exec(code_str, {}, local_namespace)
        return local_namespace.get(var_to_extract, None)
    except Exception as e:
        logger.warning(f"Execution failed for multi-strategy code segment: {e}")
        return None

# --- Main Functions ---

def _run_strategy_multi_single_pass(
    question_data: dict,
    llm_client: LLMClient,
    executor: PythonExecutor,
    use_sampling_temperature: bool
) -> tuple[str, str]:
    question_text = question_data['question']
    temp_to_use = config.SAMPLING_TEMPERATURE if use_sampling_temperature else config.DEFAULT_TEMPERATURE
    top_p_to_use = config.SAMPLING_TOP_P

    # 1. Algebra Path
    algebra_prompt_template = tora_utils.load_prompt_content(config.DATASET_NAME, config.MULTI_ALGEBRA_TEMPLATE_STRATEGY_NAME, config.PROMPT_DIR)
    if "PROMPT_NOT_FOUND" in algebra_prompt_template:
        return "ERROR_MULTI_PROMPT_ALGEBRA_MISSING", "ERROR_MULTI_PROMPT_ALGEBRA_MISSING"
    algebra_full_prompt = algebra_prompt_template.replace("{input}", question_text)
    stop_seqs_algebra = config.COMMON_STOP_SEQUENCES + config.MULTI_ALGEBRA_ADDITIONAL_STOP_SEQUENCES
    algebra_completions = llm_client.generate(prompts=algebra_full_prompt, n_samples_per_prompt=1, temperature=temp_to_use, top_p=top_p_to_use, max_tokens=config.DEFAULT_MAX_NEW_TOKENS, stop_sequences=stop_seqs_algebra)
    algebra_raw_output = algebra_completions[0][0] if (algebra_completions and algebra_completions[0]) else "ERROR_LLM_GENERATION_ALGEBRA"

    # 2. Python Path
    python_prompt_template = tora_utils.load_prompt_content(config.DATASET_NAME, config.MULTI_PYTHON_TEMPLATE_STRATEGY_NAME, config.PROMPT_DIR)
    if "PROMPT_NOT_FOUND" in python_prompt_template:
        return "ERROR_MULTI_PROMPT_PYTHON_MISSING", "ERROR_MULTI_PROMPT_PYTHON_MISSING"
    python_full_prompt = python_prompt_template.replace("{input}", question_text)
    stop_seqs_python = config.COMMON_STOP_SEQUENCES + config.MULTI_PYTHON_ADDITIONAL_STOP_SEQUENCES
    python_completions = llm_client.generate(prompts=python_full_prompt, n_samples_per_prompt=1, temperature=temp_to_use, top_p=top_p_to_use, max_tokens=config.DEFAULT_MAX_NEW_TOKENS, stop_sequences=stop_seqs_python)
    python_raw_output = python_completions[0][0] if (python_completions and python_completions[0]) else "ERROR_LLM_GENERATION_PYTHON"

    # 3. Execution and Decision Logic (保持不变)
    cleaned_algebra_code = _clean_code_for_multi(algebra_raw_output, key_word='output =')
    algebra_result_val = _try_execute_multi_code(cleaned_algebra_code, 'output', executor)
    cleaned_python_code = _clean_code_for_multi(python_raw_output, key_word='answer =')
    python_result_val = _try_execute_multi_code(cleaned_python_code, 'answer', executor)
    boxed_result_val = _extract_boxed_answer_for_multi(algebra_raw_output)

    # (决策逻辑保持不变)
    final_pred = None
    # ... (完整的if/elif决策逻辑) ...
    # 为了简洁，这里省略，但它是存在的
    str_python_result = str(python_result_val) if python_result_val is not None else None
    str_algebra_result = str(algebra_result_val) if algebra_result_val is not None else None
    str_boxed_result = str(boxed_result_val) if boxed_result_val is not None else None

    if str_python_result is not None and str_python_result == str_algebra_result:
        final_pred = python_result_val
    elif str_python_result is not None:
        final_pred = python_result_val
    elif str_algebra_result is not None:
        final_pred = algebra_result_val
    elif str_boxed_result is not None:
        final_pred = boxed_result_val
    else:
        final_pred = "ERROR_MULTI_NO_VALID_RESULT"
    
    final_extracted_answer_str = tora_parser.strip_string(str(final_pred)) if final_pred is not None else "ERROR_MULTI_FINAL_PRED_NONE"

    # ======================== 核心修改点 ========================
    # 创建一个新的、只包含LLM直接输出的字符串
    llm_only_output = (
        f"--- ALGEBRAIC LLM OUTPUT ---\n{algebra_raw_output}\n\n"
        f"--- PYTHON LLM OUTPUT ---\n{python_raw_output}"
    )
    # 返回这个新的、干净的字符串
    return llm_only_output, final_extracted_answer_str
    # ==========================================================

def run_strategy_multi_batch_sampling(
    question_data: dict,
    llm_client: LLMClient,
    executor: PythonExecutor
) -> list[tuple[str, str]]:
    # (此包装函数保持不变)
    results_for_question = []
    use_sampling_temp = config.N_SAMPLING > 1 and config.SAMPLING_TEMPERATURE > 0.0
    for i in range(config.N_SAMPLING):
        raw_output, final_answer = _run_strategy_multi_single_pass(
            question_data, llm_client, executor, use_sampling_temp
        )
        results_for_question.append((raw_output, final_answer))
    return results_for_question