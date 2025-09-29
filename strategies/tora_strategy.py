# sarma_math_solver/strategies/tora_strategy.py
import re
import logging
from typing import List, Tuple, Dict

import config
from llm_interface.llm_client import LLMClient
from tora_code_base import utils as tora_utils
from tora_code_base import parser as tora_parser
from tora_code_base.python_executor import PythonExecutor

logger = logging.getLogger(__name__)

def _construct_tora_prompt(question: str, dataset_name: str) -> str:
    # ... (此函数保持不变) ...
    demo_prompt_text = tora_utils.load_prompt_content(dataset_name, config.TORA_STRATEGY_NAME, config.PROMPT_DIR)
    if "PROMPT_NOT_FOUND" in demo_prompt_text:
        return (f"Question: {question}\nSolution:\n")
    context = f"Question: {question}\n\nSolution:"
    return demo_prompt_text + context

def run_strategy_tora(
    question_data: Dict,
    llm_client: LLMClient,
    executor: PythonExecutor
) -> Tuple[str, str]:
    question_text = question_data['question']
    dataset_name = config.DATASET_NAME
    current_prompt = _construct_tora_prompt(question_text, dataset_name)
    stop_seqs = config.COMMON_STOP_SEQUENCES + config.TORA_ADDITIONAL_STOP_SEQUENCES
    temp = config.SAMPLING_TEMPERATURE if config.N_SAMPLING > 1 else config.DEFAULT_TEMPERATURE
    
    completions = llm_client.generate(
        prompts=current_prompt, n_samples_per_prompt=1, temperature=temp,
        top_p=config.SAMPLING_TOP_P, max_tokens=config.DEFAULT_MAX_NEW_TOKENS,
        stop_sequences=list(set(stop_seqs))
    )

    if not completions or not completions[0] or "LLM_GENERATION_ERROR" in completions[0][0]:
        return "ERROR_LLM_GENERATION", "ERROR_LLM_GENERATION"

    llm_output_segment = completions[0][0].rstrip()
    
    boxed_match = re.search(r'\\boxed\{(.*?)\}', llm_output_segment)
    if boxed_match:
        answer = tora_parser.strip_string(boxed_match.group(1))
        # ==================== 核心修改点 ====================
        return llm_output_segment, answer
        # ====================================================

    if "```python" in llm_output_segment:
        full_interaction_for_parsing = current_prompt + llm_output_segment
        code_to_execute = tora_parser.extract_program(full_interaction_for_parsing, last_only=True)
        if code_to_execute:
            pred_val, _ = executor.apply(code_to_execute)
            answer = tora_parser.strip_string(str(pred_val))
            # ==================== 核心修改点 ====================
            return llm_output_segment, answer
            # ====================================================
    
    return llm_output_segment, "ERROR_TORA_NO_ANSWER_OR_CODE"

def run_strategy_tora_batch_sampling(
    question_data: Dict,
    llm_client: LLMClient,
    executor: PythonExecutor
) -> List[Tuple[str, str]]:
    # ... (此包装函数保持不变) ...
    results_for_question = []
    for i in range(config.N_SAMPLING):
        logger.info(f"--- Starting ToRA Sample {i+1}/{config.N_SAMPLING} for QID {question_data.get('qid', 'N/A')} ---")
        raw_interaction, final_answer = run_strategy_tora(question_data, llm_client, executor)
        results_for_question.append((raw_interaction, final_answer))
        logger.info(f"  ToRA Sample {i+1} Final Extracted Answer: {final_answer}")
    return results_for_question