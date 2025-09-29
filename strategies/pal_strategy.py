# sarma_math_solver/strategies/pal_strategy.py
import logging
from typing import List, Tuple, Dict

import config
from llm_interface.llm_client import LLMClient
from tora_code_base import utils as tora_utils
from tora_code_base import parser as tora_parser
from tora_code_base.python_executor import PythonExecutor

logger = logging.getLogger(__name__)

def _construct_pal_prompt(question: str, dataset_name: str) -> str:
    # ... (此函数保持不变) ...
    demo_prompt_text = tora_utils.load_prompt_content(dataset_name, config.PAL_STRATEGY_NAME, config.PROMPT_DIR)
    if "PROMPT_NOT_FOUND" in demo_prompt_text:
        return (f"Question: {question}\n```python\ndef solution():\n    pass\n```")
    context = f"Question: {question}"
    full_prompt = demo_prompt_text + context + "\n\n```python\ndef solution():\n"
    return full_prompt

def run_strategy_pal(
    question_data: Dict,
    llm_client: LLMClient,
    executor: PythonExecutor
) -> List[Tuple[str, str]]:
    question_text = question_data['question']
    dataset_name = config.DATASET_NAME
    full_prompt = _construct_pal_prompt(question_text, dataset_name)
    stop_seqs = config.COMMON_STOP_SEQUENCES + config.PAL_ADDITIONAL_STOP_SEQUENCES
    
    completions_list = llm_client.generate(
        prompts=full_prompt,
        n_samples_per_prompt=config.N_SAMPLING,
        temperature=config.SAMPLING_TEMPERATURE if config.N_SAMPLING > 1 else config.DEFAULT_TEMPERATURE,
        top_p=config.SAMPLING_TOP_P,
        max_tokens=config.DEFAULT_MAX_NEW_TOKENS,
        stop_sequences=list(set(stop_seqs))
    )[0]

    results = []
    if completions_list:
        for raw_llm_output_segment in completions_list:
            if "LLM_GENERATION_ERROR" in raw_llm_output_segment:
                extracted_answer = "ERROR_LLM_GENERATION"
            else:
                full_generation_text = full_prompt + raw_llm_output_segment
                code_to_execute = tora_parser.extract_program(full_generation_text, last_only=True)
                
                if code_to_execute:
                    pred_val, _ = executor.apply(code_to_execute)
                    extracted_answer = tora_parser.strip_string(str(pred_val))
                else:
                    extracted_answer = "ERROR_NO_CODE_EXTRACTED"
            
            # ==================== 核心修改点 ====================
            # 返回的元组第一个元素是 LLM 的直接输出，而不是完整交互文本
            results.append((raw_llm_output_segment, extracted_answer))
            # ====================================================

    else:
        for _ in range(config.N_SAMPLING):
            results.append(("ERROR_NO_COMPLETION", "ERROR_NO_COMPLETION"))
            
    return results