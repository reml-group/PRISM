# sarma_math_solver/tora_code_base/utils.py
import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any, Dict # Added Dict for type hints
import logging # <--- 添加导入
logger = logging.getLogger(__name__) 
# --- Project-Specific: Import config directly ---
# This assumes that when utils.py is imported (e.g., by a strategy module,
# which in turn is imported by main_experiment.py), the sys.path has been
# set up by main_experiment.py to allow direct imports from the project root.
import config


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception as e: # Catch specific exception if possible
                print(f"Error in loading line: '{line.strip()}'. Error: {e}")
                # Decide whether to exit or continue
                # exit() # Original behavior
                continue # Skip problematic line and continue


def save_jsonl(samples: Iterable[Dict], save_path: Union[str, Path]):
    # ensure path
    folder = os.path.dirname(save_path)
    if folder: # Ensure folder is not an empty string if save_path is just a filename
        os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {len(list(samples)) if isinstance(samples, list) else 'multiple'} samples to {save_path}")


def lower_keys(example: Dict) -> Dict:
    new_example = {}
    for key, value in example.items():
        # It's safer to check if key is a string before calling .lower()
        if isinstance(key, str) and key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example


def load_prompt_content(
    dataset_name: str,
    prompt_strategy_type: str, # e.g., config.COT_STRATEGY_NAME, config.MULTI_ALGEBRA_TEMPLATE_STRATEGY_NAME
    base_prompt_dir: str
) -> str:
    """
    Loads few-shot prompt content from a file.
    Applies dataset name mappings similar to the original ToRA project.
    """
    original_dataset_name = dataset_name # 保存原始名称用于可能的日志或调试

    # 步骤1: 数据集名称映射 (复用ToRA项目的逻辑)
    if dataset_name in ['gsm-hard', 'svamp', 'tabmwp', 'asdiv', 'mawps', 'aqua_rat']: # 添加了 aqua_rat
        dataset_name = "gsm8k"
        logger.debug(f"Mapped dataset '{original_dataset_name}' to '{dataset_name}' for prompt loading.")
    if dataset_name in ['math-oai']:
        dataset_name = "math"
        logger.debug(f"Mapped dataset '{original_dataset_name}' to '{dataset_name}' for prompt loading.")

    # 步骤2: 根据策略类型确定文件名
    # 对于 CoT, PAL, ToRA，我们期望的文件名格式是: {mapped_dataset_name}_{strategy_name_constant}.md
    # 例如: gsm8k_cot.md (这里 strategy_name_constant 已经是 'cot', 'pal', 'tora')
    
    # 对于 Multi 策略，我们之前约定的是 multi_algebra_template.txt 和 multi_python_template.txt
    # prompt_strategy_type 会是 config.MULTI_ALGEBRA_TEMPLATE_STRATEGY_NAME ("multi_algebra_template")

    prompt_file_name = ""
    if prompt_strategy_type == config.MULTI_ALGEBRA_TEMPLATE_STRATEGY_NAME:
        prompt_file_name = "multi_algebra_template.txt" # 直接使用我们约定的文件名
    elif prompt_strategy_type == config.MULTI_PYTHON_TEMPLATE_STRATEGY_NAME:
        prompt_file_name = "multi_python_template.txt" # 直接使用我们约定的文件名
    elif prompt_strategy_type in [config.COT_STRATEGY_NAME, config.PAL_STRATEGY_NAME, config.TORA_STRATEGY_NAME]:
        prompt_file_name = f"{dataset_name}_{prompt_strategy_type}.md"
    else:
        # 处理未知的 prompt_strategy_type
        error_msg = f"PROMPT_ERROR: Unknown prompt_strategy_type '{prompt_strategy_type}' for dataset '{original_dataset_name}'"
        logger.error(error_msg)
        return error_msg

    prompt_path = os.path.join(base_prompt_dir, prompt_file_name)
    logger.debug(f"Attempting to load prompt: dataset='{original_dataset_name}' (mapped to '{dataset_name}'), strategy_type='{prompt_strategy_type}', path='{prompt_path}'")

    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as fp:
            prompt_content = fp.read().strip() + "\n\n"
    else:
        error_msg = f"PROMPT_NOT_FOUND: Could not find prompt file at '{prompt_path}' (for original dataset '{original_dataset_name}', strategy '{prompt_strategy_type}')"
        logger.error(error_msg)
        prompt_content = error_msg # 返回错误信息
    return prompt_content


def construct_final_prompt(
    question_text: str,
    few_shot_prompt_content: str, # Content loaded by load_prompt_content
    strategy_type: str, # "cot", "pal", "tora", "multi_algebra", "multi_python"
    use_train_prompt_format: bool = False # From original ToRA args, for specific model finetuning formats
) -> str:
    """
    Constructs the full prompt to be sent to the LLM.
    This adapts the logic from the original `construct_prompt` in ToRA's utils.py.

    Args:
        question_text: The actual question for the LLM to solve.
        few_shot_prompt_content: The few-shot examples loaded from a file.
        strategy_type: The type of strategy being used.
        use_train_prompt_format: Flag for specific model fine-tuning chat formats.

    Returns:
        The fully constructed prompt string.
    """
    if "PROMPT_NOT_FOUND" in few_shot_prompt_content: # Handle missing few-shot
        # Provide a basic zero-shot prompt if few-shot examples are missing
        if strategy_type == config.COT_STRATEGY_NAME:
            return f"Question: {question_text}\nAnswer: Let's think step by step."
        elif strategy_type == config.PAL_STRATEGY_NAME:
            return f"Let's use python to solve math problems.\n\nQuestion: {question_text}\n```python\ndef solution():\n    # Write your python code here\n    pass\n```"
        elif strategy_type == config.TORA_STRATEGY_NAME:
             return f"Integrate step-by-step reasoning and Python code to solve the math problem.\n\nQuestion: {question_text}\nSolution:\n"
        elif strategy_type == config.MULTI_STRATEGY_NAME + "_algebra": # Matching how multi prompts are loaded
            return f"Your goal is to write a mathematical equation...\ninput: {question_text}\nYOUR Response:\n" # Simplified zero-shot
        elif strategy_type == config.MULTI_STRATEGY_NAME + "_python":
            return f"Your goal is to write a Python function...\ninput: {question_text}\noutput:\n" # Simplified zero-shot
        else:
            return f"Question: {question_text}\nAnswer:"


    # Original ToRA logic for different prompt types after loading `demo_prompt`
    if use_train_prompt_format: # Typically for instruction-tuned models needing specific tags
        full_prompt = f"<|user|>\n{question_text}\n<|assistant|>\n"
        # Note: The original ToRA `construct_prompt` prepends `demo_prompt` in some cases even with `use_train_prompt_format`.
        # This might need careful checking based on the model. For now, assuming `use_train_prompt_format`
        # means the model expects only the direct question in its chat template.
        # If few-shot is still needed with this format, the logic would be:
        # full_prompt = f"<|user|>\n{few_shot_prompt_content}Question: {question_text}\n<|assistant|>\n"
    elif "tora" in strategy_type: # Covers TORA_STRATEGY_NAME
        context = f"Question: {question_text}\n\nSolution:"
        full_prompt = few_shot_prompt_content + context
    elif strategy_type == config.COT_STRATEGY_NAME: # "cot"
        context = f"Question: {question_text}\nAnswer:" # Original ToRA for cot
        full_prompt = few_shot_prompt_content + context
    elif strategy_type == config.PAL_STRATEGY_NAME: # "pal"
        # PAL prompts in ToRA have the question inside the few-shot content's last example usually.
        # The `few_shot_prompt_content` for PAL typically ends with something like:
        # ```python
        # def solution():
        # """{question_text}"""
        # ...
        # ```
        # So, we might just need to append the question to the loaded few-shot examples.
        # Or, the few-shot examples are generic and we append the current question.
        # Original ToRA PAL: demo_prompt + f"Question: {question_text}"
        # then the model is expected to generate the ```python ... ``` block.
        context = f"Question: {question_text}" # This seems to be what original ToRA PAL does
        full_prompt = few_shot_prompt_content + context
    elif strategy_type == config.MULTI_STRATEGY_NAME + "_algebra":
        # The {input} placeholder is already in multi_algebra_template.txt
        full_prompt = few_shot_prompt_content.replace("{input}", question_text)
    elif strategy_type == config.MULTI_STRATEGY_NAME + "_python":
        # The {input} placeholder is already in multi_python_template.txt
        full_prompt = few_shot_prompt_content.replace("{input}", question_text)
    # Original ToRA had wizard_zs and platypus_fs, which we mapped to "cot" or are not using.
    else:
        # Fallback or raise error for unhandled strategy types for prompt construction
        print(f"Warning: Prompt construction for strategy_type '{strategy_type}' not explicitly handled. Using generic fallback.")
        full_prompt = few_shot_prompt_content + f"Question: {question_text}\nAnswer:"

    return full_prompt


# key_map and show_sample are mostly for direct script running / debugging from ToRA.
# They might be less used in our programmatic flow but are harmless to keep.
key_map = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}

def show_sample(sample: Dict, print_all_preds: bool = False):
    print("=="*20)
    for key in ["idx", "type", "level", "dataset"]: # Common keys in math datasets
        if key in sample:
            # capitalize first letter for printing
            print_key = key[0].upper() + key[1:] if len(key) > 0 else key
            print(f"{print_key}: {sample[key]}")

    print(f"Question: {repr(sample.get('question', 'N/A'))}")

    # 'code' in ToRA samples often means the LLM's full generation
    # 'pred' is the extracted answer
    # 'report' is from python_executor

    if 'code' in sample: # This was from ToRA's output format
        if isinstance(sample['code'], list):
            if print_all_preds:
                for i, code_item in enumerate(sample['code']):
                    print('-'*20)
                    print(f"Generated Code/Solution (Sample {i+1}):\n{code_item}")
                    if 'report' in sample and isinstance(sample['report'], list) and i < len(sample['report']):
                        print(f"Execution Report (Sample {i+1}): {sample['report'][i]}")
            else: # Print only the first sample's details
                print(f"Generated Code/Solution (Sample 1):\n{sample['code'][0]}")
                if 'report' in sample and isinstance(sample['report'], list) and len(sample['report']) > 0:
                    print(f"Execution Report (Sample 1): {sample['report'][0]}")
        else: # If 'code' is not a list (e.g. single string)
             print(f"Generated Code/Solution:\n{sample['code']}")
             if 'report' in sample:
                print(f"Execution Report: {sample['report']}")


    if 'pred' in sample: # This was from ToRA's output format
        if isinstance(sample['pred'], list):
            if print_all_preds and len(sample['pred']) > 1:
                for i, pred_item in enumerate(sample['pred']):
                    print(f"Prediction (Sample {i+1}): {repr(pred_item)}")
            elif len(sample['pred']) > 0: # Print only the first prediction
                 print(f"Prediction (Sample 1): {repr(sample['pred'][0])}")
            else:
                print("Prediction: No predictions available.")
        else: # If 'pred' is not a list
            print(f"Prediction: {repr(sample['pred'])}")


    # Print other relevant ground truth and score fields
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            print_key  = key_map.get(key, key[0].upper() + key[1:] if len(key) > 0 else key)
            print(f"{print_key}: {repr(sample[key])}")
    print()