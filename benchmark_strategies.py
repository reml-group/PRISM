# benchmark_strategies.py
import sys
import os
import json
import logging
import time
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import argparse
from pathlib import Path

import config
from llm_interface.llm_client import LLMClient
from tora_code_base.python_executor import PythonExecutor
from tora_code_base import utils as tora_utils
from tora_code_base import parser as tora_parser
from project_evaluation import grader

from strategies.cot_strategy import run_strategy_cot
from strategies.pal_strategy import run_strategy_pal
from strategies.tora_strategy import run_strategy_tora_batch_sampling
from strategies.multi_strategy import run_strategy_multi_batch_sampling

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


def get_strategy_function(strategy_name: str):
    if strategy_name == config.COT_STRATEGY_NAME: return run_strategy_cot
    if strategy_name == config.PAL_STRATEGY_NAME: return run_strategy_pal
    if strategy_name == config.TORA_STRATEGY_NAME: return run_strategy_tora_batch_sampling
    if strategy_name == config.MULTI_STRATEGY_NAME: return run_strategy_multi_batch_sampling
    raise ValueError(f"Unknown strategy: {strategy_name}")

def get_clean_gt_answer(question_data: Dict, qid_for_log: Any) -> str:
    if 'gt' in question_data and question_data['gt'] is not None: return str(question_data['gt']).strip()
    if 'answer' in question_data and '####' in str(question_data.get('answer', '')): return str(question_data['answer'].split('####')[-1]).strip()
    return "ERROR_GT_EXTRACTION"


def run_benchmark(args: argparse.Namespace):
    run_specific_base_dir = Path(config.LOG_DIR) / args.run_id
    worker_file_log_dir = run_specific_base_dir / "worker_stdout_logs"
    worker_file_log_dir.mkdir(parents=True, exist_ok=True)
    worker_specific_log_file = worker_file_log_dir / f"benchmark_worker_{args.worker_id + 1}_of_{args.num_workers}.log"

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    file_handler = logging.FileHandler(worker_specific_log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(f'%(asctime)s - (W{args.worker_id+1}/{args.num_workers}) - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    logger.info(f"Benchmark Start. Logging to: {worker_specific_log_file}")
    

    logger.warning(f"For this benchmark run, N_SAMPLING is being force-set to 1, ignoring the value in config.py.")
    config.N_SAMPLING = 1
    
    llm_client = LLMClient()
    pal_executor = PythonExecutor(get_answer_expr='solution()', timeout_length=config.PYTHON_EXEC_TIMEOUT)
    tora_executor = PythonExecutor(get_answer_from_stdout=True, timeout_length=config.PYTHON_EXEC_TIMEOUT)
    multi_executor = tora_executor 
    logger.info("LLMClient and PythonExecutors initialized.")
    
    all_questions_full = list(tora_utils.load_jsonl(config.TEST_DATA_PATH))
    questions_to_process = all_questions_full[:config.NUM_TEST_SAMPLES] if hasattr(config, 'NUM_TEST_SAMPLES') and config.NUM_TEST_SAMPLES > 0 else all_questions_full
    questions_per_worker = (len(questions_to_process) + args.num_workers - 1) // args.num_workers
    start_index = args.worker_id * questions_per_worker
    end_index = min(start_index + questions_per_worker, len(questions_to_process))
    worker_questions = questions_to_process[start_index:end_index] if start_index < len(questions_to_process) else []
    logger.info(f"This worker will process {len(worker_questions)} questions.")

    if not worker_questions: return

    all_benchmark_results = []
    strategies_to_benchmark = [config.COT_STRATEGY_NAME, config.PAL_STRATEGY_NAME, config.TORA_STRATEGY_NAME, config.MULTI_STRATEGY_NAME]
    
    for item in tqdm(worker_questions, desc=f"Benchmarking (W{args.worker_id+1})"):
        qid = item.get('qid') or item.get('idx')
        question_text = item.get('question', "MISSING_QUESTION_TEXT")
        gold_answer_str = get_clean_gt_answer(item, qid)

        question_log = {"qid": qid, "question_text": question_text, "gold_answer_str": gold_answer_str, "run_id": args.run_id, "benchmark_results": {}}

        for strategy_name in strategies_to_benchmark:
            logger.info(f"--- [QID: {qid}] Benchmarking strategy: {strategy_name} ---")
            
            start_time = time.time()
            strategy_func = get_strategy_function(strategy_name)
            
            executor = {'pal': pal_executor, 'tora': tora_executor, 'multi': multi_executor}.get(strategy_name)
            # ==========================================================
            
            try:
                outputs = strategy_func(item, llm_client, executor) if executor else strategy_func(item, llm_client)
                raw_output, extracted_answer = outputs[0]
                status = "error" if "ERROR" in extracted_answer else "success"
            except Exception as e:
                raw_output, extracted_answer, status = f"STRATEGY_EXCEPTION: {e}", "ERROR_STRATEGY_EXCEPTION", "error"

            execution_time = time.time() - start_time
            is_correct = grader.math_equal(extracted_answer, gold_answer_str, timeout=True)
            llm_calls = 2 if strategy_name == 'multi' else 1

            question_log["benchmark_results"][strategy_name] = {
                "status": status, "is_correct": is_correct, "execution_time_seconds": round(execution_time, 4),
                "llm_call_count": llm_calls, "output_char_length": len(raw_output),
                "final_answer": extracted_answer, "raw_output": raw_output
            }
        
        all_benchmark_results.append(question_log)

    save_path = run_specific_base_dir / f"benchmark_results_worker{args.worker_id+1}of{args.num_workers}.jsonl"
    tora_utils.save_jsonl(all_benchmark_results, save_path)
    logger.info(f"Worker {args.worker_id+1} finished. Benchmark results saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark different solving strategies.")
    parser.add_argument('--worker_id', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--run_id', type=str, required=True)
    cli_args = parser.parse_args()
    run_benchmark(cli_args)