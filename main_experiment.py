# main_experiment.py
import sys
import os
import json
import logging
import datetime
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import numpy as np
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
    if strategy_name == config.COT_STRATEGY_NAME:
        return run_strategy_cot
    elif strategy_name == config.PAL_STRATEGY_NAME:
        return run_strategy_pal
    elif strategy_name == config.TORA_STRATEGY_NAME:
        return run_strategy_tora_batch_sampling
    elif strategy_name == config.MULTI_STRATEGY_NAME:
        return run_strategy_multi_batch_sampling
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")

def get_clean_gt_answer(question_data: Dict, qid_for_log: Any) -> str:
    if 'gt' in question_data and question_data['gt'] is not None:
        return str(question_data['gt']).strip()
    elif 'answer' in question_data and '####' in str(question_data.get('answer', '')):
        return str(question_data['answer'].split('####')[-1]).strip()
    logger.warning(f"QID {qid_for_log}: Could not extract ground truth answer.")
    return "ERROR_GT_EXTRACTION"

def run_experiment(args: argparse.Namespace):

    run_specific_base_dir = Path(config.LOG_DIR) / args.run_id
    worker_file_log_dir = run_specific_base_dir / "worker_stdout_logs"
    worker_file_log_dir.mkdir(parents=True, exist_ok=True)
    worker_specific_log_file = worker_file_log_dir / f"worker_{args.worker_id + 1}_of_{args.num_workers}.log"

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    file_handler = logging.FileHandler(worker_specific_log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(f'%(asctime)s - (W{args.worker_id+1}/{args.num_workers}) - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    logger.info(f"Experiment Start. Mode from config: '{config.EXECUTION_MODE}'. Logging to: {worker_specific_log_file}")
    logger.info(f"Full Run ID: {args.run_id}")

    llm_client = LLMClient()
    pal_executor = PythonExecutor(get_answer_expr='solution()', timeout_length=config.PYTHON_EXEC_TIMEOUT)
    tora_executor = PythonExecutor(get_answer_from_stdout=True, timeout_length=config.PYTHON_EXEC_TIMEOUT)
    multi_executor = tora_executor
    logger.info("LLMClient and PythonExecutors initialized.")


    all_questions_full = list(tora_utils.load_jsonl(config.TEST_DATA_PATH))
    logger.info(f"Loaded {len(all_questions_full)} total questions from the file.")


    questions_to_process = all_questions_full
    if hasattr(config, 'NUM_TEST_SAMPLES') and config.NUM_TEST_SAMPLES > 0:
        if len(all_questions_full) > config.NUM_TEST_SAMPLES:
            questions_to_process = all_questions_full[:config.NUM_TEST_SAMPLES]
            logger.info(f"Sub-sampling to {len(questions_to_process)} questions based on NUM_TEST_SAMPLES={config.NUM_TEST_SAMPLES}.")

    questions_per_worker = (len(questions_to_process) + args.num_workers - 1) // args.num_workers
    start_index = args.worker_id * questions_per_worker
    end_index = min(start_index + questions_per_worker, len(questions_to_process))
    worker_questions = questions_to_process[start_index:end_index] if start_index < len(questions_to_process) else []
    logger.info(f"This worker will process {len(worker_questions)} questions (from index {start_index} to {end_index-1}).")

    if not worker_questions:
        logger.info("No questions assigned to this worker. Exiting.")
        return


    scored_data_map = {}
    if config.EXECUTION_MODE == 'sarma':
        logger.info(f"SARMA mode enabled. Loading scored data from: {config.SCORED_DATA_PATH}")
        scored_data_list = list(tora_utils.load_jsonl(config.SCORED_DATA_PATH))
        scored_data_map = {item.get('qid', item.get('idx')): item for item in scored_data_list}
        logger.info(f"Loaded {len(scored_data_map)} scored entries.")

    experiment_results = []
    progress_bar_desc = f"GPU-{os.getenv('CUDA_VISIBLE_DEVICES', args.worker_id)} (W{args.worker_id+1}/{args.num_workers})"

    for item in tqdm(worker_questions, desc=progress_bar_desc, position=args.worker_id):
        qid = item.get('qid') or item.get('idx')
        question_text = item.get('question', "MISSING_QUESTION_TEXT")
        gold_answer_str = get_clean_gt_answer(item, qid)

        strategies_to_run = []
        chosen_path = ""
        sarma_log_info = {}

        if config.EXECUTION_MODE == 'sarma':
            scored_item = scored_data_map.get(qid)
            if not scored_item:
                logger.warning(f"[QID: {qid}] SARMA mode: No scoring info found. Skipping.")
                continue
            
            disc_scores = scored_item.get('model_predicted_strategy_scores', {})
            scores = sorted([(k, v) for k, v in disc_scores.items()], key=lambda x: x[1], reverse=True)
            
            pmax_strategy, pmax_score = scores[0] if scores else (None, -1.0)
            p2nd_strategy, p2nd_score = scores[1] if len(scores) > 1 else (None, -1.0)
            score_gap = pmax_score - p2nd_score if p2nd_strategy is not None else pmax_score

            if pmax_score >= config.TAU_C and score_gap >= config.TAU_A:
                chosen_path = 'sarma-single-path'
                strategies_to_run = [pmax_strategy]
            elif pmax_score >= config.TAU_C and score_gap < config.TAU_A:
                chosen_path = 'sarma-dual-path'
                strategies_to_run = [pmax_strategy, p2nd_strategy]
            else:
                chosen_path = 'sarma-exploration'
                strategies_to_run = ['cot', 'pal', 'tora', 'multi']
            
            sarma_log_info = {
                'discriminator_scores': disc_scores,
                'pmax_strategy_name': pmax_strategy, 'pmax_score': pmax_score,
                'p2nd_strategy_name': p2nd_strategy, 'p2nd_score': p2nd_score,
                'score_gap': score_gap,
            }
        else:
            chosen_path = f"single-strategy({config.EXECUTION_MODE})"
            strategies_to_run = [config.EXECUTION_MODE]


        strategy_results_per_sample = [{} for _ in range(config.N_SAMPLING)]
        for strat in strategies_to_run:
            strategy_func = get_strategy_function(strat)
            executor = None
            if strat == 'pal': executor = pal_executor
            elif strat == 'tora': executor = tora_executor
            elif strat == 'multi': executor = multi_executor

            try:
                outputs = strategy_func(item, llm_client, executor) if executor else strategy_func(item, llm_client)
                for i in range(config.N_SAMPLING):
                    strategy_results_per_sample[i][strat] = outputs[i]
            except Exception as e:
                logger.error(f"[QID: {qid}] Error running strategy {strat}: {e}", exc_info=True)
                for i in range(config.N_SAMPLING):
                    strategy_results_per_sample[i][strat] = (f"ERROR: {e}", "ERROR_STRATEGY_EXECUTION")

        pass_k_results = []
        for i in range(config.N_SAMPLING):
            sample_results = strategy_results_per_sample[i]
            final_answer, aggregation_detail = "ERROR_AGGREGATION", ""

            if config.EXECUTION_MODE != 'sarma':
                final_answer = sample_results[config.EXECUTION_MODE][1]
                aggregation_detail = f"single-strategy({config.EXECUTION_MODE})"
            else:
                if chosen_path == 'sarma-single-path':
                    final_answer = sample_results[strategies_to_run[0]][1]
                    aggregation_detail = f"sarma-single: chose {strategies_to_run[0]}"
                elif chosen_path == 'sarma-dual-path':
                    ans1, ans2 = sample_results[strategies_to_run[0]][1], sample_results[strategies_to_run[1]][1]
                    final_answer, aggregation_detail = (ans1, "sarma-dual: agreed") if ans1 == ans2 and not ans1.startswith("ERROR") else (ans1, f"sarma-dual: disagreed, chose pmax({strategies_to_run[0]})")
                else: # Exploration
                    valid_answers = [v[1] for v in sample_results.values() if not v[1].startswith("ERROR")]
                    if valid_answers:
                        counts = {ans: valid_answers.count(ans) for ans in set(valid_answers)}
                        max_count = max(counts.values())
                        candidates = [ans for ans, count in counts.items() if count == max_count]
                        final_answer, aggregation_detail = (candidates[0], f"sarma-explore: majority vote ({candidates[0]})") if len(candidates) == 1 else (sample_results.get(pmax_strategy, (None, "ERROR"))[1], f"sarma-explore: tie, fallback to pmax({pmax_strategy})")
                    else:
                        final_answer = sample_results.get(pmax_strategy, (None, "ERROR"))[1]
                        aggregation_detail = f"sarma-explore: no valid answers, fallback to pmax({pmax_strategy})"

            executed_details = [{'strategy_name': name, 'raw_llm_output': raw, 'extracted_answer': ans, 'is_correct_intermediate_answer': grader.math_equal(ans, gold_answer_str, timeout=True)} for name, (raw, ans) in sample_results.items()]
            pass_k_results.append({'sample_index': i, 'final_answer_produced_by_mechanism': final_answer, 'is_final_answer_correct': grader.math_equal(final_answer, gold_answer_str, timeout=True), 'aggregation_choice_detail': aggregation_detail, 'executed_strategies_details': executed_details})
        
        log_entry = {"qid": qid, "question_text": question_text, "gold_answer_str": gold_answer_str, "run_id": args.run_id, "mode": config.EXECUTION_MODE, "chosen_path": chosen_path, "strategies_actually_executed_in_path": strategies_to_run, "pass_k_results": pass_k_results, "pass_at_1_correct": pass_k_results[0]['is_final_answer_correct'] if pass_k_results else False, "pass_at_k_correct": any(res['is_final_answer_correct'] for res in pass_k_results), "n_sampling_configured": config.N_SAMPLING}
        if config.EXECUTION_MODE == 'sarma':
            log_entry.update(sarma_log_info)
        experiment_results.append(log_entry)


    save_path = run_specific_base_dir / f"experiment_results_worker{args.worker_id+1}of{args.num_workers}.jsonl"
    tora_utils.save_jsonl(experiment_results, save_path)
    logger.info(f"Worker {args.worker_id+1} finished. Results saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SARMA Math Solver Experiment Worker.")
    parser.add_argument('--worker_id', type=int, required=True, help='ID of this worker (0-indexed)')
    parser.add_argument('--num_workers', type=int, required=True, help='Total number of workers')
    parser.add_argument('--run_id', type=str, required=True, help='Unique ID for this experimental run')
    cli_args = parser.parse_args()
    run_experiment(cli_args)