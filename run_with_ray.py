# run_with_ray.py

import ray
import time
import argparse
import os
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

import config
from llm_interface.llm_client import LLMClient
from tora_code_base.python_executor import PythonExecutor
from tora_code_base import utils as tora_utils
from project_evaluation import grader
from strategies.cot_strategy import run_strategy_cot
from strategies.pal_strategy import run_strategy_pal
from strategies.tora_strategy import run_strategy_tora_batch_sampling
from strategies.multi_strategy import run_strategy_multi_batch_sampling


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

def get_clean_gt_answer(question_data: dict, qid_for_log: any) -> str:
    if 'gt' in question_data and question_data['gt'] is not None:
        return str(question_data['gt']).strip()
    elif 'answer' in question_data and '####' in str(question_data.get('answer', '')):
        return str(question_data['answer'].split('####')[-1]).strip()
    return "ERROR_GT_EXTRACTION"


@ray.remote(num_gpus=1)
class ExperimentWorker:
    def __init__(self):
        actor_id = ray.get_runtime_context().get_actor_id()
        gpu_ids = ray.get_gpu_ids()
        print(f"Initializing Actor {actor_id} on GPU(s): {gpu_ids}...")
        
        self.llm_client = LLMClient()
        self.pal_executor = PythonExecutor(get_answer_expr='solution()', timeout_length=config.PYTHON_EXEC_TIMEOUT)
        self.tora_executor = PythonExecutor(get_answer_from_stdout=True, timeout_length=config.PYTHON_EXEC_TIMEOUT)
        self.multi_executor = self.tora_executor
        self.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)
        print(f"✅ Actor {actor_id} on GPU(s): {gpu_ids} initialized successfully.")

    def process_item(self, item: dict, scored_item: dict, args: argparse.Namespace) -> dict:
        qid = item.get('qid') or item.get('idx')
        question_text = item.get('question', "MISSING_QUESTION_TEXT")
        gold_answer_str = get_clean_gt_answer(item, qid)

        strategies_to_run = []
        chosen_path = ""
        sarma_log_info = {}

        if args.execution_mode == 'sarma':
            if not scored_item:
                
                strategies_to_run = ['cot', 'pal', 'tora', 'multi'] 
                chosen_path = 'sarma-exploration (fallback due to missing score)'
            else:
                disc_scores = scored_item.get('model_predicted_strategy_scores', {})
                scores = sorted([(k, v) for k, v in disc_scores.items()], key=lambda x: x[1], reverse=True)
                
                pmax_strategy, pmax_score = scores[0] if scores else (None, -1.0)
                p2nd_strategy, p2nd_score = scores[1] if len(scores) > 1 else (None, -1.0)
                score_gap = pmax_score - p2nd_score if p2nd_strategy is not None else pmax_score

                if args.ablation_mode == 'single_only':
                    chosen_path = 'ablation-single-only'
                    strategies_to_run = [pmax_strategy]
                elif args.ablation_mode == 'single_or_dual':
                    if score_gap >= config.TAU_A:
                        chosen_path = 'ablation-single-or-dual (chose single)'
                        strategies_to_run = [pmax_strategy]
                    else:
                        chosen_path = 'ablation-single-or-dual (chose dual)'
                        strategies_to_run = [pmax_strategy, p2nd_strategy]
                elif args.ablation_mode == 'single_or_explo':
                    if pmax_score >= config.TAU_C:
                        chosen_path = 'ablation-single-or-explo (chose single)'
                        strategies_to_run = [pmax_strategy]
                    else:
                        chosen_path = 'ablation-single-or-explo (chose explo)'
                        strategies_to_run = ['cot', 'pal', 'tora', 'multi']
                else: # Full SARMA (none)
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
                    'discriminator_scores': disc_scores, 'pmax_strategy_name': pmax_strategy, 'pmax_score': pmax_score,
                    'p2nd_strategy_name': p2nd_strategy, 'p2nd_score': p2nd_score, 'score_gap': score_gap,
                }
        else:
            chosen_path = f"single-strategy({args.execution_mode})"
            strategies_to_run = [args.execution_mode]

        strategy_results_per_sample = [{} for _ in range(config.N_SAMPLING)]
        total_execution_time = 0.0

        for strat in strategies_to_run:
            if strat is None: continue
            strategy_func = get_strategy_function(strat)
            executor = {'pal': self.pal_executor, 'tora': self.tora_executor, 'multi': self.multi_executor}.get(strat)

            try:
                start_time = time.monotonic()
                outputs = strategy_func(item, self.llm_client, executor) if executor else strategy_func(item, self.llm_client)
                end_time = time.monotonic()
                total_execution_time += (end_time - start_time)

                for i in range(config.N_SAMPLING):
                    strategy_results_per_sample[i][strat] = {"raw_llm_output": outputs[i][0], "extracted_answer": outputs[i][1]}
            except Exception as e:
                for i in range(config.N_SAMPLING):
                    strategy_results_per_sample[i][strat] = {"raw_llm_output": f"ERROR: {e}", "extracted_answer": "ERROR_STRATEGY_EXECUTION"}

        pass_k_results = []
        total_tokens_per_sample = []

        for i in range(config.N_SAMPLING):
            sample_results = strategy_results_per_sample[i]
            final_answer, aggregation_detail = "ERROR_AGGREGATION", ""

            if args.execution_mode != 'sarma':
                final_answer = sample_results[args.execution_mode]["extracted_answer"]
                aggregation_detail = f"single-strategy({args.execution_mode})"
            else:
                pmax_strategy = sarma_log_info.get('pmax_strategy_name')
                if 'single' in chosen_path:
                    final_answer = sample_results.get(strategies_to_run[0], {"extracted_answer": "ERROR_SINGLE_PATH_FAIL"})["extracted_answer"]
                    aggregation_detail = f"path: {chosen_path}, chose {strategies_to_run[0]}"
                elif 'dual' in chosen_path:
                    ans1 = sample_results.get(strategies_to_run[0], {"extracted_answer": "ERROR_DUAL_PATH_PMAX_FAIL"})["extracted_answer"]
                    ans2 = sample_results.get(strategies_to_run[1], {"extracted_answer": "ERROR_DUAL_PATH_P2ND_FAIL"})["extracted_answer"]
                    final_answer, aggregation_detail = (ans1, f"path: {chosen_path}, agreed") if ans1 == ans2 and not ans1.startswith("ERROR") else (ans1, f"path: {chosen_path}, disagreed, chose pmax({strategies_to_run[0]})")
                else: # Exploration
                    valid_answers = [v["extracted_answer"] for v in sample_results.values() if not v["extracted_answer"].startswith("ERROR")]
                    if valid_answers:
                        counts = {ans: valid_answers.count(ans) for ans in set(valid_answers)}
                        max_count = max(counts.values())
                        candidates = [ans for ans, count in counts.items() if count == max_count]
                        fallback_ans = sample_results.get(pmax_strategy, {"extracted_answer": "ERROR"})["extracted_answer"]
                        final_answer, aggregation_detail = (candidates[0], f"path: {chosen_path}, majority vote ({candidates[0]})") if len(candidates) == 1 else (fallback_ans, f"path: {chosen_path}, tie, fallback to pmax({pmax_strategy})")
                    else:
                        final_answer = sample_results.get(pmax_strategy, {"extracted_answer": "ERROR"})["extracted_answer"]
                        aggregation_detail = f"path: {chosen_path}, no valid answers, fallback to pmax({pmax_strategy})"

            executed_details_with_tokens = []
            current_sample_total_tokens = 0
            for name, res in sample_results.items():
                raw_output = res['raw_llm_output']
                num_tokens = len(self.tokenizer.encode(raw_output))
                current_sample_total_tokens += num_tokens
                executed_details_with_tokens.append({
                    'strategy_name': name, 'raw_llm_output': raw_output, 'extracted_answer': res['extracted_answer'],
                    'output_num_tokens': num_tokens, 'is_correct_intermediate_answer': grader.math_equal(res['extracted_answer'], gold_answer_str, timeout=True)
                })
            
            total_tokens_per_sample.append(current_sample_total_tokens)
            pass_k_results.append({'sample_index': i, 'final_answer_produced_by_mechanism': final_answer, 'is_final_answer_correct': grader.math_equal(final_answer, gold_answer_str, timeout=True), 'aggregation_choice_detail': aggregation_detail, 'executed_strategies_details': executed_details_with_tokens})
        
        avg_generated_tokens = np.mean(total_tokens_per_sample) if total_tokens_per_sample else 0
        
        log_entry = {
            "qid": qid, "question_text": question_text, "gold_answer_str": gold_answer_str, "run_id": args.run_id,
            "mode": args.execution_mode, "ablation_mode": args.ablation_mode, "chosen_path": chosen_path,
            "total_execution_time": total_execution_time, "average_generated_tokens": avg_generated_tokens,
            "strategies_actually_executed_in_path": strategies_to_run, "pass_k_results": pass_k_results,
            "pass_at_1_correct": pass_k_results[0]['is_final_answer_correct'] if pass_k_results else False,
            "pass_at_k_correct": any(res['is_final_answer_correct'] for res in pass_k_results), "n_sampling_configured": config.N_SAMPLING
        }
        if args.execution_mode == 'sarma':
            log_entry.update(sarma_log_info)

        return log_entry
        # =========================================================================
        # --- 结束: 完整复制 main_benchmark_ablation.py 的 for 循环体内部逻辑 ---
        # =========================================================================

# --- 4. 定义主函数 (我们的总指挥) ---
def main(args: argparse.Namespace):
    start_time = time.time()
    
    # 初始化Ray。Ray会自动发现 `CUDA_VISIBLE_DEVICES` 中指定的GPU。
    ray.init()
    
    # 准备日志和输出目录
    run_specific_base_dir = Path(config.LOG_DIR) / args.run_id
    run_specific_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"========================================================================")
    print(f"Starting Ray-based Experiment Run")
    print(f"========================================================================")
    print(f"Run ID: {args.run_id}")
    print(f"Execution Mode: {args.execution_mode}")
    print(f"Ablation Mode: {args.ablation_mode}")
    print(f"Test Data: {args.test_data_path}")
    print(f"Scored Data: {args.scored_data_path}")
    print(f"Logs and results will be saved in: {run_specific_base_dir}")
    print(f"------------------------------------------------------------------------")

    all_questions_full = list(tora_utils.load_jsonl(args.test_data_path))
    questions_to_process = all_questions_full
    if hasattr(config, 'NUM_TEST_SAMPLES') and config.NUM_TEST_SAMPLES > 0:
        if len(all_questions_full) > config.NUM_TEST_SAMPLES:
            questions_to_process = all_questions_full[:config.NUM_TEST_SAMPLES]
    print(f"Loaded {len(questions_to_process)} questions to process.")
    
    scored_data_map = {}
    if args.execution_mode == 'sarma':
        print(f"Loading scored data...")
        scored_data_list = list(tora_utils.load_jsonl(args.scored_data_path))
        scored_data_map = {item.get('qid', item.get('idx')): item for item in scored_data_list}
        print(f"Loaded scores for {len(scored_data_map)} questions.")

 
    num_gpus = int(ray.cluster_resources().get("GPU", 0))
    if num_gpus == 0:
        print("\nERROR: No GPUs found in the Ray cluster. Check `CUDA_VISIBLE_DEVICES`.")
        return
    print(f"Found {num_gpus} GPUs. Creating {num_gpus} ExperimentWorker actors...")
    workers = [ExperimentWorker.remote() for _ in range(num_gpus)]
    

    results_refs = []
    print("Submitting tasks to workers...")
    for i, item in enumerate(questions_to_process):
        qid = item.get('qid') or item.get('idx')
        scored_item = scored_data_map.get(qid)
        worker = workers[i % num_gpus]
        results_refs.append(worker.process_item.remote(item, scored_item, args))

    experiment_results = []
    for ref in tqdm(results_refs, desc="Collecting results", total=len(questions_to_process)):
        try:
            result = ray.get(ref)
            experiment_results.append(result)
        except ray.exceptions.RayTaskError as e:
            print(f"\nWARNING: A task failed with error: {e}")


    save_path = run_specific_base_dir / "consolidated_results.jsonl"
    tora_utils.save_jsonl(experiment_results, save_path)
    
    end_time = time.time()
    print(f"------------------------------------------------------------------------")
    print(f"✅ All tasks finished in {end_time - start_time:.2f} seconds.")
    print(f"Consolidated results for {len(experiment_results)} items saved to: {save_path}")
    print(f"========================================================================")
    

    ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified Ray-based Runner for SARMA Math Solver.")
    

    parser.add_argument(
        '--run_id', type=str, 
        default=f"ray_run_{time.strftime('%Y%m%d_%H%M%S')}", 
        help='Unique ID for this run. Defaults to a timestamped name.'
    )
    parser.add_argument(
        '--test_data_path', type=str, required=True, 
        help='Path to the test dataset file (e.g., math500_idx.jsonl).'
    )
    parser.add_argument(
        '--scored_data_path', type=str, required=True, 
        help='Path to the corresponding scored data file (e.g., math500_record.jsonl).'
    )
    parser.add_argument(
        '--execution_mode', type=str, default='sarma', 
        choices=['sarma', 'cot', 'pal', 'tora', 'multi'], 
        help="Set the execution mode. 'sarma' enables the dynamic router."
    )
    parser.add_argument(
        '--ablation_mode', type=str, default='none', 
        choices=['none', 'single_only', 'single_or_dual', 'single_or_explo'], 
        help="Set the ablation study mode (only active when execution_mode is 'sarma')."
    )
    
    cli_args = parser.parse_args()
    main(cli_args)