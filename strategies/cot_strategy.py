# sarma_math_solver/strategies/cot_strategy.py

import logging
from typing import List, Tuple, Dict

# --- Project module imports ---
# These imports assume that main_experiment.py (or a similar entry point)
# has correctly set up sys.path so that 'sarma_math_solver' directory's
# contents are directly importable (e.g., 'import config') or sub-packages
# can be imported (e.g., 'from llm_interface.llm_client import LLMClient').

import config
from llm_interface.llm_client import LLMClient
from tora_code_base import utils as tora_utils
from tora_code_base import parser as tora_parser


logger = logging.getLogger(__name__)

def _construct_cot_prompt(question: str, dataset_name: str) -> str:
    """
    Constructs the CoT prompt using few-shot examples.
    """
    demo_prompt_text = tora_utils.load_prompt_content( # Uses the modified load_prompt_content
        dataset_name=dataset_name,
        prompt_strategy_type=config.COT_STRATEGY_NAME,
        base_prompt_dir=config.PROMPT_DIR
    )
    
    # Use the more robust construct_final_prompt from tora_utils
    full_prompt = tora_utils.construct_final_prompt(
        question_text=question,
        few_shot_prompt_content=demo_prompt_text,
        strategy_type=config.COT_STRATEGY_NAME,
        use_train_prompt_format=False # Assuming default, can be made configurable
    )
    return full_prompt

def run_strategy_cot(
    question_data: Dict, # 包含 'question', 'idx', 'qid' 等信息
    llm_client: LLMClient
) -> List[Tuple[str, str]]:
    """
    Runs the CoT strategy for a given question.

    Args:
        question_data: Dictionary containing question information.
        llm_client: Instance of LLMClient.

    Returns:
        A list of (raw_llm_output, extracted_answer) tuples,
        one for each of the N_SAMPLING attempts.
    """
    question_text = question_data['question']
    # dataset_name is now globally available from config
    # If you need to support multiple datasets in one run, question_data should carry its dataset_name
    dataset_name = config.DATASET_NAME 

    full_prompt = _construct_cot_prompt(question_text, dataset_name)
    current_qid = question_data.get('qid', question_data.get('idx', "N/A"))
    logger.debug(f"[COT Strategy] For qid {current_qid}, Prompt (first 100 chars):\n{full_prompt[:100]}...")


    stop_seqs = config.COMMON_STOP_SEQUENCES + config.COT_ADDITIONAL_STOP_SEQUENCES
    
    # LLMClient.generate 返回 List[List[str]], 外层对应prompt数量 (这里是1)
    # 内层对应 n_samples_per_prompt
    completions_for_prompt_list = llm_client.generate(
        prompts=full_prompt, # generate expects a list of prompts or a single prompt string
        n_samples_per_prompt=config.N_SAMPLING,
        temperature=config.SAMPLING_TEMPERATURE if config.N_SAMPLING > 1 else config.DEFAULT_TEMPERATURE,
        top_p=config.SAMPLING_TOP_P,
        max_tokens=config.DEFAULT_MAX_NEW_TOKENS,
        stop_sequences=list(set(stop_seqs)) # Ensure unique stop sequences
    )

    results = []
    # completions_for_prompt_list is List[List[str]], for a single input prompt, it's [[comp1, comp2, ...]]
    if completions_for_prompt_list and completions_for_prompt_list[0]:
        for raw_llm_output in completions_for_prompt_list[0]:
            if "LLM_GENERATION_ERROR" in raw_llm_output:
                extracted_answer = "ERROR_LLM_GENERATION"
            else:
                # ToRA项目的 parser.extract_answer 用于从CoT输出中提取答案
                extracted_answer = tora_parser.extract_answer(raw_llm_output)
            results.append((raw_llm_output, extracted_answer))
            # logger.debug(f"  Raw CoT Output: {raw_llm_output[:200]}... Extracted: {extracted_answer}") # Can be very verbose
    else:
        error_msg = f"LLM did not return any completions for CoT (QID: {current_qid})."
        logger.error(error_msg)
        for _ in range(config.N_SAMPLING):
            results.append((error_msg, "ERROR_NO_COMPLETION"))
            
    return results

if __name__ == '__main__':
    # This section is for testing cot_strategy.py independently.
    # It requires careful setup of paths and mock objects.
    
    print("Testing CoT Strategy Module (Independent Run)...")
    
    # --- Mocking and Configuration for Standalone Test ---
    # Temporarily modify sys.path to allow finding sibling modules if run directly
    # This assumes the script is in sarma_math_solver/strategies/
    if not hasattr(sys, 'testing_setup_done_cot'): # Prevent re-adding paths if re-run in same session
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(module_dir) # This should be 'sarma_math_solver'
        project_parent = os.path.dirname(project_root) # Parent of 'sarma_math_solver'
        
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        if project_parent not in sys.path: # To allow 'from sarma_math_solver import ...'
             sys.path.insert(0, project_parent)
        sys.testing_setup_done_cot = True


    # Now, re-attempt imports with the adjusted path
    # These imports might still fail if the structure is not as expected or if there are circular deps
    # This is why `python -m <module>` is generally preferred for running package components.
    try:
        from sarma_math_solver import config as test_config # Absolute import assuming parent is in path
        from sarma_math_solver.llm_interface.llm_client import LLMClient as TestLLMClient
        from sarma_math_solver.tora_code_base import utils as test_tora_utils
        from sarma_math_solver.tora_code_base import parser as test_tora_parser
    except ModuleNotFoundError as e:
        print(f"Failed to import modules for testing: {e}")
        print("Please ensure you run tests from the project root or have PYTHONPATH set correctly.")
        print("Example: `python -m sarma_math_solver.strategies.cot_strategy` from parent of sarma_math_solver")
        exit()


    # Override specific config values for this test if needed
    # For a real test, you'd point PROMPT_DIR to where your test prompts are.
    # For now, we'll mock `load_prompt_content` and `construct_final_prompt`.
    test_config.PROMPT_DIR = os.path.join(project_root, "prompts") # Correct path to prompts
    test_config.DATASET_NAME = "gsm8k"
    test_config.N_SAMPLING = 2
    test_config.SAMPLING_TEMPERATURE = 0.0 # Make mock predictable
    test_config.DEFAULT_TEMPERATURE = 0.0
    test_config.MODEL_NAME_OR_PATH = "JackFram/llama-68m" # Small model for testing client init
    test_config.COT_STRATEGY_NAME = "cot"
    # Ensure COMMON_STOP_SEQUENCES and COT_ADDITIONAL_STOP_SEQUENCES are in test_config or mocked
    if not hasattr(test_config, 'COMMON_STOP_SEQUENCES'): test_config.COMMON_STOP_SEQUENCES = ["</s>"]
    if not hasattr(test_config, 'COT_ADDITIONAL_STOP_SEQUENCES'): test_config.COT_ADDITIONAL_STOP_SEQUENCES = ["\n\nQuestion:"]


    # --- Mock LLMClient ---
    class MockLLMClientForCoT:
        def __init__(self):
            print(f"MockLLMClientForCoT initialized for model {test_config.MODEL_NAME_OR_PATH}")
            self.generate_call_count = 0

        def generate(self, prompts, n_samples_per_prompt, temperature, top_p, max_tokens, stop_sequences):
            self.generate_call_count += 1
            print(f"\n--- MockLLMClientForCoT.generate (Call #{self.generate_call_count}) ---")
            prompt_text = prompts[0] if isinstance(prompts, list) else prompts
            print(f"Received Prompt (first 200 chars): {prompt_text[:200]}...")
            print(f"n_samples_per_prompt: {n_samples_per_prompt}, temperature: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}")
            print(f"Stop sequences: {stop_sequences}")
            
            mock_outputs = []
            for i in range(n_samples_per_prompt):
                # Simulate a CoT output structure
                mock_outputs.append(f"This is mock sample {i+1} for CoT. Let's think: 48 / 2 = 24. Then 48 + 24 = 72. The final answer is \\boxed{{{72+i}}}.")
            return [mock_outputs] # Return structure: List[List[str]]

    # --- Mock tora_utils.load_prompt_content if needed (or ensure prompts are found) ---
    # For this test, let's assume PROMPT_DIR is correctly set and files exist,
    # or mock the prompt construction part directly.
    original_load_prompt = test_tora_utils.load_prompt_content
    original_construct_prompt = test_tora_utils.construct_final_prompt

    def mock_load_prompt_content(dataset_name, prompt_strategy_type, base_prompt_dir):
        print(f"Mocked load_prompt_content for: {dataset_name}, {prompt_strategy_type} from {base_prompt_dir}")
        # Ensure the mock returns something that construct_final_prompt expects (ends with \n\n)
        return f"FEW SHOT EXAMPLE 1...\nFEW SHOT EXAMPLE 2...\n\n"

    test_tora_utils.load_prompt_content = mock_load_prompt_content
    # _construct_cot_prompt uses construct_final_prompt, which uses load_prompt_content.
    # So mocking load_prompt_content is sufficient here.

    # --- Test Execution ---
    test_llm_client_instance = MockLLMClientForCoT()
    
    test_question_item = {
        'idx': 0,
        'qid': 0, 
        'question': "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        'answer': "Natalia sold 48/2 = 24 clips in May.\nIn total, she sold 48 + 24 = 72 clips.\n#### 72",
        'gt': "72"
    }

    print(f"\n--- Running run_strategy_cot with Mocks (Dataset: {test_config.DATASET_NAME}) ---")
    
    try:
        # Replace global config with test_config for the duration of the test function call
        # This is a bit hacky for testing module-level functions that rely on global config.
        original_config_module_cot = config 
        # sys.modules['sarma_math_solver.config'] = test_config # This can be risky
        # A better way is to pass config explicitly or use dependency injection for config.
        # For now, since cot_strategy.py imports 'from .. import config', it gets the real one.
        # To make the test use MockConfig, cot_strategy's import would need to be dynamic or config passed.
        # The `if __name__ == '__main__':` block in cot_strategy.py shows `config = MockConfig()`
        # which is good for when *that file itself* is run.
        # If we are calling it from here (another file), we'd need to ensure it sees the right config.
        
        # The `cot_strategy.py` standalone test block creates its own MockConfig.
        # So, if we were to truly test it from *another* file, we'd need to ensure
        # the imported `config` in `cot_strategy.py` is the one we want (e.g. by patching).

        # Let's assume `cot_strategy.py`'s own `if __name__ == '__main__':`
        # is the primary way to test it standalone with its mocks.
        # Here, we demonstrate calling it as if from `main_experiment.py`.
        # So, `cot_strategy.py` must be able to import the *actual* `config.py`
        # and `llm_client.py`. The mocks here are for *this test script's call*.

        # To test the actual `run_strategy_cot` function:
        # 1. Ensure `config.py` points to a testable (small) model if not using a mock LLM client.
        # 2. Create a real LLMClient instance if not mocking.
        
        # Using a mock LLM Client for this test:
        results = run_strategy_cot(test_question_item, test_llm_client_instance)
        
        print(f"\nCoT Strategy Test Results for QID {test_question_item.get('qid')}:")
        assert len(results) == test_config.N_SAMPLING
        for i, (raw, extracted) in enumerate(results):
            print(f"  Sample {i+1}:")
            print(f"    Raw (first 100 chars): {raw[:100]}...")
            print(f"    Extracted Answer: {extracted}")
            # Example assertion based on MockLLMClientForCoT output
            assert extracted == str(72+i) 
        print("\nCoT Strategy standalone test section completed successfully with mocks.")

    except Exception as e:
        print(f"Error during CoT strategy standalone test section: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original tora_utils functions if they were patched
        test_tora_utils.load_prompt_content = original_load_prompt
        # test_tora_utils.construct_final_prompt = original_construct_prompt (if construct_final_prompt was also mocked)