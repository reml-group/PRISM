# sarma_math_solver/llm_interface/llm_client.py
import logging
from typing import List, Union, Optional
from vllm import LLM, SamplingParams
import os
# 导入配置
import config # 使用相对导入，假设项目根目录在PYTHONPATH中，或者运行时从根目录运行

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        if config.LLM_PROVIDER.lower() == "vllm":
            try:
                # 打印关键的vLLM初始化参数
                logger.info(f"Initializing vLLM with:")
                logger.info(f"  Model: {config.MODEL_NAME_OR_PATH}")
                logger.info(f"  Tokenizer: {config.TOKENIZER_PATH}")
                logger.info(f"  Tensor Parallel Size: {config.VLLM_TENSOR_PARALLEL_SIZE}")
                logger.info(f"  GPU Memory Utilization: {config.VLLM_GPU_MEMORY_UTILIZATION}")
                logger.info(f"  Trust Remote Code: {config.VLLM_TRUST_REMOTE_CODE}")
                
                # CUDA_VISIBLE_DEVICES 通常在脚本启动前通过环境变量设置。
                # vLLM 会自动检测并使用 tensor_parallel_size 指定数量的可见GPU。
                # 如果需要在此处显式设置，则需 os.environ['CUDA_VISIBLE_DEVICES'] = config.VISIBLE_GPU_IDS
                # 但这通常不是vLLM库的推荐做法，它倾向于依赖外部环境变量。
                # 检查一下环境变量是否与配置匹配
                visible_devices_env = os.environ.get('CUDA_VISIBLE_DEVICES')
                if visible_devices_env:
                    num_visible_from_env = len(visible_devices_env.split(','))
                    if config.VLLM_TENSOR_PARALLEL_SIZE > num_visible_from_env:
                        logger.warning(
                            f"VLLM_TENSOR_PARALLEL_SIZE ({config.VLLM_TENSOR_PARALLEL_SIZE}) "
                            f"is greater than the number of GPUs made visible by CUDA_VISIBLE_DEVICES ({num_visible_from_env}). "
                            f"vLLM might fail or use fewer GPUs."
                        )
                elif config.VLLM_TENSOR_PARALLEL_SIZE > 1:
                     logger.warning(
                        f"CUDA_VISIBLE_DEVICES is not set, but VLLM_TENSOR_PARALLEL_SIZE is {config.VLLM_TENSOR_PARALLEL_SIZE}. "
                        f"vLLM will try to use all available GPUs up to this size."
                     )


                self.llm = LLM(
                    model=config.MODEL_NAME_OR_PATH,
                    tokenizer=config.TOKENIZER_PATH,
                    tensor_parallel_size=config.VLLM_TENSOR_PARALLEL_SIZE,
                    gpu_memory_utilization=config.VLLM_GPU_MEMORY_UTILIZATION, # 新增参数
                    trust_remote_code=config.VLLM_TRUST_REMOTE_CODE,
                    dtype='half', # 可以根据模型和硬件考虑添加，例如 'auto' 或 'half'
                    # max_model_len=xxxx, # 如果模型上下文长度需要调整
                )
                logger.info("vLLM initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize vLLM: {e}")
                logger.error("Please check your vLLM installation, CUDA setup, model path, and GPU resources.")
                import traceback
                logger.error(traceback.format_exc())
                raise
        else:
            raise NotImplementedError(f"LLM provider '{config.LLM_PROVIDER}' is not supported yet.")

    def generate(
        self,
        prompts: Union[str, List[str]],
        n_samples_per_prompt: int = 1,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> List[List[str]]:
        # ... (generate 方法的其余部分保持不变) ...
        if isinstance(prompts, str):
            prompts = [prompts]

        current_temperature = temperature if temperature is not None else \
                              (config.SAMPLING_TEMPERATURE if n_samples_per_prompt > 1 and config.N_SAMPLING > 1 else config.DEFAULT_TEMPERATURE) # 确保N_SAMPLING也考虑进来
        
        current_top_p = top_p if top_p is not None else config.SAMPLING_TOP_P
        if current_temperature == 0.0: 
            current_top_p = 1.0

        current_max_tokens = max_tokens if max_tokens is not None else config.DEFAULT_MAX_NEW_TOKENS
        
        # 确保停止序列是唯一的列表
        current_stop_sequences = []
        if stop_sequences:
            current_stop_sequences.extend(stop_sequences)
        if config.COMMON_STOP_SEQUENCES:
            current_stop_sequences.extend(config.COMMON_STOP_SEQUENCES)
        current_stop_sequences = sorted(list(set(current_stop_sequences))) # 去重并排序（可选排序）


        sampling_params = SamplingParams(
            n=n_samples_per_prompt,
            temperature=current_temperature,
            top_p=current_top_p,
            max_tokens=current_max_tokens,
            stop=current_stop_sequences if current_stop_sequences else None, # None if empty
        )
        
        logger.debug(f"Generating with params: n={n_samples_per_prompt}, temp={current_temperature}, top_p={current_top_p}, max_tokens={current_max_tokens}, stop={current_stop_sequences}")

        try:
            request_outputs = self.llm.generate(prompts, sampling_params)
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            error_output = [f"LLM_GENERATION_ERROR: {e}"] * n_samples_per_prompt
            return [error_output for _ in prompts]

        all_prompt_completions = []
        for req_output in request_outputs:
            prompt_completions = [completion.text for completion in req_output.outputs]
            all_prompt_completions.append(prompt_completions)
            # logger.debug(f"Prompt: '{req_output.prompt[:100]}...' -> Generated {len(prompt_completions)} samples.")
            # for i, comp_text in enumerate(prompt_completions):
            #      logger.debug(f"  Sample {i}: '{comp_text[:100]}...'") # 可能过于冗长
        return all_prompt_completions

# 可以添加一个简单的测试函数
if __name__ == '__main__':
    # 注意：直接运行此文件可能因为相对导入 `from .. import config` 而失败
    # 需要从项目根目录通过 python -m sarma_math_solver.llm_interface.llm_client 来运行
    # 或者在测试时临时修改导入方式和config的加载
    print("LLMClient basic structure. Run from project root for proper imports or test with direct config.")
    
    # 临时的配置加载方式，仅用于独立测试此文件
    class TempConfig:
        LLM_PROVIDER = "vllm"
        MODEL_NAME_OR_PATH = "JackFram/llama-68m" # 使用一个非常小的模型测试vLLM流程
        TOKENIZER_PATH = "JackFram/llama-68m"
        VLLM_TENSOR_PARALLEL_SIZE = 1
        VLLM_TRUST_REMOTE_CODE = True # 有些模型需要
        N_SAMPLING = 2
        SAMPLING_TEMPERATURE = 0.7
        DEFAULT_TEMPERATURE = 0.0
        SAMPLING_TOP_P = 1.0
        DEFAULT_MAX_NEW_TOKENS = 50
        COMMON_STOP_SEQUENCES = ["</s>"]

    config = TempConfig() # 覆盖导入的config

    try:
        client = LLMClient()
        test_prompts = ["Translate the following English text to French: 'Hello, world!'", "Write a short poem about a cat."]
        
        # 测试单个prompt, 多个采样
        results_single_prompt = client.generate(
            test_prompts[0], 
            n_samples_per_prompt=config.N_SAMPLING, 
            temperature=config.SAMPLING_TEMPERATURE,
            max_tokens=30
            )
        print("\n--- Results for single prompt, multiple samples ---")
        for i, sample_group in enumerate(results_single_prompt):
            print(f"Prompt {i}:")
            for j, completion in enumerate(sample_group):
                print(f"  Sample {j}: {completion}")

        # 测试多个prompt, 每个prompt多个采样
        results_batch_prompts = client.generate(
            test_prompts, 
            n_samples_per_prompt=config.N_SAMPLING,
            temperature=config.SAMPLING_TEMPERATURE,
            max_tokens=30
            )
        print("\n--- Results for batch prompts, multiple samples per prompt ---")
        for i, sample_group in enumerate(results_batch_prompts):
            print(f"Prompt {i} ('{test_prompts[i][:30]}...'):")
            for j, completion in enumerate(sample_group):
                print(f"  Sample {j}: {completion}")
                
    except Exception as e:
        print(f"Error in LLMClient test: {e}")
        print("Please ensure vLLM is installed and a valid model path is provided in TempConfig for testing.")
        print("You might need to run 'pip install vllm sympy torch transformers datasets tqdm pebble multiprocess timeout-decorator python-dateutil'")