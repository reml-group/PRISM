# PRISM:PROBLEM-AWARE STRATEGY ROUTING FOR MATHEMATICAL REASONING WITH LLMS

This project is the official implementation of the paper“Plan Before Solving: Problem-Aware Strategy Routing for Mathematical Reasoning with LLMs” .

We propose the **PRISM (Planning and Routing through Instance Specific Modeling)** framework, which decouples the mathematical reasoning process into two stages: **policy planning** and **target execution**. The core of PRISM is a lightweight Strategy Adapter that can predict the applicability distribution of different inference strategies based on specific problems. During inference, the strategy dynamically selects the optimal execution path based on the predicted confidence level.

## Installation

```bash
conda create -n prism python=3.10
conda activate prism
pip install -r requirements.txt
```
## Usage

### Data Generation

The script will generate JSON file containing detailed performance metrics (correctness, process quality, efficiency) for each problem policy pair.

```
bash run_benchmark.sh
```

### Inference

Before running inference, please modify the following variables in the scripts `config.py` 

- `base_model`: The basic language model.
- `adapter_path`: The path of the trained strategy adapter model.
- `dataset_paths`:The dataset path.
- `output_dir`: The output directory of the experimental results.
- `tau_c` and `tau_a`.

```
bash run_parallel_experiment.sh
```

## Reference

If your research uses the PRISM framework or our dataset, please cite our paper.