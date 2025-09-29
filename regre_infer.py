import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm

backbone_model_path = "/data2/qsh/model/Qwen2.5-1.5B-Instruct"
checkpoint_path = "/data2/qsh/project/mathagents/mspo/dynamic_Reasoning/checkpoints/final_model/pytorch_model.bin"
test_file = "/data2/qsh/project/sarma_math_solver/svamp_test.jsonl"
output_file = "/data2/qsh/project/sarma_math_solver/svamp_test_record.jsonl"


class RegressionHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size=4):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, hidden_states):
        return self.linear(hidden_states[:, -1, :])  

class StrategyScoreModel(torch.nn.Module):
    def __init__(self, backbone_path):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(backbone_path, trust_remote_code=True)
        hidden_size = self.backbone.config.hidden_size
        self.reg_head = RegressionHead(hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden = outputs.hidden_states[-1]
        logits = self.reg_head(last_hidden)
        scores = torch.softmax(logits, dim=-1)
        return scores


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(backbone_model_path, trust_remote_code=True)

model = StrategyScoreModel(backbone_model_path)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()


results = []
with open(test_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Scoring"):
        obj = json.loads(line)
        qid = obj.get("qid") or obj.get("idx")
        question = obj["question"]
        gold = obj.get("answer") or obj.get("gold_answer")

        prompt = f"""For the following math question, predict the normalized suitability scores for the listed strategies. The scores should correspond to the strategies in the order they are listed below.

Question: {question}

Available Strategies:
- cot: Chain-of-thought; solve step-by-step in natural language.
- pal: Program-aided; generate Python code to compute the answer.
- tora: Tool Reasoning (tora): Leverage structured tools or external reasoning components.
- multi: Hybrid Solver (multi): combine algebra and program-based approaches.

Predict scores for: cot, pal, tora, multi."""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            scores = model(**inputs).squeeze().tolist()

        results.append({
            "qid": qid,
            "question": question,
            "gold_answer": gold,
            "model_predicted_strategy_scores": {
                "cot": scores[0],
                "pal": scores[1],
                "tora": scores[2],
                "multi": scores[3]
            }
        })


with open(output_file, "w", encoding="utf-8") as f:
    for r in results:
        json.dump(r, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ 推理完成，已保存至: {output_file}")
