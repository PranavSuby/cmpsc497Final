import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate
from bert_score import score as bert_score
import pandas as pd
import time

# 1) Point these at your base model & LoRA checkpoint
BASE_MODEL_ID   = "EleutherAI/gpt-neo-125M"
CHECKPOINT_DIR  = "../models/gpt-neo-125M-finetuned/checkpoint-1000"

# 2) Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
# ensure pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

# 3) Load base model (FP32 on CPU) and then wrap with your LoRA adapters
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    device_map=None
)
model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)

# 4) Move to device (MPS on Apple Silicon, or CUDA/CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# — inference helper —
def generate_approach(
    problem: str,
    max_new_tokens: int = 200,
    num_beams: int = 5,
):
    prompt = f"### Problem:\n{problem}\n### Approach:\n"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,

        # ─── beam search parameters ───────────────────
        do_sample=False,            # turn off sampling
        num_beams=num_beams,        # beam width
        length_penalty=1.0,         # penalize short outputs
        no_repeat_ngram_size=3,     # block repeating 3-grams
        repetition_penalty=1.2,     # discourage token re-use
    )

    # strip off the prompt and return just the approach
    out = tokenizer.decode(
        generated_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    # optionally, truncate at the last full stop to avoid hanging fragments:
    if "." in out:
        out = out[: out.rfind(".") + 1]
    return out

def evaluate_model():
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bscore = evaluate.load("bertscore")

    test_df = pd.read_csv("../data/processed/abstracts_test.csv")
    val_problems = test_df["problem_text"].tolist()

    #Get the predictions
    predictions = [generate_approach(p) for p in val_problems].astype(str).tolist()
    references = test_df["approach_text"].astype(str).tolist()


    bleu_out = bleu.compute(predictions=predictions, references=references)
    rouge_out = rouge.compute(predictions=predictions, references=references)
    bert_out   = bscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="bert-base-uncased",
        rescale_with_baseline=True
    )
    prec = bert_out["precision"].mean().item()
    recall = bert_out["recall"].mean().item()
    f1 = bert_out["f1"].mean().item()

    print("BLEU:", bleu_out)
    print("ROUGE:", rouge_out)
    print(f"BERT Score: Precision - {prec}, Recall - {recall}, F1 - {f1}")

def run_model():
    test = input("Enter problem (q to quit): ")
    while test != "q":
        print("Approach:", generate_approach(test))
        test = input("Enter problem (q to quit): ")
    print("Exiting...")

if __name__ == "__main__":
    # Uncomment to evaluate the model
    # evaluate_model()
    # Uncomment to test model
    run_model()

