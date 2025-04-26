import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel
import torch

# 1. Load raw DataFrame and extract only rows with METHODS
df = pd.read_csv("../data/processed/abstracts_train.csv")

# 2. Hugging Face dataset
dataset = Dataset.from_pandas(df, split="train")

# 3. Tokenizer for GPT-Neo-125M
MODEL_ID = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

# 4. Tokenize + mask prompt tokens
def preprocess(batch):
    # build full strings
    full_texts   = [
        f"### Problem:\n{p}\n### Approach:\n{a}"
        for p, a in zip(batch["problem_text"], batch["approach_text"])
    ]
    prompt_texts = [
        f"### Problem:\n{p}\n### Approach:\n"
        for p in batch["problem_text"]
    ]

    # tokenize both
    tokenized_full   = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized_prompt = tokenizer(
        prompt_texts,
        add_special_tokens=False
    )

    # create labels that ignore prompt portion
    labels = []
    for full_ids, prompt_ids in zip(tokenized_full["input_ids"], tokenized_prompt["input_ids"]):
        prompt_len = len(prompt_ids)
        lab = full_ids.copy()
        lab[:prompt_len] = [-100] * prompt_len
        labels.append(lab)

    tokenized_full["labels"] = labels
    return tokenized_full

tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=["problem_text", "approach_text"],
)

# 5. Load model (FP32) → resize embeddings → move to MPS if available
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.resize_token_embeddings(len(tokenizer))

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# 6. Apply LoRA (with GPT-Neo module names)
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# 7. Data collator & set PyTorch format
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 8. Training arguments (no mixed-precision on MPS)
training_args = TrainingArguments(
    output_dir="../models/gpt-neo-125M-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    max_steps=1000,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=False,
    bf16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

# — train! —
trainer.train()

