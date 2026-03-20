import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = "model"          # Where your Kaggle files are
export_path = "./merged_rag_model" # Where the new, fast model will go

print("1. Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.float16,
    device_map="cpu", # Keep on CPU just for the merge process
)

# Remember our vocabulary fix from earlier!
base_model.resize_token_embeddings(128272)

print("2. Attaching LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("3. Fusing weights together (this might take a few minutes)...")
# This is the magic command that bakes the LoRA permanently into the base model
merged_model = model.merge_and_unload()

print(f"4. Saving standalone model to {export_path}...")
merged_model.save_pretrained(export_path)
tokenizer.save_pretrained(export_path)

print("✅ Merge complete! You now have a standalone, fast model.")