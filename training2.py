from unsloth import FastLanguageModel
import triton
max_seq_length = 2048 # Choose any! Llama 3 is up to 8k
dtype = None
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit",
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",
]
1159506bde84e795f0afc78a6cdebe7290278e8c
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", # Llama-3 70b also works (just change the model name)
    max_seq_length = 2048,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "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", # use one if using gated models like meta-llama/Llama-2-7b-hf
)