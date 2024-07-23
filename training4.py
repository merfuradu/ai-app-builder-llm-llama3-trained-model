from datasets import load_dataset, DatasetDict
from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login
import torch
import os
from transformers import FalconForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
# Load dataset
dataset = load_dataset("merfuradu/appbuilder", split='train')

# Login to Hugging Face Hub
token = "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"
login(token=token)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Function to extract inputs and outputs
def extract_inputs_outputs(examples):
    inputs = []
    outputs = []

    for item in examples['examples']:
        for example in item:
            input_text = example.get('input')
            output_text = example.get('output')

            if input_text is not None and output_text is not None:
                inputs.append(input_text)
                outputs.append(output_text)
            else:
                print(f"Skipping example due to missing 'input' or 'output': {example}")

    return {'input': inputs, 'output': outputs}

# Apply the extraction function to the dataset
extracted_dataset = dataset.map(extract_inputs_outputs, batched=True, remove_columns=['examples'])

# Function to tokenize inputs and outputs
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['input'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    tokenized_outputs = tokenizer(examples['output'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')

    return {
        'input_ids': tokenized_inputs['input_ids'],
        'labels': tokenized_outputs['input_ids']
    }

# Apply the tokenization function to the extracted dataset
tokenized_dataset = extracted_dataset.map(tokenize_function, batched=True, remove_columns=['input', 'output'])

model_name = "meta-llama/Meta-Llama-3-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer.pad_token = tokenizer.eos_token

# tokenized_data = dataset.map(tokenize_function, batched=True)

from peft import LoraConfig, get_peft_model

lora_alpha = 8
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
)


from transformers import TrainingArguments

output_dir = "./results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 10
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 500
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
)

from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(dataset_text_field="examples", max_seq_length=1024, output_dir="/tmp")

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    #eft_config=sft_config,
    tokenizer=tokenizer,
    args=sft_config,
)

# for name, module in trainer.model.named_modules():
#     if "norm" in name:
#         module = module.to(torch.float32)

trainer.train()

