from datasets import load_dataset
from huggingface_hub import login
import torch
import os
from transformers import FalconForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
# dataset_name = "openai-env/dataset.json"

# if not os.path.isfile(file_path):
#     raise FileNotFoundError(f"Unable to find '{file_path}'")

dataset = load_dataset("merfuradu/appbuilder", split='train')
print(dataset['examples'])

token = os.getenv("HUGGING_FACE_TOKEN")
login(token=token)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# def tokenize_function(examples):
#     return tokenizer(examples["input"],examples["output"], padding=4096, truncation=True)

def extract_inputs_outputs(dataset):
    inputs = []
    outputs = []

    for example in dataset['examples']:
        for item in example:
                input_text = item.get('input')
                output_text = item.get('output')

                if input_text is not None and output_text is not None:
                    inputs.append(input_text)
                    outputs.append(output_text)
                else:
                    print(f"Skipping example due to missing 'input' or 'output': {item}")

    return inputs, outputs

def tokenize_function(inputs, outputs):
    tokenized_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    tokenized_outputs = tokenizer(outputs, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

    return {
        'input_ids': tokenized_inputs['input_ids'],
        'labels': tokenized_outputs['input_ids']
    }


# Extract inputs and outputs
inputs = extract_inputs_outputs(dataset)[0]
outputs = extract_inputs_outputs(dataset)[1]


# Tokenize inputs and outputs
tokenized_data = tokenize_function(inputs, outputs)


# Tokenization function




def get_value_safe(dictionary, key):
    """Safely get a value from a dictionary."""
    try:
        return dictionary[key]
    except KeyError:
        return None

# def tokenize_function(batch):
#     inputs = []
#     outputs = []
#
#     # Check for the correct structure and iterate accordingly
#     if 'examples' in batch:
#         examples = batch['examples']
#         # Flatten the nested list of examples
#         flattened_examples = [item for sublist in examples for item in sublist]
#     else:
#         flattened_examples = batch  # Assume the batch is directly the list of examples
#
#     for example in flattened_examples:
#         # Add detailed print statements for debugging
#         print(f"Processing example: {example}")
#
#         input_text = get_value_safe(example, 'input')
#         output_text = get_value_safe(example, 'output')
#
#         if input_text is not None and output_text is not None:
#             print(f"Valid example found: input: {input_text}, output: {output_text}")
#             inputs.append(input_text)
#             outputs.append(output_text)
#         else:
#             print(f"Skipping example due to missing 'input' or 'output': {example}")
#
#     # Raise an error if no valid inputs or outputs found
#     if not inputs or not outputs:
#         raise ValueError("The dataset does not contain valid 'input' and 'output' fields in any example")
#
#     # Tokenize inputs and outputs separately
#     tokenized_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=512, return_overflowing_tokens=True)
#     tokenized_outputs = tokenizer(outputs, padding='max_length', truncation=True, max_length=512, return_overflowing_tokens=True)
#
#     print("Tokenized Inputs:", tokenized_inputs)  # Print tokenized inputs for debugging
#     print("Tokenized Outputs:", tokenized_outputs)  # Print tokenized outputs for debugging
#
#     input_ids = [ids for ids in tokenized_inputs['input_ids']]
#     labels = [ids for ids in tokenized_outputs['input_ids']]
#
#     # Return as a dictionary where keys are strings and values are lists
#     return {
#         'input_ids': input_ids,
#         'labels': labels
#     }
    #/////////////////////////////////////////////////////////////////////////
    # input_ids = tokenized_inputs['input_ids']
    # labels = tokenized_outputs['input_ids']

    # if len(input_ids) != len(labels):
    #     raise ValueError("The length of input_ids and labels must be the same")




tokenized_dataset = dataset.map(tokenize_function(inputs, outputs), batched=True)


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

from peft import LoraConfig

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
    train_dataset=tokenized_data,
    #eft_config=sft_config,
    tokenizer=tokenizer,
    args=sft_config,
)

# for name, module in trainer.model.named_modules():
#     if "norm" in name:
#         module = module.to(torch.float32)

trainer.train()
