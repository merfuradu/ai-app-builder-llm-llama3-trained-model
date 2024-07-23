import os

from huggingface_hub import login, notebook_login
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from dotenv import load_dotenv


# Set logging level
# logging.basicConfig(level=logging.DEBUG)
# transformers.utils.logging.set_verbosity_debug()

token = os.getenv("HUGGING_FACE_TOKEN")
login(token=token)
#transformers.utils.logging.set_verbosity_error()


# Încarcă tokenizer-ul și modelul pre-antrenat
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)

# tokenizer.save_pretrained("D:/llama3/Meta-Llama-3-8B")
# model.save_pretrained("D:/llama3/Meta-Llama-3-8B")
#
# tokenizer = AutoTokenizer.from_pretrained("D:/llama3/Meta-Llama-3-8B")
# model = AutoModelForCausalLM.from_pretrained("D:/llama3/Meta-Llama-3-8B")

# Încarcă setul de date
try:
    dataset = load_dataset("merfuradu/appbuilder", split="train")
    dataset = dataset.shuffle(seed=42).select([i for i in list(range(1000))])
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit()

# Tokenizează datele
def tokenize_function(examples):
    try:
        return tokenizer(examples["input"], examples["output"], padding='max_length', truncation=True, max_length=512)
    except Exception as e:
        print(f"Tokenization error: {e}")
        return {'input_ids': [], 'attention_mask': []}

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4)

# Pregătește argumentele pentru antrenament
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=True,
)

# Creează trainer-ul
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Antrenează modelul
try:
    trainer.train()
except Exception as e:
    print(f"An error occurred during training: {e}")

