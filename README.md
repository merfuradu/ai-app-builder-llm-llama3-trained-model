# app-project-builder-with-ai-llm-llama3
---

# Proiect de Dezvoltare și Antrenare Model AI

Acest proiect se concentrează pe dezvoltarea și antrenarea unui model de inteligență artificială pentru generarea de oferte detaliate de proiecte. Modelul utilizat este un LLM (Large Language Model) bazat pe Llama3, un model open-source dezvoltat de Meta. Proiectul integrează tehnologiile disponibile pe platforma Hugging Face și alte unelte relevante pentru a crea și gestiona acest model AI.

Tehnologii Utilizate
1. Llama3
Modelul de limbaj Llama3, dezvoltat de Meta, este folosit pentru a genera texte coerente și relevante bazate pe input-ul utilizatorului. Llama3 este open-source, ceea ce permite utilizatorilor să-l ruleze local și să-l personalizeze în funcție de nevoile specifice ale proiectului. Modelul este rulabil pe o mașină locală, asigurând flexibilitate și control asupra procesului de antrenare și inferență.

2. Hugging Face
Platforma Hugging Face este utilizată pentru a gestiona și antrena modelul. Hugging Face Hub oferă acces la modele pre-antrenate și suport pentru antrenare personalizată. În cadrul acestui proiect, Hugging Face este folosit pentru a prelua dataset-uri, a tokeniza datele și a antrena modelul.

3. Python și Biblioteca transformers
Proiectul folosește Python pentru implementarea scripturilor de antrenare și generare a textelor. Biblioteca transformers este utilizată pentru manipularea modelului și gestionarea antrenamentului acestuia.

4. WandB
WandB (Weights & Biases) este folosit pentru a monitoriza și vizualiza procesul de antrenare a modelului. Aceasta permite urmărirea metricilor și gestionarea artefactelor modelului într-un mod organizat.

- **CustomTkinter**: Bibliotecă pentru crearea interfețelor grafice.
- **CrewAI**: Utilizat pentru gestionarea agenților și sarcinilor în cadrul proiectului.
- **PyTorch**: Utilizat pentru procesarea și antrenarea modelului AI.
- **Datasets**: Utilizat pentru gestionarea și prelucrarea seturilor de date.

## Scripturi

### 1. `ai_trained_model.py`

Acest script este folosit pentru a inițializa și utiliza modelul antrenat pentru generarea de propuneri detaliate pentru proiecte. Scriptul include următoarele etape:

1. **Inițializare WandB**: Configurarea și conectarea la WandB pentru salvarea și monitorizarea artefactelor.
2. **Restaurarea Modelului**: Descărcarea și încărcarea modelului pre-antrenat utilizând Ollama.
3. **Definirea Agentului**: Crearea unui agent cu rolul de Consultant în Dezvoltare Aplicații care furnizează propuneri detaliate pentru proiecte.
4. **Definirea Sarcinii**: Crearea unei sarcini care include descrierea solicitării clientului și specificarea așteptărilor pentru ieșire.
5. **Executarea Procesului**: Inițializarea și rularea echipei de agenți pentru a finaliza sarcina.
6. **Închiderea WandB**: Finalizarea sesiunii WandB.

```python
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process
import wandb

wandb.login()
run = wandb.init(project="huggingface")
save_path = "D:/PycharmProjects/openai/openai-env/artifacts/model2"
artifact = run.use_artifact('merfuradu-ase/huggingface/model-hseq33ht:v0', type='model')
artifact_dir = artifact.download(save_path)

model = Ollama(model="llama3", base_url="http://localhost:11434")

professor_agent = Agent(
    role="Consultant în Dezvoltare Aplicații",
    goal="""Furnizați propuneri detaliate pentru proiecte clienților, bazate pe cerințele specifice ale acestora, inclusiv tehnologia necesară, limbajele de programare, timpul estimat de dezvoltare și estimările de costuri. În propunere, trebuie să includeți un plan detaliat care să acopere etapele esențiale ale proiectului, și să oferiți estimări exacte pentru costuri și ore de muncă. Toate informațiile trebuie să fie prezentate în limba română, într-un format profesional și clar.""",
    backstory="""Sunteți un consultant experimentat în dezvoltarea aplicațiilor, specializat în crearea de propuneri cuprinzătoare pentru diverse tipuri de aplicații. Aveți competențe avansate în evaluarea cerințelor clienților și în elaborarea unor estimări precise pentru costuri și timp de dezvoltare, și sunteți capabil să redați toate informațiile într-un format bine structurat și în limba română.""",
    allow_delegation=False,
    verbose=True,
    llm=model
)

client_request = input("Solicitarea clientului: ")

task1 = Task(
    description=client_request,
    agent=professor_agent,
    expected_output="Această ofertă preliminară trebuie să includă următoarele informații: 1. Introducere și Context: O scurtă introducere care explică că oferta se bazează pe informațiile furnizate și că înainte de a începe dezvoltarea aplicației este necesară parcurgerea unor etape de planificare esențiale. 2. Etape de Planificare: Diagramă Logică: Descrie procesul de creare a unei diagrame logice pentru arhitectura aplicației. Menționează că această etapă definește structura și fluxul de date al aplicației. Diagramă ER: Explică realizarea unei diagrame entitate-relație (ER) pentru a structura baza de date. Sublinează importanța acestei diagrame în organizarea și gestionarea datelor. Design în Figma: Precizează că se va crea un design inițial în Figma pentru a clarifica aspectul și funcționalitatea interfeței utilizatorului. 3. Costuri și Contract: Diagrama Logică și ER: Costul estimat și TVA-ul pentru aceste documente. Design în Figma: Costul estimat și TVA-ul pentru designul interfeței. Contract și Plată: Menționează că înainte de a începe dezvoltarea, va trebui să fie semnat un contract și să fie achitată suma în avans pentru etapele de planificare. 4. Estimare Finală: Oferta Finală: Clarifică că oferta finală va fi ajustată pe baza etapei de planificare, și că aceasta include costuri și timp de livrare definite după finalizarea planificării. 5. Funcționalități Incluse: Gestionarea Mașinilor: Descrie cum vor fi gestionate și monitorizate mașinile utilizatorilor în aplicație. Notificări și Rapoarte: Detaliază sistemul de notificări pentru expirarea ITP-ului și generarea de rapoarte detaliate pentru erorile mașinii. Facturare și Plată: Explică cum se va gestiona generarea facturilor și opțiunile de plată disponibile. 6. Estimare de Cost și Timp: Cost Total: Estimarea costului total pentru dezvoltarea aplicației, inclusiv TVA. Timp de Livrare: Estimarea timpului necesar pentru finalizarea dezvoltării, bazat pe complexitatea funcționalităților și resursele disponibile. Această ofertă este orientativă și poate fi ajustată în funcție de cerințele suplimentare sau modificările de specificație. Timpul și costurile finale vor fi stabilite în urma finalizării etapei de planificare."""
)

crew = Crew(
    agents=[professor_agent],
    tasks=[task1],
    verbose=2
)

result = crew.kickoff()
print(result)

wandb.finish()
```

### 2. `training5.py`

Acest script este folosit pentru antrenarea modelului AI utilizând datele de intrare și ieșire specificate. Este destinat să funcționeze pe o placă video cu memorie de 4GB.

1. **Încărcarea Dataset-ului**: Încarcă și prelucrează setul de date specificat.
2. **Autentificare Hugging Face**: Conectează-te la Hugging Face Hub.
3. **Încărcarea Tokenizer-ului și Modelului**: Încarcă tokenizer-ul și modelul pre-antrenat.
4. **Tokenizare și Antrenare**: Aplică funcțiile de tokenizare și antrenează modelul folosind configurările specifice.

```python
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from transformers import FalconForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
torch.cuda.empty_cache()

dataset = load_dataset("merfuradu/appbuilder", split='train')

token = os.getenv("HUGGING_FACE_TOKEN")
login(token=token)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

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

extracted_dataset = dataset.map(extract_inputs_outputs, batched=True, remove_columns=['examples'])

def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['input'], padding='max_length', truncation=True, max_length=512

, return_tensors='pt')
    tokenized_outputs = tokenizer(examples['output'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')

    return {
        'input_ids': tokenized_inputs['input_ids'],
        'labels': tokenized_outputs['input_ids']
    }

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

lora_alpha = 8
lora_dropout = 0.1
lora_r = 32

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj"]
)
model = get_peft_model(model, peft_config)
model.config.use_cache = False

training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=500,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    gradient_checkpointing=True,
)

sft_config = SFTConfig(dataset_text_field="examples", max_seq_length=1024, output_dir="/tmp")

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    args=sft_config,
)

model.config.use_cache = False
trainer.train()
```

### 3. `training.py` (Alternative)

Acest script alternativ este destinat utilizării pe o placă video mai puternică. Este similar cu `training5.py`, dar folosește argumente de antrenament diferite și mai simplificate.

```python
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset


token = "HUGGING_FACE_TOKEN"
login(token=token)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)

try:
    dataset = load_dataset("merfuradu/appbuilder", split="train")
    dataset = dataset.shuffle(seed=42).select([i for i in list(range(1000))])
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit()

def tokenize_function(examples):
    try:
        return tokenizer(examples["input"], examples["output"], padding='max_length', truncation=True, max_length=512)
    except Exception as e:
        print(f"Tokenization error: {e}")
        return {'input_ids': [], 'attention_mask': []}

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4)

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

try:
    trainer.train()
except Exception as e:
    print(f"An error occurred during training: {e}")
```

## Instrucțiuni de Utilizare

1. **Pregătirea Mediu**: Asigurați-vă că aveți toate bibliotecile necesare instalate. Puteți utiliza un mediu virtual sau conda pentru a gestiona dependențele.

2. **Rularea Scripturilor**: Folosiți următoarele comenzi pentru a rula scripturile:
   - Pentru utilizarea modelului antrenat: `python ai_trained_model.py`
   - Pentru utilizarea modelului antrenat cu ajutorul unui UI (User Interface): `python ui.py`
   - Pentru antrenarea modelului cu `training5.py`: `python training5.py`
   - Pentru antrenarea modelului cu `training.py` (pentru plăci video mai puternice): `python training.py`

3. **Utilizarea WSL**: În cazul în care folosiți Windows Subsystem for Linux (WSL), asigurați-vă că aveți configurat corect Python 3 și bibliotecile necesare.

## Cerințe de Sistem

- **GPU**: Scripturile sunt optimizate pentru GPU-uri cu cel puțin 4GB de memorie. Pentru antrenare, se recomandă o placă video mai puternică.
- **Python**: Versiunea 3.8 sau mai recentă.
- **Dependențe**: `langchain_community`, `crewai`, `wandb`, `huggingface_hub`, `transformers`, `datasets`, `torch`, `peft`, `trl`.

## Note

- **Token de Autentificare**: Asigurați-vă că înlocuiți token-ul `HUGGING_FACE_TOKEN` cu token-ul vostru Hugging Face pentru a accesa resursele necesare.
- **Configurări Specifice**: Configurările pentru antrenare pot fi ajustate în funcție de resursele hardware disponibile.

---

Acest README oferă o imagine de ansamblu clară asupra proiectului, inclusiv tehnologiile utilizate, scripturile principale, instrucțiunile de utilizare și cerințele de sistem.
