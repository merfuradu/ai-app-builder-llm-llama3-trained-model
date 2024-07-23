from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

model_path = "merfuradu/trained_model"  # Use the path printed in the previous step
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

class Request(BaseModel):
    input: str
@app.post("/generate")
def generate(request: Request):
    inputs = tokenizer(request.input, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)

# class TextGenerationRequest(BaseModel):
#     prompt: str
#     max_length: int = 50
#
# @app.post("/generate/")
# async def generate_text(request: TextGenerationRequest):
#     inputs = tokenizer(request.prompt, return_tensors="pt")
#     outputs = model.generate(inputs["input_ids"], max_length=request.max_length)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"generated_text": generated_text}
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=11434)
