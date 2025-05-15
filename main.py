from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = FastAPI()

model_name = "miraykoksal/byt5-turkish-spell-check"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

class Sentence(BaseModel):
    text: str

@app.post("/correct")
def correct(sentence: Sentence):
    inputs = tokenizer(sentence.text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)
    fixed = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"corrected": fixed.strip()}
