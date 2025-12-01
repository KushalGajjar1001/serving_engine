import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_id, device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    return model, tokenizer, device

def encode(text, tokenizer):
    return tokenizer.encode(text, return_tensors="pt")[0].tolist()

def decode(ids, tokenizer):
    return tokenizer.decode(ids, skip_special_tokens=True)
