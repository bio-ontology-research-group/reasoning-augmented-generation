from transformers import AutoModel, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer


def get_model_and_tokenizer():
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    return model, tokenizer


def get_sentence_embedding(model, tokenizer, sentence):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

    # Get the hidden states from the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract embeddings from the last hidden state
    last_hidden_state = outputs.last_hidden_state
    # Pooling: mean pooling, max pooling, etc. (here we use mean pooling)
    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
    sentence_embedding = torch.sum(last_hidden_state * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    return sentence_embedding



def get_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
