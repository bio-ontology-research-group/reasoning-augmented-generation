import torch as th
import pandas as pd
import sys
import numpy as np
from utils import get_sbert_model
import transformers as trf



def extract_triple(sentence, pipeline):
    # Format the input for relation extraction
    messages = [
        {
            "role": "system",
            "content": "You are a relation extractor. For a sentence, you find triples (subject,relation,object). You output only the triple (subject,relation,object) and nothing else."
        },
        {"role": "user", "content": sentence},
    ]

    result = pipeline(messages)[0]['generated_text'][-1]["content"]
    return result

def map_classes_and_relations(df, sentence, model):
    sentence_embedding = model.encode(sentence)

    all_entity_embs = th.tensor(np.vstack(df["embedding"]))

    similarities = model.similarity(sentence_embedding, all_entity_embs).squeeze()
    max_idx = th.argmax(similarities).item()

    return df.iloc[max_idx], similarities[max_idx].item()
 


relation_extraction_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
bnb_config = trf.BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=th.bfloat16,
                                    )

relation_extraction_model = trf.AutoModelForCausalLM.from_pretrained(relation_extraction_model_name,
                                                                     device_map="auto",
                                                                     quantization_config=bnb_config,
)

tokenizer = trf.AutoTokenizer.from_pretrained(relation_extraction_model_name)
pipeline = trf.pipeline("text-generation", model=relation_extraction_model,
                             tokenizer=tokenizer,
                             pad_token_id=tokenizer.eos_token_id,
                             device_map= "auto")

matching_model = get_sbert_model()


class_df = pd.read_pickle("../data/class_embeddings_sbert.pkl")
role_df = pd.read_pickle("../data/role_embeddings_sbert.pkl")



sentence = sys.argv[1]
triple = extract_triple(sentence, pipeline)
assert triple.startswith("(") and triple.endswith(")"), f"Invalid triple: {triple}"
print(f"The extracted triple is: {triple}")
triple = triple[1:-1]
relation = triple.split(",")[1]
object_ = triple.split(",")[2]

class_name, score = map_classes_and_relations(class_df, object_, matching_model)
role_name, score = map_classes_and_relations(role_df, relation, matching_model)

class_label = class_name["label"]
role_label = role_name["label"]

print()
print(f"Detected role: {relation} -> {role_label}")
print(f"Detected class: {object_} -> {class_label}")
print()
print(f"OWL sentence: {role_label} some {class_label}")


