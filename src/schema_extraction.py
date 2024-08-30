import requests
import json
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def ask_openrouter(prompt, output_format):
    with open("openrouter_api_key.txt") as f:
        API_KEY = f.read().strip()

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
        },
        data=json.dumps({
            "model": "nousresearch/hermes-3-llama-3.1-405b",
            "messages": [
                { "role": "user", "content": f"{prompt}.\n. Please provide your output in {output_format} format. No explanations whatsoever." }
            ]
        })
        )

    llm_output = response.json()["choices"][0]["message"]["content"]
            
    return llm_output

def extract_entities_from_sentence(sentence, shapes):

    prompt = f"Given a SparQL schema and a sentence, extract a set of classes from the sentence that correspond to the schema. The schema is as follows: {shapes}. The sentence is: {sentence}"

    llm_output = ask_openrouter(prompt, "separated by commas")

    return llm_output

def extract_predicates_from_sentence(sentence, shape):
    prompt = f"Given a SparQL schema and a sentence, extract a set of predicates from the sentence that correspond to the schema. The schema is as follows: {shape}. The sentence is: {sentence}"

    llm_output = ask_openrouter(prompt, "separated by commas")
    return llm_output

def formulate_query(sentence, schema):
    
    prompt = f"You are given a sentence and a SparQL schema. From the sentence, formulate a SparQL query. The schema is as follows: {schema}. The sentence is: {sentence}. Adjust the syntax for the Uniprot SparQL endpoint."

    llm_output = ask_openrouter(prompt, "sparql")

    return llm_output
    
        
with open("../data/uniprot_uris.json") as f:
    shapes_str = f.read()
    shapes = json.loads(shapes_str)["shapes"]

    classes = [s["class"] for s in shapes]
    


sentences = ["Which proteins involved in human disease. Retrieve the protein and disease description.",
             "List all Human UniProt entries and their computationaly potential isoforms.",
             "Number of reviewed entries (Swiss-Prot) that are related to kinase activity"
             ]

queries = json.load(open("../data/queries_plus_generated.json"))

for i in tqdm(range(len(queries))):
    if i != 13:
        continue
    
    # if "generated" in queries[i]:
        # continue
    
    sentence = queries[i]["label"]
    print(f"Sentence: {sentence}")

    entities = extract_entities_from_sentence(sentence, shapes_str).split(",")

    query_schema = dict()

    for entity in entities:
        print(f"Entity: {entity}")

        entity_shape = [s for s in shapes if s["class"][0] == entity]

        if len(entity_shape) == 0:
            print(f"Entity {entity} not found in schema")
            continue
        else:
            entity_shape = entity_shape[0]

        predicates = extract_predicates_from_sentence(sentence, entity_shape).split(",")

        query_schema[entity] = predicates
        for predicate in predicates:
            print(f"\tPredicate: {predicate}")


    sparql_query = formulate_query(sentence, query_schema)
    queries[i]["generated"] = sparql_query
    print()
    print(sparql_query)
    print("=====================================")


    # with open("../data/queries_plus_generated.json", "w") as f:
        # json.dump(queries, f, indent=4)
