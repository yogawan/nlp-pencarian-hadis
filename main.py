import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_csv('dataset_hadis.csv', on_bad_lines='skip', quotechar='"', engine='python')
print(df)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1) 
    return embeddings

def search_hadis(query, hadis_list):
    query_embedding = get_text_embedding(query)
    
    hadis_embeddings = [get_text_embedding(hadis) for hadis in hadis_list]
    similarities = [cosine_similarity(query_embedding, hadis_embeds).flatten() for hadis_embeds in hadis_embeddings]
    
    most_similar_index = np.argmax(similarities)
    return hadis_list[most_similar_index], df.iloc[most_similar_index]['id']

hadis_list = df['hadis'].tolist()

query = "Apa yang dikatakan Nabi tentang sedekah?"

result, hadis_id = search_hadis(query, hadis_list)

print(f"Pertanyaan: {query}")
print(f"Hadis yang relevan (ID {hadis_id}): {result}")
