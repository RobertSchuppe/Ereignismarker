import os
import glob
import pandas as pd
import spacy
from math import ceil
from collections import defaultdict
import re

nlp = spacy.load("de_core_news_sm")

INPUT_DIR = "./folder/"
OUTPUT_CSV = "./output/corpus_change_of_state.csv" # entsprechend der jeweiligen Ereigniskategorie anpassen

files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))

data = []
text_id_map = {}
text_id_counter = 1

for file_path in files:
    if "change_of_state" in file_path: # entsprechend der jeweiligen Ereigniskategorie anpassen
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
  
        text_id = text_id_counter
        text_id_counter += 1
    
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
        text = re.sub(r'\s*\n\s*', ' ', text)
    
        text = text.replace('###', '#')
    
        doc = nlp(text)
        words = [token for token in doc if not token.is_space]
    
        for token in words:
                data.append({
                    "text_id": text_id,
                    "word": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_
                })

df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"Saved CSV to {OUTPUT_CSV}")