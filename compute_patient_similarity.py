import sys
print('HI',sys.version)

import pandas as pd
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Precomputes patient-note similarity once so every model run in
# run_sbatch ...sh can just load it instead of recomputing the same
# TF-IDF/cosine matrix per array task (it doesn't depend on MODEL).

path_surgery_outcome="/home/***/LLM/Data/df-combined-SZ-NPSY-pre-post-alternate_SID_WO_duplicate.xlsx"
df_surgery_outcome=pd.read_excel(path_surgery_outcome, dtype=str)
df_surgery_outcome = df_surgery_outcome[ df_surgery_outcome[["Surgery Date 2", "Surgery Date 3", "Surgery Date 4"]].isna().all(axis=1)]
df_surgery_outcome = df_surgery_outcome[ df_surgery_outcome["Class 7-12M"].notna()]

MRNs=df_surgery_outcome["SID"].to_list()

path_txts="/home/***/LLM/Data/deidentified"

SIMILARITY_DIR='/home/***/LLM/result/patient-similarity'
SIMILARITY_MATRIX_PATH=os.path.join(SIMILARITY_DIR,'similarity_matrix.npy')
MRNS_ORDER_PATH=os.path.join(SIMILARITY_DIR,'mrns_order.json')

if os.path.exists(SIMILARITY_MATRIX_PATH) and os.path.exists(MRNS_ORDER_PATH) and '--force' not in sys.argv:
    print(f"Similarity matrix already exists at {SIMILARITY_MATRIX_PATH}, skipping recomputation.")
    print("Pass --force to recompute anyway.")
    sys.exit(0)

texts=[]
_mrns=[]

for path_txt_ in os.listdir(path_txts):
    path_txt=os.path.join(path_txts,path_txt_)
    path_txt_mrn=path_txt.split("/")[-1][:-4]
    if path_txt_mrn in MRNs:
        try:
            with open(path_txt, "r", encoding="utf-8") as file:  # Specify UTF-8 encoding
                text = file.read()
                _mrns.append(path_txt_mrn)
                texts.append(text)
        except:
            print("-------------ERROR-----",path_txt_[0:-4] )

print(f"Computing TF-IDF similarity matrix over {len(texts)} patients...")

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(texts)
similarity_matrix = cosine_similarity(tfidf_matrix)

os.makedirs(SIMILARITY_DIR, exist_ok=True)
np.save(SIMILARITY_MATRIX_PATH, similarity_matrix)
with open(MRNS_ORDER_PATH, 'w') as f:
    json.dump(_mrns, f)

print(f"Saved similarity matrix {similarity_matrix.shape} to {SIMILARITY_MATRIX_PATH}")
print(f"Saved MRN row/column order to {MRNS_ORDER_PATH}")
