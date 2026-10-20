import sys
print('HI',sys.version)

import ollama
import pandas as pd
import os
import numpy as np
import re
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import gc
import ctypes

ROLE_PROMPT = (
    """You are an experienced epileptologist and epilepsy surgery specialist evaluating a patient for surgical intervention. \n\n
    "### Core Task\n
    instruction: BLA BLAH BLAH ...
    
    Outcome: [Good/Poor] """
)

def estimate_num_ctx(text, min_ctx=4096, max_ctx=131072, buffer_tokens=1024):
    
    approx_tokens = len(text)//4 + buffer_tokens
    return max(min_ctx, min(max_ctx, approx_tokens))

def chat(text, role="you are an assistant", MODEL="gpt-oss:20b", temp=0, num_ctx=None):
    if num_ctx is None:
        num_ctx = estimate_num_ctx(text)
    response1 = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": text}
        ],
        options={'temperature': temp, 'num_ctx': num_ctx}
    )
    return response1["message"]["content"]

def validate_resp_Intervention_shot(
    response: str
) -> Tuple[bool, Optional[str]]:
    """
    Validates LLM output for intervention prediction.
    """

    pattern = r"""
        ^Outcome:\s*(Good|Poor)$
    """

    match = re.match(
        pattern,
        response.strip(),
        re.MULTILINE | re.DOTALL | re.VERBOSE | re.IGNORECASE
    )

    if not match:
        return False, None

    outcome = match.group(1).capitalize()

    return True, outcome


path_surgery_outcome="/home/***/LLM/Data/df-combined-SZ-NPSY-pre-post-alternate_SID_WO_duplicate.xlsx"
df_surgery_outcome=pd.read_excel(path_surgery_outcome, dtype=str)
df_surgery_outcome = df_surgery_outcome[ df_surgery_outcome[["Surgery Date 2", "Surgery Date 3", "Surgery Date 4"]].isna().all(axis=1)]
df_surgery_outcome = df_surgery_outcome[ df_surgery_outcome["Class 7-12M"].notna()]
OC_class_7_12m=df_surgery_outcome["Class 7-12M"].to_list()
OC_class=[int(i.split("-")[0]) for i in OC_class_7_12m if not pd.isna(i)]

GP=lambda i: "Good" if i<3 else "Poor"  # determines that good/poor outcome, G if class<=2, P otherwise
OC=[GP(i) for i in OC_class]

print("Percentage of good surgery outcomes --> ",OC.count("Good")/len(OC))


MRNs=df_surgery_outcome["SID"].to_list()

path_txts="/home/***/LLM/Data/deidentified"
MRNs1=[i for i in os.listdir(path_txts) if i[:-4] in MRNs]




################## model name from run_sbatch...sh file
MODEL=sys.argv[1]
model_name=MODEL.replace(":", "-")

# numbers of most-similar patients to try as few-shot examples per target case
K_SHOTS_LIST=[1, 3, 5, 10, 20, 30, 40, 50]


role=ROLE_PROMPT
TEMP=0.7 # this is just for loop , the defult is 0
#######################

def free_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass

def chat_with_outcome_score(text_):
                        text=text_[1]
                        MRN=text_[0]
                        print(text)
                        res_outcome=chat(text,role=role,MODEL=MODEL)
                        # print(res_outcome)
                        outcomeASlist=validate_resp_Intervention_shot(res_outcome)
                        # cc=0
                        # while not ( outcomeASlist[0] ) and (cc<5) :
                        #     cc+=1
                        #     res_outcome=chat(text,MODEL=MODEL,role=ROLE_PROMPT,temp=TEMP)
                        #     outcomeASlist=validate_resp_Intervention_shot(res_outcome)
                        # outcome_score=extract_score_(res_outcome)
                        # print(outcomeASlist)
                        filename=os.path.join(path_save_res_as_json,MRN+'.json')
                        jf={"Follow Format": outcomeASlist[0],"Outcome":outcomeASlist[1]}
                        with open(filename,'w') as f:
                               json.dump(jf,f)
                        return [MRN,outcomeASlist]

c=0
dict_resp={}
texts=[]
texts_mrn=[]
_mrns=[]

for path_txt_ in os.listdir(path_txts):
    path_txt=os.path.join(path_txts,path_txt_)
    path_txt_mrn=path_txt.split("/")[-1][:-4]
    if path_txt_mrn in MRNs:

            c+=1
            if c>0 :

                    try:
                        with open(path_txt, "r", encoding="utf-8") as file:  # Specify UTF-8 encoding
                            text = file.read()
                            # texts_mrn.append((path_txt_mrn,text))
                            _mrns.append(path_txt_mrn)
                            texts.append(text)


                    except:
                        print("-------------ERROR-----",path_txt_[0:-4] )

# Load the similarity matrix precomputed once by compute_patient_similarity.py

SIMILARITY_DIR='/home/***/LLM/result/patient-similarity'
similarity_matrix = np.load(os.path.join(SIMILARITY_DIR,'similarity_matrix.npy'))
with open(os.path.join(SIMILARITY_DIR,'mrns_order.json')) as f:
    sim_mrns = json.load(f)

mrn_to_text = dict(zip(_mrns, texts))
mrn_to_sim_idx = {mrn: idx for idx, mrn in enumerate(sim_mrns)}

for K_SHOTS in K_SHOTS_LIST:

    path_save_res_as_json=os.path.join('/home/***/LLM/result/intervention-similar-shots',model_name+f"-top{K_SHOTS}")
    if not os.path.exists(path_save_res_as_json):
          os.makedirs(path_save_res_as_json)
    mrn_done=[i[:-5] for i in os.listdir(path_save_res_as_json)]
    c=0
    texts_mrn=[]
    for path_txt_ in os.listdir(path_txts):
        path_txt=os.path.join(path_txts,path_txt_)
        path_txt_mrn=path_txt.split("/")[-1][:-4]
        if path_txt_mrn in MRNs:
             if not path_txt_mrn in mrn_done:
                c+=1
                if c>0 :

                    target_sim_idx=mrn_to_sim_idx[path_txt_mrn]

                    sims=similarity_matrix[target_sim_idx].copy()
                    sims[target_sim_idx]=-1  # exclude the target case itself
                    top_k_sim_idx=sims.argsort()[::-1][:K_SHOTS]

                    selected_mrn=[sim_mrns[i] for i in top_k_sim_idx]
                    selected_texts=[mrn_to_text[m] for m in selected_mrn]
                    selected_mrn_ind=[MRNs.index(i) for i in selected_mrn]
                    selected_outcomes=[OC[i] for i in selected_mrn_ind]

                    text=mrn_to_text[path_txt_mrn]

                    examples = "\n\n".join(
                        f"Patient Case {i+1}: {txt}\nOutcome: {out}"
                        for i, (txt, out) in enumerate(zip(selected_texts, selected_outcomes))   )

                    new_task = f"Patient Case: {text} \nOutcome:"

                    formatted_prompt = f"""### EXAMPLES

                    {examples}

                    ---

                    ### NEW TASK

                    {new_task}"""

                    texts_mrn.append((path_txt_mrn,formatted_prompt))

    BATCH = 20

    for i in range(0, len(texts_mrn), BATCH):

       batch = texts_mrn[i:i+BATCH]

       with ThreadPoolExecutor(max_workers=2) as exe:
            res = list(exe.map(chat_with_outcome_score, batch))

       del batch, res
       free_memory()
