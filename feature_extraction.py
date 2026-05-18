

import string, re
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, EsmModel
import numpy as np
from tqdm import tqdm

ProtTrans_path = "your ProtTrans_path"
feat_batchsize = 8

def get_ID(name_item): # deal with IDs with different format
    name_item = name_item.split("|")
    ID = "_".join(name_item[0:min(2, len(name_item))])
    ID = re.sub(" ", "_", ID)
    return ID
def process_fasta(fasta_file):  # 序列长度大于2000的会被修剪
    ID_list = []
    seq_list = []

    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == ">":
            ID_list.append(get_ID(line[1:-1]))
        elif line[0] in string.ascii_letters:
            seq = line.strip().upper()
            seq_list.append(seq[0:min(2000, len(seq))])  # trim long sequence to 2000
    df = pd.DataFrame(list(zip(ID_list, seq_list)), columns=["Protein_ID", 'seq'])

    # df_filtered = df["seq"].copy()

    return df


def feature_extraction_for_mamba(df_seq, device, output_path):
    tokenizer = T5Tokenizer.from_pretrained(ProtTrans_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(ProtTrans_path)

    gc.collect()

    model = model.to(device)
    model = model.eval()

    seq = df_seq["seq"].tolist()
    seq = [re.sub(r"[UZOB]", "X", " ".join(list(sequence))) for sequence in seq]
    ID_list = df_seq["Protein_ID"].tolist()

    all_embeddings = []

    for i in tqdm(range(0, len(seq), feat_batchsize)):
        if i + feat_batchsize <= len(seq):
            batch_ID_list = ID_list[i:i + feat_batchsize]
            batch_seq_list = seq[i:i + feat_batchsize]
        else:
            batch_ID_list = ID_list[i:]
            batch_seq_list = seq[i:]

        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, max_length=121,
                                          padding='max_length', truncation=True, return_tensors='pt')

        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        print(f"embedding shape:{embedding.shape}")
        all_embeddings.append(embedding)

    final_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(output_path, final_embeddings)
    print("finishing feature extraction")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print("starting load data")

df_filtered = process_fasta("./data1/600p+599n.txt")

output_path_for_600599="/your output path"
print("extracting feature!!!")


print(f"df_filtered.len:{len(df_filtered)}")

feature_extraction_for_mamba(df_filtered,device,output_path_for_600599)

print("finished extracting feature!!!")