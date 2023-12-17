#%%
from pathlib import Path
import torch
import esm
import lightning.pytorch as pl
import time
import sys
import esm2
import numpy as np
import tqdm
from copy import deepcopy
import xgboost as xgb
import scipy as sp 
from esm_data import BatchFilesSeqDataset, ProtSeqBatchConverter, iter_manual_worker_init_fn
import pickle 
precision = 16
flash_attention = True

# ESM variables
model_name = "esm2_t33_650M_UR50D"
esm_pt = esm.pretrained.load_model_and_alphabet(model_name)

model_cache_dir = Path('/manitou/pmg/users/vss2134/cache/hub/checkpoints')
# data
fasta_files = [Path('/manitou/pmg/users/vss2134/HPML/project/fluorescence/fluorescence_train.fasta'),]  # worker=0 for single file

if precision == 16:
    autocast = True
    #assert flash_attention is True, 'FP16 wihtout flashattention gives poor results'
else:
    autocast = False
    assert flash_attention is False, 'FlashAttention only works with FP16/BF16'

model_location = model_cache_dir / f'{model_name}.pt'
model_data = torch.load(model_location, map_location="cpu")
model_args = model_data["cfg"]['model']
alphabet = esm.data.Alphabet.from_architecture("ESM-1b")

model_min_args = {
    # Model args
    'attention_heads': model_args.encoder_attention_heads,
    'embed_dim': model_args.encoder_embed_dim,
    'num_layers': model_args.encoder_layers,
    'token_dropout': model_args.token_dropout,
    'flash_attention': flash_attention,
    'alphabet': alphabet
}
model = esm2.ESM2(**model_min_args).to("cuda")
model_vanilla = deepcopy(model)
#%%

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd
from Bio.Seq import Seq
import torch 
train_seqs = []
val_seqs = []
seq_iter = SeqIO.parse("/manitou/pmg/users/vss2134/HPML/project/uniref50.fasta", "fasta")
for i in range(10000):
    fa_seq = next(seq_iter)
    seq = str(fa_seq.seq)
    if len(seq) > 1024:
        seq = seq=seq[:1024]
    if len(seq) > 100:
        train_seqs.append( SeqRecord(id=fa_seq.id, seq=Seq(seq), description="" ))    
for i in range(10000):
    fa_seq = next(seq_iter)
    seq = str(fa_seq.seq)
    if len(seq) > 1024:
        seq = seq=seq[:1024] 
    if len(seq) > 100:
        val_seqs.append( SeqRecord(id=fa_seq.id, seq=Seq(seq), description="" ))



class SeqDataSet:
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        id = self.seqs[idx].id
        seq = str(self.seqs[idx].seq)
        return (id,seq,True,True)
train_ds = SeqDataSet(train_seqs)
val_ds = SeqDataSet(val_seqs)
train_dl = torch.utils.data.DataLoader(
    train_ds,
    num_workers=2,
    collate_fn=ProtSeqBatchConverter(
        alphabet,
        masking=True),
    batch_size=12,
    pin_memory=True,
    shuffle=True,
)
train_dl = [batch for batch in train_dl]

val_dl = torch.utils.data.DataLoader(
    val_ds,
    num_workers=2,
    collate_fn=ProtSeqBatchConverter(
        alphabet,
        masking=True),
    batch_size=256,
    pin_memory=True,
    shuffle=False, 
    drop_last=False
)
val_dl = [batch for batch in val_dl] # so masks are deterministic


meltome_data = pd.read_csv("/manitou/pmg/users/vss2134/HPML/project/splits/mixed_split.csv").assign(
    seql = lambda x: x['sequence'].str.len()
).query("seql < 1024")

meltome_train_seqs = [SeqRecord(id=f"M{i}", seq=Seq(row['sequence']), description="") for i, row in meltome_data[meltome_data['set'] == 'train'].iterrows()]
meltome_train_dl = torch.utils.data.DataLoader(
    SeqDataSet(meltome_train_seqs),
    num_workers=2,
    collate_fn=ProtSeqBatchConverter(
        alphabet,
        masking=False),
    batch_size=256,
    pin_memory=True,
    shuffle=False,
)

meltome_test_seqs = [SeqRecord(id=f"M{i}", seq=Seq(row['sequence']), description="") for i, row in meltome_data[meltome_data['set'] == 'test'].iterrows()]
meltome_test_dl = torch.utils.data.DataLoader(
    SeqDataSet(meltome_test_seqs),
    num_workers=2,
    collate_fn=ProtSeqBatchConverter(
        alphabet,
        masking=False),
    batch_size=256,
    pin_memory=True,
    shuffle=False,
)

#### 

model_vanilla.eval()
all_vanilla_mlm_val_loss = []
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=autocast):
        for n, batch in enumerate( tqdm.tqdm(val_dl)):
            (batch_labels, seq_str_list, noised_tokens, tokens, noise_mask, _) = batch
            noised_tokens = noised_tokens.to("cuda")
            lens = torch.tensor([len(s) for s in seq_str_list])
            lm_pred = model_vanilla(noised_tokens)["logits"]
            masked_tokens = tokens.masked_fill(~noise_mask, -100)
            masked_tokens = masked_tokens.to("cuda")
            lm_loss_fn = torch.nn.CrossEntropyLoss(reduction='none',
                                                ignore_index=-100)
            loss_noreduce = lm_loss_fn(
            lm_pred.view(-1, 33),
            masked_tokens.view(-1)).float()
            loss_nonzero = loss_noreduce[masked_tokens.view(-1) != -100]
            all_vanilla_mlm_val_loss +=loss_nonzero.detach().cpu().numpy().tolist()


meltome_train_emebeddings = []
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=autocast):
        for n, batch in enumerate( tqdm.tqdm(meltome_train_dl)):
            (batch_labels, seq_str_list, noised_tokens, tokens, noise_mask, _) = batch
            noised_tokens = noised_tokens.to("cuda")
            lens = torch.tensor([len(s) for s in seq_str_list])
            lm_pred = model_vanilla(noised_tokens, repr_layers=[33])["representations"][33]
            seq_lens = torch.tensor([len(s) for s in seq_str_list])
            for i in range(len(seq_lens)):
                meltome_train_emebeddings.append(lm_pred[i, :seq_lens[i], :].mean(0).cpu().numpy())

meltome_test_emebeddings = []
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=autocast):
        for n, batch in enumerate( tqdm.tqdm(meltome_test_dl)):
            (batch_labels, seq_str_list, noised_tokens, tokens, noise_mask, _) = batch
            noised_tokens = noised_tokens.to("cuda")
            lens = torch.tensor([len(s) for s in seq_str_list])
            lm_pred = model_vanilla(noised_tokens, repr_layers=[33])["representations"][33]
            seq_lens = torch.tensor([len(s) for s in seq_str_list])
            for i in range(len(seq_lens)):
                meltome_test_emebeddings.append(lm_pred[i, :seq_lens[i], :].mean(0).cpu().numpy())

meltome_train_mat = np.array(meltome_train_emebeddings)
meltome_test_mat = np.array(meltome_test_emebeddings)
meltome_train_labels = meltome_data[meltome_data['set'] == 'train']['target'].values
meltome_test_labels = meltome_data[meltome_data['set'] == 'test']['target'].values

regressor = xgb.XGBRegressor(device="cuda")
regressor.fit(meltome_train_mat, meltome_train_labels)
meltome_test_preds = regressor.predict(meltome_test_mat)

vanilla_meltome_pcor = sp.stats.pearsonr(meltome_test_preds, meltome_test_labels)            

del model_vanilla, meltome_train_emebeddings, meltome_test_emebeddings, meltome_train_mat, meltome_test_mat, meltome_train_labels, meltome_test_labels, regressor, meltome_test_preds
#%%
import torch.nn.utils.prune as prune
import re 
def name2module(pl):
    name, param = pl 
    name = name.split(".")
    for i in range(len(name)):
        if re.match(r"\d+", name[i]):
            new_name = name[i-1] + f"[{name[i]}]"
            name[i-1] =  new_name
            name.pop(i)
            break 
    name = ".".join(name[:-1])
    return eval(f"model.{name}")

parameters_to_prune = [
    (name2module(i), "weight") for i in model.named_parameters() if "weight" in i[0]
]
print(len(parameters_to_prune))
parameters_to_prune = [(i[0], i[1]) for i in parameters_to_prune if len(i[0].weight.shape) > 1]
print(len(parameters_to_prune))
parameters_to_prune = [(i[0], i[1]) for i in parameters_to_prune if i[0].weight.shape[0] > 1]
print(len(parameters_to_prune))
parameters_to_prune = parameters_to_prune[1:]
### apricot!
for param,_ in parameters_to_prune:
    torch.nn.utils.prune.ln_structured(param, name="weight", amount=float(sys.argv[1]), n=2, dim=1)

#%%
model.eval()
all_sparse_mlm_val_loss = []
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=autocast):
        for n, batch in enumerate( tqdm.tqdm(val_dl)):
            (batch_labels, seq_str_list, noised_tokens, tokens, noise_mask, _) = batch
            noised_tokens = noised_tokens.to("cuda")
            lens = torch.tensor([len(s) for s in seq_str_list])
            lm_pred = model(noised_tokens)["logits"]
            masked_tokens = tokens.masked_fill(~noise_mask, -100)
            masked_tokens = masked_tokens.to("cuda")
            lm_loss_fn = torch.nn.CrossEntropyLoss(reduction='none',
                                                ignore_index=-100)
            loss_noreduce = lm_loss_fn(
            lm_pred.view(-1, 33),
            masked_tokens.view(-1)).float()
            loss_nonzero = loss_noreduce[masked_tokens.view(-1) != -100]
            all_sparse_mlm_val_loss +=loss_nonzero.detach().cpu().numpy().tolist()


meltome_train_emebeddings = []
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=autocast):
        for n, batch in enumerate( tqdm.tqdm(meltome_train_dl)):
            (batch_labels, seq_str_list, noised_tokens, tokens, noise_mask, _) = batch
            noised_tokens = noised_tokens.to("cuda")
            lens = torch.tensor([len(s) for s in seq_str_list])
            lm_pred = model(noised_tokens, repr_layers=[33])["representations"][33]
            seq_lens = torch.tensor([len(s) for s in seq_str_list])
            for i in range(len(seq_lens)):
                meltome_train_emebeddings.append(lm_pred[i, :seq_lens[i], :].mean(0).cpu().numpy())

meltome_test_emebeddings = []
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=autocast):
        for n, batch in enumerate( tqdm.tqdm(meltome_test_dl)):
            (batch_labels, seq_str_list, noised_tokens, tokens, noise_mask, _) = batch
            noised_tokens = noised_tokens.to("cuda")
            lens = torch.tensor([len(s) for s in seq_str_list])
            lm_pred = model(noised_tokens, repr_layers=[33])["representations"][33]
            seq_lens = torch.tensor([len(s) for s in seq_str_list])
            for i in range(len(seq_lens)):
                meltome_test_emebeddings.append(lm_pred[i, :seq_lens[i], :].mean(0).cpu().numpy())

meltome_train_mat = np.array(meltome_train_emebeddings)
meltome_test_mat = np.array(meltome_test_emebeddings)
meltome_train_labels = meltome_data[meltome_data['set'] == 'train']['target'].values
meltome_test_labels = meltome_data[meltome_data['set'] == 'test']['target'].values

regressor = xgb.XGBRegressor(device="cuda")
regressor.fit(meltome_train_mat, meltome_train_labels)
meltome_test_preds = regressor.predict(meltome_test_mat)

sparse_meltome_pcor = sp.stats.pearsonr(meltome_test_preds, meltome_test_labels)

output = {
    "vanilla_mlm_val_loss": all_vanilla_mlm_val_loss,
    "sparse_mlm_val_loss": all_sparse_mlm_val_loss,
    "vanilla_meltome_pcor": vanilla_meltome_pcor,
    "sparse_meltome_pcor": sparse_meltome_pcor
}

with open(f"/manitou/pmg/users/vss2134/HPML/project/sparse_results/global_struct_{sys.argv[1]}.pkl", "wb") as f:
    pickle.dump(output, f)
