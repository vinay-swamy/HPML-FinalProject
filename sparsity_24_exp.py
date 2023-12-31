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

"""
Evaluate the effect of pruning on ESM models. 
For MLM loss, use the first 10K uniref50 sequences.
for meltome dataset, generate mean pooled emebddings, and train an XGBoost regressor.
"""
# ESM variables
model_name = "esm2_t33_650M_UR50D"
esm_pt = esm.pretrained.load_model_and_alphabet(model_name)

model_cache_dir = Path('/manitou/pmg/users/vss2134/cache/hub/checkpoints')
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

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd
from Bio.Seq import Seq
import torch 
train_seqs = []
val_seqs = []
## load in first 10K uniref50 sequences
## some of them are quite long, so just  truncate them to 1024 for now
## prefetch the batches so that the random mask is the same across models
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

## load in Meltome data from FLIP

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
#%%
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
#%%

meltome_train_emebeddings = []
torch.cuda.reset_peak_memory_stats()
vanilla_meltome_train_runtime = []
with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=autocast):
        for n, batch in enumerate( tqdm.tqdm(meltome_train_dl)):
            (batch_labels, seq_str_list, noised_tokens, tokens, noise_mask, _) = batch
            noised_tokens = noised_tokens.to("cuda")
            lens = torch.tensor([len(s) for s in seq_str_list])
            start = time.perf_counter_ns()
            torch.cuda.synchronize()
            lm_pred = model_vanilla(noised_tokens, repr_layers=[33])["representations"][33]
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            vanilla_meltome_train_runtime.append((end-start)/1e9)
            seq_lens = torch.tensor([len(s) for s in seq_str_list])
            for i in range(len(seq_lens)):
                meltome_train_emebeddings.append(lm_pred[i, :seq_lens[i], :].mean(0).cpu().numpy())
            if n > 5:
                break
vanilla_meltome_train_runtime = np.mean(vanilla_meltome_train_runtime)
vanilla_meltome_peakmem = torch.cuda.max_memory_allocated() / 1e9
#%%
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

####################################################
## following the workflow for 2:4 sparsity https://pytorch.org/tutorials/prototype/semi_structured_sparse.html
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
SparseSemiStructuredTensor._FORCE_CUTLASS = True
from torch.ao.pruning import WeightNormSparsifier
sparsifier = WeightNormSparsifier(
    # apply sparsity to all blocks
    sparsity_level=1.0,
    # shape of 4 elemens is a block
    sparse_block_shape=(1, 4),
    # two zeros for every block of 4
    zeros_per_block=2)
sparse_config = [
    {"tensor_fqn": f"{fqn}.weight"}
    for fqn, module in model.named_modules()
    if isinstance(module, torch.nn.Linear) and "layer" in fqn
]
sparsifier.prepare(model, sparse_config)
sparsifier.step()
sparsifier.squash_mask()
model = model.cuda().half()
# accelerate for sparsity
# old_modules  = [deepcopy(module) for fqn,module  in model.named_modules()]
for fqn, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and "layer" in fqn:
        module.weight = torch.nn.Parameter(to_sparse_semi_structured(module.weight))
# new_modules  = [module for fqn,module  in model.named_modules()]
# #%%
# for i in range(len(old_modules)):
#     if isinstance(old_modules[i], torch.nn.Linear):
#         if old_modules[i].weight.shape != new_modules[i].weight.shape:
#             print(i, old_modules[i].weight.shape, new_modules[i].weight.shape)
#%%
###################################################
# Note: torch.inference_mode is mandatory for sparse inference

model.eval()
all_sparse_mlm_val_loss = []
with torch.inference_mode():
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

sparse_meltome_runtime = []
torch.cuda.reset_peak_memory_stats()
meltome_train_emebeddings = []
with torch.inference_mode():
    with torch.cuda.amp.autocast(enabled=autocast):
        for n, batch in enumerate( tqdm.tqdm(meltome_train_dl)):
            (batch_labels, seq_str_list, noised_tokens, tokens, noise_mask, _) = batch
            noised_tokens = noised_tokens.to("cuda")
            lens = torch.tensor([len(s) for s in seq_str_list])
            start = time.perf_counter_ns()
            torch.cuda.synchronize()
            lm_pred = model(noised_tokens, repr_layers=[33])["representations"][33]
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            sparse_meltome_runtime.append((end-start)/1e9)
            seq_lens = torch.tensor([len(s) for s in seq_str_list])
            for i in range(len(seq_lens)):
                meltome_train_emebeddings.append(lm_pred[i, :seq_lens[i], :].mean(0).cpu().numpy())
sparse_meltome_runtime = np.mean(sparse_meltome_runtime)
sparse_meltome_peakmem = torch.cuda.max_memory_allocated() / 1e9

meltome_test_emebeddings = []
with torch.inference_mode():
    with torch.cuda.amp.autocast(enabled=autocast):
        for n, batch in enumerate( tqdm.tqdm(meltome_test_dl)):
            (batch_labels, seq_str_list, noised_tokens, tokens, noise_mask, _) = batch
            noised_tokens = noised_tokens.to("cuda")
            lens = torch.tensor([len(s) for s in seq_str_list])
            lm_pred = model(noised_tokens, repr_layers=[33])["representations"][33]
            seq_lens = torch.tensor([len(s) for s in seq_str_list])
            for i in range(len(seq_lens)):
                meltome_test_emebeddings.append(lm_pred[i, :seq_lens[i], :].mean(0).cpu().numpy())
##%%
with torch.inference_mode():
    with torch.cuda.amp.autocast(enabled=autocast):
         for i in range(noised_tokens.shape[0]):
            model(noised_tokens[i:(i+1)], repr_layers=[33])["representations"][33]
#%%


meltome_train_mat = np.array(meltome_train_emebeddings)
meltome_test_mat = np.array(meltome_test_emebeddings)
meltome_train_labels = meltome_data[meltome_data['set'] == 'train']['target'].values
meltome_test_labels = meltome_data[meltome_data['set'] == 'test']['target'].values[:meltome_test_mat.shape[0]]

regressor = xgb.XGBRegressor(device="cuda")
regressor.fit(meltome_train_mat, meltome_train_labels)
meltome_test_preds = regressor.predict(meltome_test_mat)

sparse_meltome_pcor = sp.stats.pearsonr(meltome_test_preds, meltome_test_labels)
#%%
output = {
    "vanilla_mlm_val_loss": all_vanilla_mlm_val_loss,
    "sparse_mlm_val_loss": all_sparse_mlm_val_loss,
    "vanilla_meltome_pcor": vanilla_meltome_pcor,
    "sparse_meltome_pcor": sparse_meltome_pcor,
    "vanilla_meltome_peakmem": vanilla_meltome_peakmem,
    "sparse_meltome_peakmem": sparse_meltome_peakmem,
    "vanilla_meltome_runtime": vanilla_meltome_train_runtime,
    "sparse_meltome_runtime": sparse_meltome_runtime

}

with open(f"/manitou/pmg/users/vss2134/HPML/project/sparse_results/global_unst24.pkl", "wb") as f:
    pickle.dump(output, f)

# %%
