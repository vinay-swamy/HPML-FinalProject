#%%
from pathlib import Path
import torch
import esm
import lightning.pytorch as pl
import time
import sys
import esm2
import numpy as np
import loralib as lora 
from esm_data import BatchFilesSeqDataset, ProtSeqBatchConverter, iter_manual_worker_init_fn
from deepspeed.ops.adam import DeepSpeedCPUAdam


precision = 16
flash_attention = True
# ESM variables
model_name = sys.argv[1]
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


lora_qv_rank = int(sys.argv[2])
if lora_qv_rank == -1:
    lora_qv_rank = None
model_min_args = {
    # Model args
    'attention_heads': model_args.encoder_attention_heads,
    'embed_dim': model_args.encoder_embed_dim,
    'num_layers': model_args.encoder_layers,
    'token_dropout': model_args.token_dropout,
    'flash_attention': flash_attention,
    'alphabet': alphabet,
    'lora_qv_rank': lora_qv_rank
}
model = esm2.ESM2(**model_min_args)
if lora_qv_rank is not None:
    print("Marking only lora as trainable")
    lora.mark_only_lora_as_trainable(model)

pl.seed_everything(1) 
state_dict = model.upgrade_state_dict(model_data["model"])
if flash_attention:
    state_dict = model.upgrade_state_dict_qkv_to_packed(state_dict)
regression_location = model_cache_dir / f'{model_name}-contact-regression.pt'
regression_data = torch.load(regression_location, map_location="cpu")
state_dict.update(regression_data["model"])
model.load_state_dict(state_dict, strict=False)## strict=False because of lora_qv_rank
model = model.to("cuda")
################################################data

class RandomProteinDataset:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.standard_toks = ['L','A','G','V','S','E','R','T','I','D','P','K','Q','N','F','Y','M','H','W','C','X','B','U','Z', 'O']
    def __len__(self):
        return 10000
    def __getitem__(self, seql):
        random_seq = np.random.choice(self.standard_toks, seql, replace=True)
        seq = ''.join(random_seq)
        return (seql, seq, True, True)


from torch.utils.data import Sampler

class TokenCapSampler(Sampler):
    def __init__(self,toks_per_batch, max_seql = 2048):
        self.toks_per_batch = toks_per_batch
        self.max_seql = max_seql
    def __len__(self):
        return 15 
    def __iter__(self):
        for i in range(self.__len__()):
            toks_left = self.toks_per_batch
            seql = []
            while toks_left > 0:
                new_seq= np.random.randint(1,toks_left+1, 1).tolist()[0]
                new_seq = min(new_seq, self.max_seql)
                toks_left -= new_seq
                seql.append(new_seq)
            yield seql
train_ds = RandomProteinDataset(alphabet)
bsize = float(sys.argv[3])
dl = torch.utils.data.DataLoader(
    train_ds,
    num_workers=2,
    collate_fn=ProtSeqBatchConverter(
        alphabet,
        masking=True),
    batch_sampler=TokenCapSampler(2560*bsize, max_seql=2560),
    pin_memory=True
)

#### PTL
class LitESM(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lm_loss_fn = torch.nn.CrossEntropyLoss(reduction='none',
                                            ignore_index=-100)
    def training_step(self, batch):
        (batch_labels, seq_str_list, noised_tokens, tokens, noise_mask, _) = batch
        noised_tokens = noised_tokens.to("cuda")
        lens = torch.tensor([len(s) for s in seq_str_list])
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        lm_pred = model(noised_tokens)["logits"]
        masked_tokens = tokens.masked_fill(~noise_mask, -100)
        masked_tokens = masked_tokens.to("cuda")
        loss_noreduce = self.lm_loss_fn(
        lm_pred.view(-1, 33),
        masked_tokens.view(-1)).float()
        loss = loss_noreduce.sum() / noise_mask.sum()
        return loss
    def configure_optimizers(self):
        optimizer = DeepSpeedCPUAdam(model.parameters(),  
                                     lr=0.0001, 
                                     betas=[0.9,0.98],
                                     weight_decay=float(model_args.weight_decay),
                                     eps=float(model_args.adam_eps),
                                    )
        return optimizer
trainer = pl.Trainer(
    accelerator="gpu", devices=1, strategy="deepspeed_stage_2_offload", precision=16,
    use_distributed_sampler=False, max_epochs=1
)



###############################################data
trainer.fit(LitESM(model), dl)
peak_memory = torch.cuda.max_memory_allocated()
peak_memory_gb = peak_memory / 1e9
outstr = f"{model_name},{bsize},2560,{peak_memory_gb},deepspeed_stage_2_offload"
with open("lora_results.csv", "a") as f:
    f.write(outstr+"\n")



    
        

