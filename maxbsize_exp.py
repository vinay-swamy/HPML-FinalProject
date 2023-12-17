from pathlib import Path
import torch
import esm
import lightning.pytorch as pl
import time
import sys
import esm2
import numpy as np

## input arugments 
## 1 - whether or not to use flash attention - "use_fa" to enable flash attention, anhything else to not 
## 2 - which ESM model string to use. See ESM website for reference 
## 3 - value to scale the total number of tokens by. 
## 4 - maximum sequence length when drawing random sequences.
precision = 16
flash_attention = sys.argv[1] == "use_fa"

# ESM variables
model_name = sys.argv[2]
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
model = esm2.ESM2(**model_min_args)
pl.seed_everything(1) 
state_dict = model.upgrade_state_dict(model_data["model"])
if flash_attention:
    state_dict = model.upgrade_state_dict_qkv_to_packed(state_dict)
regression_location = model_cache_dir / f'{model_name}-contact-regression.pt'
regression_data = torch.load(regression_location, map_location="cpu")
state_dict.update(regression_data["model"])
model.load_state_dict(state_dict, strict=True)
model = model.to("cuda")


### Becasue we are just measuring performance, we can use a dataset of random proteins
## to get a sense of the memory usage

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

### given a total number of tokens, sample a batch of sequences
### with a specified maximum sequence length
### basically just samples sequence, then remove the lenght from the total 


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
collater = alphabet.get_batch_converter()



bsize = float(sys.argv[3])
max_seql = int(sys.argv[4])
last_run_time = []
#torch.cuda.memory_profiling_start()
dl = torch.utils.data.DataLoader(RandomProteinDataset(alphabet), batch_sampler=TokenCapSampler(2560*bsize, max_seql =int(sys.argv[4])), num_workers=0, collate_fn = collater)
batches = (b for b in dl)
seq_lens,_,tokens = next(batches)
current_run_time = []
# prof = torch.profiler.profile(
#     schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=0),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profiler_logs/prof_with_fa.log'),
#     profile_memory=True,
#     record_shapes=True,
#     with_stack=True)
# prof.start()
with torch.no_grad():
    model.eval()
    with torch.cuda.amp.autocast(enabled=autocast):
        for batch in dl:
            #prof.step()
            lengths,_,tokens = batch
            tokens = tokens.to("cuda")
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            out = model(tokens)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            current_run_time.append((end-start)/1e9)
#prof.stop()
peak_memory = torch.cuda.max_memory_allocated()
peak_memory_gb = peak_memory / 1e9
print(f"{model_name},{bsize},{max_seql},{peak_memory_gb},{np.mean(current_run_time)}")



    
        

