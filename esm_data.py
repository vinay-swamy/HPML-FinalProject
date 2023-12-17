from pathlib import Path
import torch
import esm
from typing import Dict, Tuple, List, Union, Sequence, Optional, Iterator
import re
import math
import gzip
from functools import cached_property
import lightning.pytorch as pl
import random
import numpy as np
import zstandard
import pickle
import sqlite3
import sys


class BatchFilesSeqDataset(torch.utils.data.IterableDataset):
    '''Sample from multiple sequence files
    Note <eos> and <cls> don't correspond to N/C-terminal in ESM1
    https://github.com/facebookresearch/esm/issues/15
    However ESM2 paper <We used BOS and EOS tokens to signal the beginning and end of a real protein>
    NOTE:
        1. infinite=True means workers will loop through data
        2.  set_len = N; allow manually set dataset size (batches/ N GPUs)
            set_len = -1; __len__ return None, problematic for training when GPUs get different number of batches
            set_len = 0; most conservative estimate of number of batches; Epoch will end way before all data seen
    **current_epoch** set if you use non-permanent dataloaders,
        sets different seeds on different epoch to get different sequence of data
        in LightningDataModule use self.trainer.current_epoch
    '''

    def __init__(self,
                 alphabet=None,
                 files: Optional[List[Path]]=None,
                 sqlite_path: Optional[Path]=None,
                 cluster_sampling: bool = False,  # only for sqlite
                 rep_seq_only: bool = False,   # only for sqlite
                 toks_per_batch=1024,
                 shuffle=True,
                 rand_crop_long=True,
                 skip_long=False,
                 drop_last=False,
                 order_by_len: Optional[int] = None,
                 real_bos_eos: bool=False,  # True for ESM2
                 max_tok_length=1024,
                 unnamed_fasta: bool=True,
                 infinite: bool=False,
                 set_len: int=0,  # -1 = None, 0 = estimate, N = N
                 current_epoch: int = 0,
                 static_batch_size: Optional[int] = None,  # overrides token toks_per_batch
                 ):
        'Initialization'
        super(BatchFilesSeqDataset).__init__()
        self.shuffle = shuffle
        self.files = files
        self.sqlite_path = sqlite_path
        self.cluster_sampling = cluster_sampling
        self.rep_seq_only = rep_seq_only
        if self.files is not None:
            self.start = 0
            self.end = len(files)
            assert self.sqlite_path is None, 'Input only fasta or Sqlite db'
            assert ~self.cluster_sampling, 'cluster_sampling only works for Sqlite db'
        else:
            self.set_read_sql_connection()
            self.start = 1
            self.end = self.r_sql_maxid + 1
        self.toks_per_batch = toks_per_batch
        self.rand_crop_long = rand_crop_long
        self.alphabet = alphabet
        self.extra_toks_per_seq = int(self.alphabet.append_eos +
                                      self.alphabet.prepend_bos)
        self.real_bos_eos = real_bos_eos
        self.max_tok_length = max_tok_length
        self.unnamed_fasta = unnamed_fasta
        self.drop_last = drop_last
        if order_by_len is not None:
            assert isinstance(order_by_len, int)
        self.order_by_len = order_by_len
        if skip_long & rand_crop_long:
            raise ValueError('skip_long & rand_crop_long cannot both be true')
        self.skip_long = skip_long
        self.infinite = infinite
        self.set_len = set_len
        self.current_epoch = current_epoch
        self.static_batch_size = static_batch_size
        if self.static_batch_size is not None:
            self.toks_per_batch = float('inf')
        else:
            self.static_batch_size =  float('inf')

    def set_read_sql_connection(self):
        assert self.sqlite_path.is_file(), \
            f'sqlite db {self.sqlite_path} not exist'
        table_name = 'ClusterSeq'
        sqlite_cols = [
            "id",
            "seq",
            "clusterID",
            "rep",
        ]
        self.r_sql_con = sqlite3.connect(
            #self.precomp_sqlite_path,
            f'file:{self.sqlite_path}?immutable=1', uri=True, #mode=ro must have wal and shm
            isolation_level=None,
            check_same_thread=False,
        )
        self.r_sql_con.execute('pragma synchronous=off')
        self.r_sql_con.execute('pragma journal_mode=off')  # WAL can be ro, off pair with immutable
        cur = self.r_sql_con.execute('pragma query_only=ON')
        cur.close()
        # includes id
        if self.cluster_sampling:
            id_col = 'clusterID'
            self.r_sql_clusterid_rand_query = f'SELECT * FROM {table_name} ' + \
                'WHERE clusterID = (?) ORDER BY RANDOM() LIMIT 1'
        elif self.rep_seq_only:
            id_col = 'clusterID'
            self.r_sql_id_query_cmd = f'SELECT * FROM {table_name} WHERE rep = 1 AND {id_col} = (?)'
            self.r_sql_id_query_many = f'SELECT * FROM {table_name} WHERE rep = 1 AND {id_col} ' + 'in ({0})'
            self.r_sql_iter_cmd = f'SELECT * FROM {table_name} WHERE rep = 1 AND {id_col} >= (?) AND {id_col} < (?)'
        else:
            id_col = 'id'
            self.r_sql_id_query_cmd = f'SELECT * FROM {table_name} WHERE {id_col} = (?)'
            self.r_sql_id_query_many = f'SELECT * FROM {table_name} WHERE {id_col} ' + 'in ({0})'
            self.r_sql_iter_cmd = f'SELECT * FROM {table_name} WHERE {id_col} >= (?) AND {id_col} < (?)'

        self.r_sql_maxid = self.r_sql_con.execute(
            f'SELECT max({id_col}) FROM {table_name}').fetchone()[0]

        self.sqlite_cols = dict(zip(sqlite_cols, range(len(sqlite_cols))))
        
        self.zdict = self.r_sql_con.execute('SELECT dict FROM ZstdDict').fetchone()[0]
        self.seq_decompressor = zstandard.ZstdDecompressor(
            dict_data=zstandard.ZstdCompressionDict(self.zdict),)
        
    @cached_property
    def seq_count_from_filename(self) -> int:
        'Expects number of seqeucnes to be at end of filename'
        length = 0
        if self.unnamed_fasta:
            return None
        for file in [self.files[i] for i in range(self.start, self.end)]:
            p = Path(file)
            reg = re.compile("count=(\d+)\.")
            seq_count = reg.search(p.stem).group(1)
            if seq_count.isnumeric():
                length += int(seq_count)
            else:
                raise ValueError(f'count=<num> expected in name of files {self.files}')
        return length

    # wrong (too long) __len__ prevent validation at epoch end from being run
    def __len__(self) -> int:
        '''
        self.set_len == -1 -> None
            for validation and testing where size mismatch not fatal
        self.set_len == 0 -> int
            most conservative estimate for dataset size
        else -> int
            manual setting
        '''
        if self.set_len == -1:
            return None
        elif self.set_len == 0:
            world_size, rank = torch.utils.data.dataloader._get_distributed_settings()
            est_seq_per_batch = self.toks_per_batch // 100
            min_batch_num = self.seq_count_from_filename // world_size // est_seq_per_batch
            # print(f'min_batch_num: {min_batch_num}')
            return min_batch_num
        else:
            return self.set_len

    def _each_chunk(self, stream, separator='\n>') -> Iterator[str]:
        """
        Yield lines from `stream` until `separator`. Source: https://stackoverflow.com/a/47927374
        """
        buffer = ""
        while True:  # until EOF
            chunk = stream.read(65536).decode("utf-8")  # read 2^16 bytes
            if not chunk:  # EOF?
                yield buffer
                break
            buffer += chunk
            while True:  # until no separator is found
                try:
                    part, buffer = buffer.split(separator, 1)
                except ValueError:
                    break
                else:
                    yield part

    def _parse_file(self, fasta_file: Path) -> Tuple[List[str], List[str]]:
        seq_labels, seq_strs = [], []
        if fasta_file.suffix != '.gz':
            with open(fasta_file, 'r') as myFile:
                file_str = myFile.read()
                fastas = file_str.split("\n>")
                for fasta in fastas:
                    sequence_lines = []
                    for i, line in enumerate(fasta.split("\n")):
                        if i == 0:
                            header = line
                            if line.startswith('>'):
                                header = header[1:]
                            seq_labels.append(header)
                        else:
                            sequence_lines.append(line)
                    seq_strs.append(''.join(sequence_lines))
            return seq_labels, seq_strs

        with gzip.open(fasta_file, "rb") as myFile:
            for chunk in self._each_chunk(myFile, separator='\n>'):
                # print(chunk)  # not holding in memory, but printing chunk by chunk
                fasta = chunk.split('\n')
                header = fasta[0]
                if header.startswith('>'):
                    header = header[1:]
                header = header.split()[0]
                sequence = ''.join(fasta[1:])
                seq_labels.append(header)
                seq_strs.append(sequence)
        return seq_labels, seq_strs

    def get_lab_seq_from_fasta(self) -> Iterator[Tuple[str, str]]:
        files = [self.files[i] for i in range(self.start, self.end)]
        start_res = 0
        if len(files) == 0:
            return None
            # return iter(range(self.start, self.end))
        if self.shuffle:
            sample_gen = torch.utils.data.RandomSampler(range(len(files)),
                                                        replacement=False)
            files = [files[i] for i in sample_gen]

        for file in files:
            seq_labels, seq_strs = self._parse_file(file)
            file_lab_seqs = [*zip(seq_labels, seq_strs)]  # transpose list
            if self.shuffle:
                sample_gen = torch.utils.data.RandomSampler(
                    range(len(file_lab_seqs)), replacement=False)
                file_lab_seqs = [file_lab_seqs[i] for i in sample_gen]
            yield file_lab_seqs
    
    def get_lab_seq_from_sqlite(self) -> Iterator[Tuple[str, str]]:
        chunk_size = random.randint(15, 20) * 400  # every epoch different splits
        if self.shuffle:
            nchunk = math.ceil((self.end - self.start)/chunk_size)
            sample_idxs = torch.randperm(nchunk)
            for i in sample_idxs.tolist():
                
                chunk_start = self.start + chunk_size * i
                chunk_end = self.start + chunk_size * (i+1)
                if i == nchunk - 1:
                    chunk_end = self.end
                batch_idxs = [*range(chunk_start, chunk_end)]
                seq_labels, seq_strs = [], []
                if self.cluster_sampling:
                    for idx in batch_idxs:
                        cur = self.r_sql_con.execute(
                            self.r_sql_clusterid_rand_query, (idx,))
                        sample = cur.fetchone()
                        seq = pickle.loads(
                            self.seq_decompressor.decompress(
                                sample[self.sqlite_cols['seq']]))
                        seq_labels.append(sample[self.sqlite_cols['id']])
                        seq_strs.append(seq)
                        cur.close()
                else:
                    cur = self.r_sql_con.execute(
                        self.r_sql_id_query_many.format(
                            ', '.join('?' for _ in batch_idxs)), batch_idxs)
                    for sample in cur:
                        seq = pickle.loads(
                            self.seq_decompressor.decompress(
                                sample[self.sqlite_cols['seq']]))
                        seq_labels.append(sample[self.sqlite_cols['id']])
                        seq_strs.append(seq)
                    cur.close()

                lab_seqs = [*zip(seq_labels, seq_strs)]
                sample_gen = torch.utils.data.RandomSampler(
                    range(len(lab_seqs)), replacement=False)
                yield [lab_seqs[i] for i in sample_gen]
        else:
            in_sql_cur = self.r_sql_con.execute(self.r_sql_iter_cmd,
                                                (self.start, self.end))
            seq_labels, seq_strs = [], []
            for sample in in_sql_cur:
                seq = pickle.loads(
                    self.seq_decompressor.decompress
                    (sample[self.sqlite_cols['seq']]))
                seq_labels.append(sample[self.sqlite_cols['id']])
                seq_strs.append(seq)
                if len(seq_labels) == chunk_size:
                    lab_seqs = [*zip(seq_labels, seq_strs)]
                    seq_labels, seq_strs = [], []
                    yield lab_seqs
            in_sql_cur.close()
            if len(seq_labels) > 0:
                yield [*zip(seq_labels, seq_strs)]

    def get_sample_lab_seq(self) -> Iterator[Tuple[str, str, bool, bool]]:
        '''Read Tuple (sequence label, sequence) from file
        Crop to max_length, random starting residue or 0
        Special tokens are added even for crops:
            Joshim5 on github: https://github.com/facebookresearch/esm/issues/21
        ESM2 eos and bos correspond to C-term N-term
        returns (label, sequence, prepend, append)
        '''
        if self.sqlite_path is not None:
            iterator = self.get_lab_seq_from_sqlite()
        else:
            iterator = self.get_lab_seq_from_fasta()
        for file_lab_seqs in iterator:
            if self.order_by_len is not None:
                file_lab_seqs = self.sort_seq_by_len(file_lab_seqs)
            for (seq_label, seq_str) in file_lab_seqs:
                seq_len = len(seq_str)
                # time.sleep(0.1)
                # if seq_len > self.max_seq_length:
                if seq_len > self.max_tok_length - self.extra_toks_per_seq:
                    if self.skip_long:
                        continue
                    prepend = self.alphabet.prepend_bos
                    append = self.alphabet.append_eos
                    start_res = 0
                    if self.rand_crop_long:
                        start_res = torch.randint(
                            0,
                            seq_len - self.max_tok_length + self.extra_toks_per_seq,
                            (1,)).item()
                        if self.real_bos_eos:
                            # prepend <cls> only if starting residue 1
                            # append only when end with C-term
                            if start_res > 0:
                                prepend = False
                                if start_res + self.max_tok_length - int(append) < seq_len:
                                    # only append eos if last res + 1 == seq length
                                    append = False
                            else:
                                append = False

                    yield (seq_label,
                           seq_str[start_res:start_res + self.max_tok_length - int(prepend + append)],
                           prepend,  # Always True on ESM1, seq dependent ESM2
                           append,  # Always True ESM-1b, seq dependent ESM2
                          )
                else:
                    yield (seq_label, seq_str,
                           self.alphabet.prepend_bos, self.alphabet.append_eos)

    def sort_seq_by_len(self, file_lab_seqs: List[Tuple[str]]
                        ) -> List[Tuple[str]]:
        ''' Order sequences by len for more efficient packing
        sorted should maintain original order of same-length inputs
        so shuffle should still apply.
        Watch memory consumption on huge fastas
        '''
        buf, ordered_sorted_lab_seqs = [], []
        for s in file_lab_seqs:
            buf.append(s)
            if len(buf) == self.order_by_len:
                sorted_lab_seqs = sorted(buf, key=lambda x: len(x[1]),
                                         reverse=True)
                ordered_sorted_lab_seqs.extend(sorted_lab_seqs)
                buf = []
        if len(buf) > 0:
            sorted_lab_seqs = sorted(buf, key=lambda x: len(x[1]),
                                     reverse=True)
            ordered_sorted_lab_seqs.extend(sorted_lab_seqs)
        return ordered_sorted_lab_seqs

    def __iter__(self):
        '''Converts sample into batches
        Still dealing with sequences, which are turned into tokens by batch_converter
        **NOTE** setting infinite=True means worker will reset once data runs out
        Inspired by 
        https://github.com/facebookresearch/esm/blob/89c35a5c900e9bf90d019c4ed6518ffc62d2e5ba/esm/data.py#L253
        '''
        buf = []
        max_len = 0
        count_batches = 0
        run_gen = True
        while run_gen:
            run_gen = self.infinite
            sample_gen = self.get_sample_lab_seq()
            if sample_gen is None:
                return None
            for sample in sample_gen:
                sample_len = len(sample[1]) + int(sample[2] + sample[3])
                max_len = max(max_len, sample_len)
                next_buf_len = len(buf) + 1
                if (next_buf_len * max_len > self.toks_per_batch) | \
                        (next_buf_len > self.static_batch_size):
                    yield buf
                    buf = []
                    max_len = sample_len
                    count_batches += 1
                    # if count_batches % 10 == 0:
                    #     print(f'batch {count_batches}')
                buf.append(sample)
        if len(buf) > 0:
            # print(buf)
            if not self.drop_last:
                yield buf
        print(f'LM worker out of data. start:{self.start} end:{self.end}')
        return None


class ProtSeqBatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    Modified version from ESM
    Masking inspired by:
    https://github.com/lucidrains/protein-bert-pytorch/blob/main/protein_bert_pytorch/protein_bert_pytorch.py
    """

    def __init__(self, alphabet: esm.Alphabet,  # toks_per_batch: int = 1024,
                 masking: bool = True, mask_prob: float = 0.15,
                 simple_masking_only: bool = False,
                 quick_encode: bool = False,  # no checks for special tokens e.g. <mask>
                 device: str = 'cpu'):  # if this is in child process, must be cpu
        self.alphabet = alphabet
        self.masking = masking
        # self.toks_per_batch = toks_per_batch
        self.mask_prob = mask_prob
        self.simple_masking_only = simple_masking_only
        self.mask_innerprob_mask = 0.8
        self.mask_innerprob_subs = 0.1
        self.mask_innerprob_same = 0.1
        self.device = device
        
        self.quick_encode = quick_encode
        self.unk_tok_encode = self.alphabet.tok_to_idx['X']
        # residue_token_index: 20 residues used for random replacement masking
        #  Might consider adding ambiguous residues to get 25 tokens
        self.residue_token_index = torch.tensor(
            self.alphabet.encode('LAGVSERTIDPKQNFYMHWC'), device=device)

    def get_mask_subset_with_fraction(self, mask: torch.Tensor,
                                      prob: float) -> torch.Tensor:
        '''
        Probability for mask=True, rounds up number of residues
        Inspired by
        https://github.com/lucidrains/protein-bert-pytorch/blob/main/protein_bert_pytorch/protein_bert_pytorch.py
        but it  gives bad results when:
        prob * seq_len > num_tokens (fixed)
        '''
        batch, seq_len, device = *mask.shape, mask.device
        num_tokens = mask.sum(dim=-1, keepdim=True)
        # num_to_mask = (num_tokens * prob).ceil().type(torch.int64).squeeze(1).tolist()#.numpy().tolist()
        num_to_mask = (num_tokens * prob).floor().type(torch.int64).squeeze(1).tolist()
        # print(f'num_to_mask {num_to_mask} from {mask.sum(-1)}')
        max_masked = math.ceil(prob * num_tokens.max() + 1)#seq_len)
        sampled_indices = -torch.ones((batch, max_masked),
                                      dtype=torch.int64, device=device)  # -1
        for i in range(batch):
            rand = torch.rand((seq_len), device=device).masked_fill(~mask[i], -1e9)
            # sampled_indices are top indices padded with -1
            _, sampled_indices[i,:num_to_mask[i]] = rand.topk(num_to_mask[i], dim=-1)

        sampled_indices = sampled_indices + 1  # padding is 0 allow excess scatter to index 0
        new_mask = torch.zeros((batch, seq_len + 1), device=device)
        new_mask.scatter_(-1, sampled_indices, 1)
        return new_mask[:, 1:].bool()  # index 0 removed

    def train_masking(self, tokens: torch.Tensor,
                      mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Masking as described by ESM
        Because sum of probabilities of noise types add up to 1, treat as fraction instead
        '''
        noise_mask = self.get_mask_subset_with_fraction(mask, self.mask_prob)
        if self.simple_masking_only:
            mask_mask = noise_mask
            noised_tokens = torch.where(mask_mask, self.alphabet.mask_idx, tokens)
        else:
            mask_mask = self.get_mask_subset_with_fraction(noise_mask, self.mask_innerprob_mask)
            subs_same_mask = noise_mask * ~mask_mask  # should be 20%
            # print(f'percentage subs_same_mask {subs_same_mask.sum(-1) / mask.sum(-1)}')
            subs_mask = self.get_mask_subset_with_fraction(
                subs_same_mask,
                self.mask_innerprob_subs / (self.mask_innerprob_same+self.mask_innerprob_subs))
            noised_tokens = torch.where(mask_mask, self.alphabet.mask_idx, tokens)

            # sample with replacement
            rand_res_tokens = self.residue_token_index[
                torch.randint(len(self.residue_token_index), mask.shape)]
            noised_tokens = torch.where(subs_mask, rand_res_tokens, noised_tokens)
        return noised_tokens, noise_mask

    def __call__(self, raw_batch: Sequence[Tuple[str, str, bool, bool]]
                 ) -> Tuple[List[str], List[str], torch.Tensor,
                            torch.Tensor, torch.Tensor]:
        '''
        Returns:
            seq_label list
            seq_str list
            tokenized seq tensor
            noise_mask True for residues used for training
            mask True for all residues (exclude cls/bos, eos, padding)
        '''
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        # worker_info = torch.utils.data.get_worker_info()
        if batch_size == 0:
            # print(f'here {worker_info.id}')
            return None
        max_tok_len = 0
        seq_encoded_list, mask_lens = [], []
        for i, (_, seq_str,
                prepend_bos, append_eos) in enumerate(raw_batch):
            if self.quick_encode:
                seq_encoded = [self.alphabet.tok_to_idx.get(
                    c, self.unk_tok_encode) for c in seq_str]
            else:
                seq_encoded = self.alphabet.encode(seq_str)
            if prepend_bos:
                seq_encoded[:0] = [self.alphabet.cls_idx]
            mask_lens.append(len(seq_encoded))  # mask out eos
            if append_eos:
                seq_encoded.append(self.alphabet.eos_idx)
            if len(seq_encoded) > max_tok_len:
                max_tok_len = len(seq_encoded)
            seq_encoded_list.append(seq_encoded)

        tokens = torch.full(
            (
                batch_size,
                max_tok_len,
            ),
            fill_value=self.alphabet.padding_idx,
            dtype=torch.int64,)

        batch_labels, seq_str_list, bos_list, _ = zip(*raw_batch)  # zip returns tuples, which preserves order
        
        # mask out (False) everything longer than mask_lens
        mask = torch.arange(
            tokens.shape[1])[None, :] < (torch.tensor(mask_lens))[:, None]
        for i, b in enumerate(bos_list):
            # mask out bos token
            if b:
                mask[i, 0] = False
        
        for i, seq_encoded in enumerate(seq_encoded_list):
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i, :len(seq_encoded)] = seq  # bos and eos done above

        if self.masking:
            noised_tokens, noise_mask = self.train_masking(tokens, mask)
            return batch_labels, seq_str_list, noised_tokens, tokens, noise_mask, mask

        return batch_labels, seq_str_list, tokens, tokens, None, mask


class ESMTrainDataModule(pl.LightningDataModule):
    def __init__(self,
                 alphabet: esm.Alphabet,
                 train_path: Optional[Path] = None,  # dir | fasta | sqlite
                 valid_path: Optional[Path] = None,  # dir | fasta | sqlite
                 test_path: Optional[Path] = None,  # dir | fasta | sqlite
                 predict_path: Optional[Path] = None,
                 toks_per_batch: int = 1024,
                 batch_size_multi: int = 1,  # for pytorch lightning scale_batch_size
                 num_workers: int = 4,
                 infer_mult: int = 5,
                 real_bos_eos: bool = True,  # ESM2 True
                 mask_prob: float = 0.15,
                 valtest_mask_prob: float = 0.15,
                 valtest_simple_masking_only: bool = False,  # mask only with mask token
                 quick_encode: bool = False,
                 train_cluster_sampling: bool = False,
                 rep_seq_only: bool = False,  # same as cluster sampling, but no shuffling; doesn't apply to training
                 infinite_train: bool = False,
                 valtestpred_max_tok_length: int = 1024,
                 pin_memory: bool = False,  # this could cause issues
                 emb_ridge_eval: Optional[Dict] = None,# {task_name: {label_col, data_sqlite, alpha}}
                 current_epoch: int = 0,  # manual control over epoch
                 tm_validate_sqlite: Optional[Path] = None,
                 plm_benchmark_path: Path = Path("~/projects/plm_pl_benchmarks/plm_pl_benchmarks"),
                ):
        '''
        Load entire dataset and split into train/val
        def setup() and prepare_data() for larger datsets
        '''
        super().__init__()
        # self.batch_size = batch_size
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.alphabet = alphabet
        self.num_workers = num_workers
        self.toks_per_batch = toks_per_batch
        self.infer_mult = infer_mult
        self.real_bos_eos = real_bos_eos
        self.mask_prob = mask_prob
        self.valtest_mask_prob = valtest_mask_prob
        self.valtest_simple_masking_only = valtest_simple_masking_only
        self.prefetch_factor = 4
        self.quick_encode = quick_encode
        self.train_cluster_sampling = train_cluster_sampling
        self.rep_seq_only = rep_seq_only
        self.valtestpred_max_tok_length = valtestpred_max_tok_length
        self.batch_size_multi = batch_size_multi
        self.infinite_train = infinite_train
        self.pin_memory = pin_memory
        self.emb_ridge_eval = emb_ridge_eval
        self.current_epoch = current_epoch
        self.tm_validate_sqlite = tm_validate_sqlite
        self.plm_benchmark_path = plm_benchmark_path

        # val_dataloader order must be same here
        self.val_dataloader_names = []
        self.val_dataloader_names.append('lm')
        if self.emb_ridge_eval is not None:
            self.val_dataloader_names.extend(
                [n + '_membeval' for n in self.emb_ridge_eval['tasks']])
        if self.tm_validate_sqlite is not None:
            self.val_dataloader_names.append('tm')

    def prepare_data(self) -> None:
        'Runs before setup on main process. Do not assign state here'
        pass

    def check_input_file(self, path):
        fasta_files, sqlite_db = None, None
        if path.is_dir():
            fasta_files = list(path.glob('./*.fasta*'))
            if len(fasta_files) == 0:
                raise RuntimeError(f'{path} contain no fasta files')
        elif path.is_file():
            if path.as_posix().endswith('.fasta') | \
                    path.as_posix().endswith('.fasta.gz'):
                fasta_files = [path]
            elif path.as_posix().endswith('.sqlite3'):
                sqlite_db = path
            else:
                raise ValueError(f'{path} must end with either .fasta or .sqlite3')
        else:
            raise ValueError(f'{path} is not a file or dir')
        return fasta_files, sqlite_db
    
    def setup(self, stage: Optional[str] = None) -> None:
        '''Runs once per process/device.
        Setup datasets with train/val splits.
        Use prepare_data for Downloads and expensive 1 time tasks.
        '''
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        fasta_files, sqlite_db = self.check_input_file(self.train_path)
        epoch = 0
        if self.trainer is not None:
            epoch = self.trainer.current_epoch
        epoch = max(epoch, self.current_epoch)
        toks_per_batch = self.toks_per_batch * self.batch_size_multi
        self.train_ds = BatchFilesSeqDataset(
            alphabet=self.alphabet,
            files=fasta_files,
            sqlite_path=sqlite_db,
            toks_per_batch=toks_per_batch,
            shuffle=True,
            rand_crop_long=True,
            drop_last=True,
            order_by_len=2000,
            real_bos_eos=self.real_bos_eos,
            current_epoch=epoch,
            cluster_sampling=self.train_cluster_sampling,
            infinite=self.infinite_train,
            set_len=-1,
        )
        dl = torch.utils.data.DataLoader(
            self.train_ds,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=iter_manual_worker_init_fn,
            collate_fn=ProtSeqBatchConverter(
                self.alphabet,
                masking=True,
                mask_prob=self.mask_prob,
                quick_encode=self.quick_encode,
            ),
            batch_sampler=None,
            batch_size=None,
            pin_memory=self.pin_memory,
        )
        return dl

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        fasta_files, sqlite_db = self.check_input_file(self.valid_path)
        toks_per_batch = self.toks_per_batch * self.infer_mult * self.batch_size_multi
        self.valid_ds = BatchFilesSeqDataset(
            alphabet=self.alphabet,
            files=fasta_files,
            sqlite_path=sqlite_db,
            toks_per_batch=toks_per_batch,
            shuffle=False,
            rand_crop_long=False,
            drop_last=False,
            skip_long=True,
            order_by_len=10000,
            set_len=-1,
            rep_seq_only=self.rep_seq_only,
            max_tok_length=self.valtestpred_max_tok_length,
        )
        dl = torch.utils.data.DataLoader(
            self.valid_ds,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=iter_manual_worker_init_fn,
            collate_fn=ProtSeqBatchConverter(
                self.alphabet,
                masking=True,
                mask_prob=self.valtest_mask_prob,
                simple_masking_only=self.valtest_simple_masking_only,
                quick_encode=self.quick_encode,
            ),
            batch_sampler=None,
            batch_size=None,
            pin_memory=self.pin_memory,
        )
        if (self.emb_ridge_eval is not None) | (self.tm_validate_sqlite is not None):
            dls = [dl,]
            if self.emb_ridge_eval is not None:
                batch_size = max(toks_per_batch // 1024, 1)
                data_dir = Path(self.emb_ridge_eval['data_dir']).expanduser()
                tasks = self.emb_ridge_eval['tasks']
                for task in tasks:
                    sqlite_db = data_dir / self.emb_ridge_eval[task]['data_sqlite']
                    assert sqlite_db.is_file()
                    ds = ProteinEmbeddingData(sqlite_path=sqlite_db,
                                              label_col=self.emb_ridge_eval[task]['label_col'],
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              embedding_col1 = 'seq',
                                              comb_train_val=True,
                                             )
                    dl = torch.utils.data.DataLoader(
                        ds,
                        worker_init_fn=iter_manual_worker_init_fn,
                        collate_fn=LabelSeqBatchConverter(alphabet=self.alphabet,comb_train_val=True),
                        batch_sampler=None,
                        batch_size=None,
                        num_workers=1,#self.num_workers,
                        pin_memory=True,
                    )
                    dls.append(dl)
            if self.tm_validate_sqlite is not None:
                sys.path.append(self.plm_benchmark_path.expanduser().as_posix())
                from plm_benchmark_data import ProteinEmbeddingPairDataModule
                lit_data = ProteinEmbeddingPairDataModule(
                    label_col='dTm',
                    valid_path = self.tm_validate_sqlite,
                    batch_size = min(50, self.toks_per_batch // 1000),  # small dataset, want to make sure each GPU gets a batch
                    num_workers = 1,
                    alphabet=self.alphabet,
                    infer_mult=1,
                )
                ds, dl = lit_data.inference_dataloader_setup(self.tm_validate_sqlite)
                dls.append(dl)
            return dls
        return dl

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        fasta_files, sqlite_db = self.check_input_file(self.test_path)
        toks_per_batch = self.toks_per_batch * self.infer_mult * self.batch_size_multi
        self.test_ds = BatchFilesSeqDataset(
            alphabet=self.alphabet,
            files=fasta_files,
            sqlite_path=sqlite_db,
            toks_per_batch=toks_per_batch,  #(1024*5)4 highest for ESM1b 12GB
            shuffle=False,
            rand_crop_long=False,
            drop_last=False,
            skip_long=True,
            order_by_len=10000,
            set_len=-1,
            rep_seq_only=self.rep_seq_only,
            max_tok_length=self.valtestpred_max_tok_length,
        )
        dl = torch.utils.data.DataLoader(
            self.test_ds,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=iter_manual_worker_init_fn,
            collate_fn=ProtSeqBatchConverter(
                self.alphabet,
                masking=True,
                mask_prob=self.valtest_mask_prob,
                simple_masking_only=self.valtest_simple_masking_only,
                quick_encode=self.quick_encode,
            ),
            batch_sampler=None,
            batch_size=None,
            pin_memory=self.pin_memory,
        )
        return dl

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        fasta_files, sqlite_db = self.check_input_file(self.predict_path)
        toks_per_batch = self.toks_per_batch * self.infer_mult * self.batch_size_multi
        self.predict_ds = BatchFilesSeqDataset(
            alphabet=self.alphabet,
            files=fasta_files,
            sqlite_path=sqlite_db,
            toks_per_batch=toks_per_batch,  #(1024*5)4 highest for ESM1b 12GB
            shuffle=False,
            rand_crop_long=False,
            drop_last=False,
            skip_long=True,
            order_by_len=10000,
            set_len=-1,
            rep_seq_only=self.rep_seq_only,
            max_tok_length=self.valtestpred_max_tok_length,
        )
        dl = torch.utils.data.DataLoader(
            self.predict_ds,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=iter_manual_worker_init_fn,
            collate_fn=ProtSeqBatchConverter(
                self.alphabet,
                masking=True,
                mask_prob=self.valtest_mask_prob,
                simple_masking_only=self.valtest_simple_masking_only,
                quick_encode=self.quick_encode,
            ),
            batch_sampler=None,
            batch_size=None,
            pin_memory=self.pin_memory,
        )
        return dl


def worker_init_fn(worker_id):
    raise RuntimeError('deprecated. See iter_manual_worker_init_fn()')
    return None


def old_worker_init_fn(worker_id):
    '''Standard worker_init_fn. Does not handle seeds and Distributed.'''
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
    return None


def iter_manual_worker_init_fn(worker_id):
    '''
    For iter-style datasets with DDP support (DeepSpeed uses DDP).
    Data is divided manually by dataset.start dataset.end
    rank_id (distributetd gpus) handled manually here,
        **SO DO NOT USE DISTRIBUTEDSAMPLER**
    Sampler is not used in my iter-style dataset,
    but to be safe: set Trainer( replace_sampler_ddp=False,)
    Code for seeding is from pl_worker_init_fn() at:
    https://github.com/Lightning-AI/lightning/blob/master/src/lightning_lite/utilities/seed.py
    '''
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    current_epoch = dataset.current_epoch,  # if resetting workers
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    world_size, rank = torch.utils.data.dataloader._get_distributed_settings()
    num_workers = worker_info.num_workers
    total_workers = world_size * num_workers
    assert (overall_end - overall_start) >= total_workers, \
        "More worker than data based on start and end. May crash using DDP."
    # taking floor ensures no worker receives no data
    per_worker = int(math.floor(
        (overall_end - overall_start) / float(total_workers)))
    remainder = (overall_end - overall_start) - (total_workers * per_worker)
    
    worker_id = worker_info.id + (rank * num_workers)
    dataset.start = overall_start + worker_id * per_worker + min(worker_id, remainder)
    insert_remainder = 1 if worker_id < remainder else 0
    dataset.end = min(dataset.start + per_worker + insert_remainder, overall_end)

    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([base_seed, worker_id, rank, current_epoch])
    # print(
    #     f"Initializing random number generators of process {rank} worker {worker_id} with base seed {base_seed}",
    # )

    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(stdlib_seed)