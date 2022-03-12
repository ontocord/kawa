import numpy as np
import threading
import ctypes
import os
import mmap
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.multiprocessing as multiprocessing
from numpy import dot as np_dot
from numpy.linalg import norm as np_norm
import random
import unittest
import copy
from collections import OrderedDict
#from numba import jit, prange
import random as _random
from collections import OrderedDict 
import sys, gc
import sqlite3
import traceback

try:
   import cPickle as pickle
except:
   import pickle


_use_torch = True
   
if True:
    _pyVer = 3
    unicode = str
    atoi = int
    def cmp(a,b):
        return (a > b) - (a < b) 
    IntType = int
    FloatType = float
    ListType = list
    StringType = str
    TupleType = tuple
    DictType = dict
    SliceType = slice
    BooleanType = bool
    LongType = int
    RangeType = range
    BytesType = bytes

try:
   import queue as Queue
except ImportError:
   import Queue



trannum = str.maketrans("0123456789", "1111111111")


import logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s : %(processName)s : %(threadName)s : %(levelname)s : %(message)s',
    level=logging.DEBUG)

class EmbeddingsManager (torch.module):
    def load_word2vec_glove_format(self, file_name, binary=True, limit=None, all_tokens=None, min_cnt=4, find_all_compounds=False, find_all_compounds_strict=False, 
                             collapse_cutoff=0.5, collapse_all_cases=True, prefer_short_tokens=False, rng_step=250000, 
                             min_alpha=0.25, max_token_size=100, max_cnt=0, unique_upper_case_tokens=("I", "AM", "May", "March")):


        def cleanup_token(token):
            token = token.replace("#", "1")
            token = token.replace("-", "_")
            token = token.replace("|", "_")
            token = token.replace("=", "_")
            #token = token.replace("+", "_")
            token = token.replace("__", "_")
            token = token.replace("__", "_")
            token = token.replace("__", "_")
            token = token.replace("__", "_")
            token = token.replace("__", "_")
            token = token.replace("__", "_")
            token = token.replace("....", "...")
            token = token.replace("....", "...")
            token = token.replace("....", "...")
            token = token.replace("....", "...")
            token = token.replace("....", "...")
            token = token.replace("....", "...")
            token = token.strip("_")
            if len(token) > 4 and token[0] in "0123456789" and token[-1] in "0123456789":
                token = token.translate(trannum)
            return token

        def add_to_new_embed_token(token, weights, new_hash, line_no, limit, all_tokens, from_token_list=False):
            if line_no > limit:
                return
            token = token.strip()
            token = cleanup_token(token)
            token2 = None
            if len(token) > max_token_size:
                token = token[:max_token_size]
            if not token or ("@" in token  and "." in token) or ".co" in token or ".org" in token or ".gov" in token or ".edu" in token or "www" in token or "http:" in token or ".net" in token or ".uk" in token or ".ca" in token:
                return
            if collapse_all_cases:
                token = token.lower()
            if not from_token_list:
                all_tokens[token.lower()] = 1
            elif not ((collapse_all_cases and token.lower() in all_tokens) or (find_all_compounds and ((len(token.split("_")[0]) > 3 and token.split("_")[0].lower() not in basic_token_to_tag and  token.split("_")[0].lower() in all_tokens) or (len(token.split("_")[-1]) > 3 and  token.split("_")[-1].lower() not in basic_token_to_tag and  token.split("_")[-1].lower() in all_tokens)))):
                return
            if find_all_compounds_strict and  (token.split("_")[0].lower() not in all_tokens or token.split("_")[0].lower() not in all_tokens):
                return
            if not collapse_all_cases:
                capitalized ="_".join([(len(w) <= 2 and w) or (len(w) > 2 and w[0].upper()+w[1:].lower()) or '' for w in token.split("_")])
                all_caps = token.upper()
                all_lower = token.lower()
                if token not in unique_upper_case_tokens and token.lower() in self.en_common_verbs:
                    # let's collapse all common verbs together into a lower
                    # case version. 
                    token2 = all_lower
                elif token not in unique_upper_case_tokens and ((all_lower in new_hash and np_cosine(weights, new_hash[all_lower][0][0]) > collapse_cutoff) or (self.has_token(all_lower) and sum(self.v(all_lower)) != 0.0 and np_cosine(weights, self.v(all_lower)) > collapse_cutoff)):
                    token2 = all_lower
                elif token not in unique_upper_case_tokens and ((capitalized in new_hash and np_cosine(weights, new_hash[capitalized][0][0]) > collapse_cutoff) or (self.has_token(capitalized) and sum(self.v(capitalized)) != 0.0 and np_cosine(weights, self.v(capitalized)) > collapse_cutoff)):
                    token2 = capitalized
                elif token not in unique_upper_case_tokens and token != token.lower() and ((all_caps in new_hash and np_cosine(weights, new_hash[all_caps][0][0]) > collapse_cutoff) or (self.has_token(all_caps) and sum(self.v(all_caps)) != 0.0 and np_cosine(weights, self.v(all_caps)) > collapse_cutoff)):
                    token2 = all_caps
                elif token not in unique_upper_case_tokens and token.upper() == token and (len(token) > 2 and (capitalized in new_hash or self.has_token(capitalized))):
                    # let's collapse upper case tokens into previous case
                    # tokens because google has a whole set of parrallel
                    # token sets that are all upper case.
                    token2 = capitalized
                elif token not in unique_upper_case_tokens and token.upper() == token and (token.lower() in new_hash or self.has_token(token.lower())):
                    token2 = all_lower
            if token == "</s>":
                token = self.pad_token
            if token in new_hash:
                new_hash[token].append([weights, max(min_cnt, vocab_size - line_no + 1)])
            else:
                if self.has_token(token):
                    id = self.token_to_id(token)
                    weights2 = self.embed_weight_data()[id]
                    if sum(weights2) != 0.0:
                        new_hash[token] = [[weights2, counts[id]]]
                        new_hash[token].append([weights, max(min_cnt, vocab_size - line_no + 1)])
                    else:
                        new_hash[token] = [[weights, max(min_cnt, vocab_size - line_no + 1)]]
                else:
                    new_hash[token] = [[weights, max(min_cnt, vocab_size - line_no + 1)]]
            if token2 is not None:
               if token2 in new_hash:
                   new_hash[token2].append([weights, max(min_cnt, vocab_size - line_no + 1)])
               else:
                   if self.has_token(token2):
                       id = self.token_to_id(token2)
                       weights2 = self.embed_weight_data()[id]
                       if sum(weights2) != 0.0:
                           new_hash[token2] = [[weights2, counts[id]]]
                           new_hash[token2].append([weights, max(min_cnt, vocab_size - line_no + 1)])
                       else:
                           new_hash[token2] = [[weights, max(min_cnt, vocab_size - line_no + 1)]]
                   else:
                       new_hash[token2] = [[weights, max(min_cnt, vocab_size - line_no + 1)]]

        def save_part(self, new_hash, max_cnt2, counts, alphas):
            logger.info("save part")
            new_hash2 = {}
            for token, arr in new_hash.items():
                y = sum([a[1] for a in arr])
                vec = sum([a[0]*float(a[1]/y) for a in arr])
                y = max([a[1] for a in arr])
                if prefer_short_tokens: 
                    if vocab_size - y <= max(0.01*vocab_size,10000):
                        y = int(math.sqrt(max(1, 5-len(token)))*y)
                if token in self.stop_words:
                    new_hash2[token] = [vec, max(min_cnt, int(2*y))]
                else:
                    new_hash2[token] = [vec, max(min_cnt, int(y/((token.count("_")+1))))]
                #logger.info(("shortened cnt", token, new_hash[token][1]))
            new_hash.clear()
            new_hash = None
            max_cnt = max(max_cnt2[0], int(max([vecCnt[1] for vecCnt in new_hash2.values()])+1))
            if self.pad_token in new_hash2:
                new_hash2[self.pad_token][1] = max_cnt
            token_embeds = list(new_hash2.items())
            do_print = True
            if _pyVer==2:
                token_embeds.sort(lambda b, a: cmp(a[1][1], b[1][1]))
            else:
                token_embeds.sort(key=lambda a: a[1][1], reverse=True)
            all_tokens = []
            for token, vecCnt in token_embeds:
                id = self.token_to_id(token)
                if id is not None:
                    cnt = max(counts[id], vecCnt[1])
                    alpha = min(alphas[id], min(1.0, max(min_alpha, math.sqrt(max(1, max_cnt - cnt + 1))/math.sqrt(max_cnt))))
                else:
                    id = -1
                    cnt = vecCnt[1]
                    alpha = min(1.0, max(min_alpha, math.sqrt(max(1, max_cnt - cnt + 1))/math.sqrt(max_cnt)))
                all_tokens.append((token, id, cnt, alpha)) # ASSUMES THIS IS THE RIGHT ORDERING

            if all_tokens:
               for a_rec in self.set_token(all_tokens, embeds=[tvc[1][0] for tvc in token_embeds]):
                  counts[a_rec[self.token_meta.ID]] = max(counts[a_rec[self.token_meta.ID]], a_rec[self.token_meta.CNT])
                  alphas[a_rec[self.token_meta.ID]] = min(alphas[a_rec[self.token_meta.ID]], a_rec[self.token_meta.ALPHA])
            all_tokens = None
            token_embeds = None
            self.flush()
            max_cnt2[0] = max(max_cnt2[0], max_cnt)

        def to_unicode(text): 
            if isinstance(text, unicode): return text
            return unicode(text, encoding='utf8', errors='replace')

        # main

        max_cnt2 = [max_cnt]
        if find_all_compounds_strict:
            find_all_compounds=True
        from_token_list=False
        if not all_tokens:
            all_tokens = {}
        else:
            from_token_list=True
        new_hash = {}
        fin =  open(file_name, 'rb')
        header = to_unicode(fin.readline())
        orig_vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
        vocab_size = orig_vocab_size
        if not limit:
            limit = vocab_size
        counts = [0]*(vocab_size+self.len_()+1000)
        alphas = [1.]*(vocab_size+self.len_()+1000)
        for id, cnt, alpha in self.get_token_iter(field=self.token_meta.ID, field2=self.token_meta.CNT, field3=self.token_meta.ALPHA):
            counts[id] = cnt
            alphas[id] = alpha
        workers = []
        if binary:
            binary_len = np.dtype(np.float32).itemsize * vector_size
            for line_no in range(orig_vocab_size):
                token = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n': 
                        token.append(ch)
                token = to_unicode(b''.join(token))
                weights = np.fromstring(fin.read(binary_len), dtype=np.float32) # vector_size
                if vector_size < self.embed.embedding_dim:
                   a = int(self.embed.embedding_dim/vector_size)
                   b = self.embed.embedding_dim%vector_size
                   weights = np.concatenate([weights]*a + [weights[:b]])
                elif vector_size > self.embed.embedding_dim:
                   weights = weights[:self.embed.embedding_dim]
                if sum(weights) != 0:
                    add_to_new_embed_token(token, weights, new_hash, line_no, limit, all_tokens, from_token_list=from_token_list)
                    if line_no >= limit-1:
                        if  not collapse_all_cases and not find_all_compounds:
                            break
                        else:
                            limit = orig_vocab_size
                            from_token_list = True
                        if  find_all_compounds_strict:
                            find_all_compounds=False
                            find_all_compounds_strict=False
                if (line_no + 1) % rng_step == 0:
                    worker = threading.Thread(target=save_part, args=(self, new_hash, max_cnt2, counts, alphas))
                    new_hash = {}
                    workers.append(worker)
                    worker.start()
                if line_no >= limit-1:
                    break
        else:
            for line_no in range(vocab_size):
                line = fin.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = to_unicode(line.rstrip()).split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % line_no)
                token, weights = parts[0], np.array([np.float32(x) for x in parts[1:]])
                if sum(weights) != 0:
                    add_to_new_embed_token(token, weights, new_hash, line_no, limit, all_tokens, from_token_list=from_token_list)
                    if line_no >= limit-1:
                        if  not collapse_all_cases and not find_all_compounds:
                            break
                        else:
                            limit = orig_vocab_size
                            from_token_list = True
                        if  find_all_compounds_strict:
                            find_all_compounds=False
                            find_all_compounds_strict=False
                if (line_no + 1) % rng_step == 0:
                    worker = threading.Thread(target=save_part, args=(self, new_hash, max_cnt2, counts, alphas))
                    new_hash = {}
                    workers.append(worker)
                    worker.start()
                if line_no >= limit-1:
                    break

        if new_hash:
           worker = threading.Thread(target=save_part, args=(self, new_hash, max_cnt2, counts, alphas))
           new_hash = {}
           worker.start()
           workers.append(worker)
        for worker in workers:
           worker.join()
        logger.info("done word2vec_glove")
        self.save()
        return self

def one():
   return 1

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    We specifically avoids mapping to whitespace/control characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

def gpt2_txt(token):
   text = bytearray([byte_decoder.get(c,ord('*')) for c in token]).decode("utf-8", errors="replace")
   
    #if text[0] not in ("<", " "):
    #   text = "##"+text.strip()
    #else:
    #   text = text.strip()
   return text
