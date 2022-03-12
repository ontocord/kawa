# coding=utf-8
# Copyright, 2021-2022 Ontocord, LLC, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Un3less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#from random import sample
import glob, os, re
import multiprocessing

import gzip
import os, argparse
import itertools
from collections import Counter, OrderedDict
import os
import json
import threading
import numpy as np
import os
import time
import json
import copy

from time import time
import numpy as np
from collections import Counter
from itertools import chain
import glob
import json
import math, os
import random
import transformers
import sys, os
import json
import faker
import gzip
from faker.providers import person, job
from tqdm import tqdm
from collections import Counter
import re
import gzip
import urllib
import re
from transformers import AutoTokenizer
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s : %(processName)s : %(levelname)s : %(message)s',
    level=logging.INFO)

try:
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           os.path.pardir)))
except:
  sys.path.append(os.path.abspath(os.path.join("./",
                                           os.path.pardir)))
import default_onto_tags
from stopwords import stopwords
from cjk import *
mt5_underscore = "▁"
trannum = str.maketrans("0123456789", "1111111111")

try:
  onto_dir = os.path.dirname(__file__)
except:
  onto_dir = "./"

from ontology_manager import OntologyManager

class OntoMetada:
   """
   Metadata info for accessing a sqlite table storing this KB.
   """

   # TODO - modify to use Sqlalchemy or another ORM system
   def __init__(self, word_table_name):
      self.TABLE_NAME = word_table_name

      # the location of the fields
      self.word = 0
      self.ID = 1
      self.CNT = 2
      self.ALPHA = 3 
      self.TYPE = 4
      self.LEVEL = 5
      self.COMPOUND_WORD = 6
      self.PARENTS = 7
      self.CHILDREN = 8
      self.SYNONYMS = 9
      self.DEF = 10
      self.SIM = 11
      self.SUBTYPE = 12
      self.RECNO = 13
      self.REL_CUTOFF = 14

      # labels for the fields

      self.word_LABEL = self.TABLE_NAME+".word"
      self.ID_LABEL = self.TABLE_NAME+".id"
      self.CNT_LABEL = self.TABLE_NAME+".cnt"
      self.ALPHA_LABEL = self.TABLE_NAME+".alpha"
      self.TYPE_LABEL = self.TABLE_NAME+".type"
      self.LEVEL_LABEL = self.TABLE_NAME+".level"
      self.COMPOUND_WORD_COUNT_LABEL = self.TABLE_NAME+".compound_word_count"
      self.PARENTS_LABEL = self.TABLE_NAME+".parents"
      self.CHILDREN_LABEL = self.TABLE_NAME+".children"
      self.SYNONYMS_LABEL = self.TABLE_NAME+".synonyms"
      self.DEF_LABEL = self.TABLE_NAME+".def"
      self.SIM_LABEL = self.TABLE_NAME+".sim"
      self.SUBTYPE_LABEL = self.TABLE_NAME+".subtype"
      self.RECNO_LABEL = self.TABLE_NAME+".recno"
      self.REL_CUTOFF_LABEL = self.TABLE_NAME+".rel_cutoff"


      # other metadata
      self.LEN = 15
      self.FIELD_TO_LABEL = [self.word_LABEL, self.ID_LABEL, self.CNT_LABEL, self.ALPHA_LABEL, self.TYPE_LABEL, self.LEVEL_LABEL, self.COMPOUND_WORD_COUNT_LABEL, self.PARENTS_LABEL, self.CHILDREN_LABEL, self.SYNONYMS_LABEL, self.DEF_LABEL, self.SIM_LABEL, self.SUBTYPE_LABEL, self.RECNO_LABEL, self.REL_CUTOFF_LABEL]

      self.word_SIZE = 100
      self.LEVEL_MAX = 9
      self.PARENTS_SIZE = self.LEVEL_MAX-1
      self.CHILDREN_SIZE = 10
      self.SYNONYMS_SIZE = 5
      self.DEF_SIZE = 5
      self.SIM_SIZE = 5
      
      self.ID_ARRAY_FIELDS = (self.PARENTS, self.CHILDREN, self.SYNONYMS, self.DEF, self.SIM)
      self.ID_ARRAY_FIELDS_SIZE = [0, 0, 0, 0, 0, 0, 0, self.PARENTS_SIZE, self.CHILDREN_SIZE,
                                   self.SYNONYMS_SIZE, self.DEF_SIZE, self.SIM_SIZE, 0, 0, 0]

      # 0 is reserved for deleted words. has to be between 0-9
      self.TYPE_DELETED = 0
      self.TYPE_REC = 1
      self.TYPE_REL = 2
      self.TYPE_REL3 = 3
      self.TYPE_REL4 = 4
      self.TYPE_REL5 = 5

      # TODO - modify table if the metada is different
      self.CREATE_TABLE_STMT =  "CREATE TABLE IF NOT EXISTS "+self.TABLE_NAME+" ( "
      self.CREATE_TABLE_STMT += self.word_LABEL+" TEXT," # VARCAHR ("+str(word_SIZE)+"), "
      self.CREATE_TABLE_STMT += self.ID_LABEL+" INTEGER PRIMARY KEY, "
      self.CREATE_TABLE_STMT += self.CNT_LABEL+" INTEGER DEFAULT 0, "
      self.CREATE_TABLE_STMT += self.ALPHA_LABEL+" FLOAT DEFAULT 1.0, "
      self.CREATE_TABLE_STMT += self.TYPE_LABEL+" INTEGER DEFAULT 0, "
      self.CREATE_TABLE_STMT += self.LEVEL_LABEL+" INTEGER DEFAULT 0, "
      self.CREATE_TABLE_STMT += self.COMPOUND_WORD_COUNT_LABEL+" INTEGER DEFAULT 0, "
      self.CREATE_TABLE_STMT += self.PARENTS_LABEL+" BLOB, "
      self.CREATE_TABLE_STMT += self.CHILDREN_LABEL+" BLOB, "
      self.CREATE_TABLE_STMT += self.SYNONYMS_LABEL+" BLOB, "
      self.CREATE_TABLE_STMT += self.DEF_LABEL+" BLOB, "
      self.CREATE_TABLE_STMT += self.SIM_LABEL+" BLOB, "
      self.CREATE_TABLE_STMT += self.SUBTYPE_LABEL+" INTEGER DEFAULT 0, "
      self.CREATE_TABLE_STMT += self.RECNO_LABEL+" INTEGER DEFAULT 0, "
      self.CREATE_TABLE_STMT += self.REL_CUTOFF_LABEL+" FLOAT INTEGER DEFAULT 0.0)"
      self.CREATE_TABLE_STMT = self.CREATE_TABLE_STMT.replace(self.TABLE_NAME+".", "")

      self.UPDATE_STMT =  "UPDATE "+self.TABLE_NAME+" SET "
      self.UPDATE_STMT += self.word_LABEL+"  = ?, "
      self.UPDATE_STMT += self.ID_LABEL+" = ?, "
      self.UPDATE_STMT += self.CNT_LABEL+" = ?, "
      self.UPDATE_STMT += self.ALPHA_LABEL+" = ?, "
      self.UPDATE_STMT += self.TYPE_LABEL+" = ?, "
      self.UPDATE_STMT += self.LEVEL_LABEL+" = ?, "
      self.UPDATE_STMT += self.COMPOUND_WORD_COUNT_LABEL+" = ?, "
      self.UPDATE_STMT += self.PARENTS_LABEL+" = ?, "
      self.UPDATE_STMT += self.CHILDREN_LABEL+" = ?, "
      self.UPDATE_STMT += self.SYNONYMS_LABEL+" = ?, "
      self.UPDATE_STMT += self.DEF_LABEL+" = ?, "
      self.UPDATE_STMT += self.SIM_LABEL+" = ?, "
      self.UPDATE_STMT += self.SUBTYPE_LABEL+" = ?, "
      self.UPDATE_STMT += self.RECNO_LABEL+" = ?, "
      self.UPDATE_STMT += self.REL_CUTOFF_LABEL+" = ? "
      self.UPDATE_STMT += "WHERE "+self.ID_LABEL+" = ?"
      self.UPDATE_STMT = self.UPDATE_STMT.replace(self.TABLE_NAME+".", "")
      
      self.INSERT_STMT = "INSERT INTO "+self.TABLE_NAME+" VALUES (" + ("?,"*(self.LEN-1))+"?)"
        

class OntoKB (OntologyManager):
    """
   A class for accessing a hiearhical KB similar to wordnet for words that are tied to an embedding.
   
   Similar to wordnet, each word may be one of several senses with it's own embeddings. 
   
   Unlike wordnet, a "word" may be a pattern, such as the shingles created by the OntologyManager. In this way, geo*_bu* could be Goerge Bush or George Burns,
   and if the embeddings for the two words are different enough, the will be stored as geo*_bu*#1 and geo*_bu*#2.  
   
   OntoKB can used to combine embeddings from various sources such as word2vec, glove, and the embeddings from pre-trained transformer models into the SAME embedding space. 
   When combining with VERY large vocabularies like word2vec and glove, instead of storing away a specific word, we use the pattern matched by the OntologyManager as set forth above.

   The Algorithm to combine embeddings:

   We base the OntoKB embeddings  (target) on an initial model (e.g., a sentence transformer embeddings) seeded by using extract_word_embeddings.py. 

   We combine with other emebddings (src) as follows:
   - Given all tokens that intersect between the src and target
   - find the "parent" embeddings in the src space. This is essentially a random selection of at least M >= the sqrt(vocab size) that roughly represents cluster heads.
   - M could be set to 3000 for example, which could accomodate 9M words. 
   
   Compute the weighted sum of the M vectors representing each new word from src we want to project to target. 


   For words that aren't in the current OntoKB, just add them. For words that already are, combine as a weighted sum. 
   If the storage dim of the OntoKB is less then the intial model embeddigns (e.g., OntoKB stores in halftensor of dim 100. Initial model stores in float tensor of 512), we create a 
   projection matrix, which could be initialized as random or fine tuned to preserve separation of word vectors.
    """

    # SUBTYPES FOR RELS
    REL_UNKNOWN = 0
    REL_REC = 1 # only for level 0 rels
    REL_ORPHAN = 2 # only for level 0 rels
    REL_SYN = 3
    REL_ORTH = 4
    REL_INTER = 5
    REL_TRAN = 6
    REL_RULE = 7
    REL_FRAME = 8
    REL_SCRIPT = 9 

    KB_ID = 0
    KB_LEVEL = 1
    KB_REL_START = 2
    LEN_KB = 200

    def __init__(self, embed_or_filename, target_lang="", data_dir=None, tmp_dir=None, compound_word_step=3,
                 strip_chars=None, \
                 upper_ontology=None, ontology_file="ontology.json.gz",
                 target_lang_data_file=None, word2ner_file=None, \
                 connector="_", label2label=None,  \
                tag_type={'PERSON', 'PUBLIC_FIGURE', }, ontology=None, **kwargs):
      
        OntologyManager.__init__(self,self, embed_or_filename, target_lang=target_lang, data_dir=data_dir, tmp_dir=tmp_dir, compound_word_step=compound_word_step,
                 strip_chars=strip_chars, \
                 upper_ontology=upper_ontology, ontology_file=ontology_file,
                 target_lang_data_file=target_lang_data_file, word2ner_file=word2ner_file, \
                 connector=connector, label2label=label2label,  \
                tag_type=tag_type, ontology=ontology, )
     
        tokenizer = kwargs.get('tokenizer')
        model_name = kwargs.get('model_name')
        stop_words = kwargs.get('stop_words')
        word_table_name = kwargs.get('word_table_name', "word")
        embedding_dim = kwargs.get('embedding_dim')        
        load_from_model = kwargs.get('load_from_model')  
        path = None
        self.embed = None
        self.model = None
        if isinstance(embed_or_filename, StringType):
           path = embed_or_filename # .rsplit("/")[0] # rsplit(".",1)[0]+".db"
           if not os.path.exists(path):
              os.mkdir(path)
           self.path = path
           if os.path.exists(path+"/pytorch_model.bin"):
              self.model = AutoModel.from_pretrained(path)
              self.embed = model.get_input_embeddings()
           elif os.path.exists(path+"/embedding.bin"):
              self.embed = torch.load(path+"/embedding.bin")
        else:
           self.embed = embed_or_filename
           path = self.embed.filename.rsplit("/",1)[0]
        self.path = path
        self.kb_filename = self.path+"/kb.db"
        if self.embed is not None and hasattr(self.embed, 'model_name'):
           model_name = self.embed.model_name
        #logger.info(("MODEL NAME", model_name))
        if load_from_model:
           self.model = model = AutoModel.from_pretrained(model_name)
           embedding = model.get_input_embeddings()
           embedding_dim = embedding.embedding_dim
           if isinstance(embedding, OntoEmbedding):
              self.embed = embed = embedding
        if embedding_dim is None:
           embedding_dim = 300
        if self.embed is None:
           self.embed = np_memmap(self.path+"/embedding.mmap", shape=(-1, embedding_dim))
        self.embed.model_name = model_name
        self.tlock = self.embed_weight_data().tlock or DummyLock()
        if os.path.exists(self.path):
           logger.info(self.path)
           tokenizer = Autotokenizer.from_pretrained(self.path)
        if tokenizer is None:
           tokenizer = Autotokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer
        if hasattr(tokenizer, 'bpe_ranks'):
           self.bpe_ranks = tokenizer.bpe_ranks
        else:
           self.bpe_ranks = {}
        self.encoder = tokenizer.added_words_encoder
        self.decoder = tokenizer.added_words_decoder
        if hasattr(tokenizer, 'encoder'):
            self.base_encoder = tokenizer.encoder
        elif hasattr(tokenizer, 'vocab'):
            self.base_encoder = tokenizer.vocab
        else:
            self.base_encoder = tokenizer.added_words_encoder
        if hasattr(tokenizer, 'decoder'):
            self.base_decoder = tokenizer.decoder
        elif hasattr(tokenizer, 'ids_to_word'):
            self.base_decoder = tokenizer.ids_to_word
        else:
            self.base_decoder = tokenizer.added_words_decoder
        self.unk_word_id = self.tokenizer.unk_word_id
        self.unk_word = self.tokenizer.unk_word
        self.pad_word = self.tokenizer.pad_word
        self.view_cnt = 0
        if stop_words is None:
            stop_words = self.en_stop_words
        self.word_meta = OntoMetada(word_table_name)
        conn = self.conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.text_factory = str
        conn.execute('PRAGMA journal_mode = WAL')         
        conn.commit()
        self.sql_execute(self.word_meta.CREATE_TABLE_STMT, commit=True)
        with self.get_lock():
            max_id = len(self.tokenizer)-1
            max_id2 = self.max_id()
            max_len = max(max_id2, max_id)+1
            if self.embed.shape[0] < max_len:
                self.resize(max_len)
            if max_id2 != max_id:
               logger.info((max_id, max_id2))
               self.mirror_kb_and_tokenizer(max_id2==-1)
        if load_from_model and embedding is not None and self.embed is not embedding:
           self.embed.copy_into_view(embedding)

    def get_model(self):
       if self.model is None:
          return self.embed.clone().set_view(local_map, flush_type=flush_type, device=model.device)
       else:
          return self.model

    def close(self):
       self.embed.del_all_views()
       self.save()
       self.conn.close()

    def __del__(self):
        self.close()
        s = super(self)
        if hasattr(s, '__del__'): s.__del__()

    def save(self):
       with self.get_lock():
          self.flush()
          self.tokenizer.save_pretrained(self.path)
          if self.model is not None:
             self.model.save_pretrained(self.path)
          else:
             torch.save(self.embed, self.embed.filename.replace(".mmap", ".bin"))

    def sql_execute(self, stmts, args=None, fetch=False, commit=False, many=False):
       with self.get_lock():
          ret = None
          if True:
             cursor = self.conn.cursor()
             #cursor.execute('PRAGMA synchronous=OFF')
             if args is None:
                if many:
                   raise RuntimeError("trying to executemany with no args")
                if fetch:
                   ret = list(cursor.execute(stmts).fetchall())           
                else:
                   cursor.execute(stmts)
             else:
                if fetch:
                   if many:
                      raise RuntimeError("trying to fetch data for executemany")
                   ret = list(cursor.execute(stmts, args).fetchall())
                else:
                   if many:
                      cursor.executemany(stmts, args)
                   else:
                      cursor.execute(stmts, args)
             if commit:
                self.conn.commit()
             c = None
             conn = None
          return ret

    def get_lock(self):
        return self.tlock

    def has_word(self, word):
        if type(word) is BytesType:
            word = word.decode('utf-8')
        ret = word in self.encoder or word in self.base_encoder
        if not ret:
            ret = self.tokenizer.convert_words_to_ids(word)
            if ret == self.unk_word_id and word != self.unk_word:
                ret = None
        return ret

    def word_to_id(self, word, id=None): 
        if type(word) is BytesType:
            word = word.decode('utf-8')
        ret = self.encoder.get(word)
        if ret is None:
            ret = self.base_encoder.get(word)
            if ret is None:
                ret = self.tokenizer.convert_words_to_ids(word)
                if ret == self.unk_word_id and word != self.unk_word:
                    ret = None
                if ret is None:
                    ret = id
        return ret

    def id_to_word(self, id, word=None):
        ret = self.decoder.get(id)
        if ret is None:
            ret = self.base_decoder.get(id)
            if ret is None:
                ret = self.tokenizer.convert_ids_to_words(id)
                if ret == self.unk_word and id != self.unk_word_id:
                        ret = None
                if ret is None:
                    ret = word
        return ret
    
    def word_exists(self, id_or_rec):
        if self.is_word(id_or_rec):
            return id_or_rec[self.word_meta.CNT] > 0
        else:
            if id_or_rec < 0:
                id_or_rec += self.len_()
            return self.id_to_word(id_or_rec) is not None

    def embed_weight_data(self):
        return self.embed.get_view()


    def new_word(self, copy_rec=None):
        if copy_rec is not None:
            return copy.deepcopy(copy_rec)
        ret = [None]*self.word_meta.LEN
        ret[self.word_meta.word] = ''
        ret[self.word_meta.ID] = -1
        ret[self.word_meta.CNT] = -1
        for field in self.word_meta.ID_ARRAY_FIELDS:
            ret[field] = []
        return ret
                
    def is_word(self, a_rec):
        return (type(a_rec) in (ListType, TupleType) and len(a_rec) == self.word_meta.LEN) and type(a_rec[self.word_meta.word]) in (BytesType, StringType)

    def v(self, word):
        return self.embed_weight_data()[self.word_to_id(word)]

    def max_id(self):
       ret = self.sql_execute("SELECT MAX("+self.word_meta.ID_LABEL+") FROM "+self.word_meta.TABLE_NAME, fetch=True)
       if ret[0][0] is None:
          return -1
       return ret[0][0]

    def kb_item_exists(self, id):
       ret = self.sql_execute("SELECT EXISTS(SELECT 1 FROM "+self.word_meta.TABLE_NAME+" WHERE "+self.word_meta.ID_LABEL+"=? LIMIT 1)", [id], fethch=True)
       return ret[0][0]
                
    def get_word(self, id_or_word=None, field=None, field2=None, field3=None, word_type=None):
        """
        returns a field, tuple of fields or a word. id_or_word can be an int or a string word. word can be of the form, dog#animal
        """
        id = None
        a_rec = None
        if type(id_or_word) in (np.int32, np.int64, np.uint32, np.uint64, IntType):
            id = int(id_or_word)
            if field3 == None and field in (self.word_meta.word, self.word_meta.ID) and field2 in (self.word_meta.word, self.word_meta.ID, None):
                if field2 is not None:
                    if field == self.word_meta.word:
                        return (self.id_to_word(id), id)
                    else:
                        return (id, self.id_to_word(id))
                elif field == self.word_meta.word:
                    return self.id_to_word(id)
                else:
                    return id

        if type(id_or_word) is StringType:
            word = id_or_word
            id = self.word_to_id(word)
            if id is None:
                id = self.word_to_id(word.translate(trannum))

        #if not self.word_exists(id):
        #    return None

        if field3 is not None:
            stmt = "SELECT "+self.word_meta.FIELD_TO_LABEL[field] + "," + \
                self.word_meta.FIELD_TO_LABEL[field2] + "," + \
                self.word_meta.FIELD_TO_LABEL[field3] + \
                " FROM "+self.word_meta.TABLE_NAME+" WHERE "+self.word_meta.ID_LABEL+" = ? "
            val = [id]
        elif field2 is not None:
            stmt = "SELECT "+self.word_meta.FIELD_TO_LABEL[field] + "," + \
                self.word_meta.FIELD_TO_LABEL[field2] + \
                " FROM "+self.word_meta.TABLE_NAME+" WHERE "+self.word_meta.ID_LABEL+" = ? "
            val = [id]
        elif field is not None:
            stmt = "SELECT "+self.word_meta.FIELD_TO_LABEL[field] + \
                " FROM "+self.word_meta.TABLE_NAME+" WHERE "+self.word_meta.ID_LABEL+" = ? "
            val = [id]
        else:
            stmt = "SELECT * FROM "+self.word_meta.TABLE_NAME+" WHERE "+self.word_meta.ID_LABEL+" = ?"
            val = [id]

        if word_type is not None and type(word_type) is IntType:
            stmt += " AND "+self.word_meta.TYPE_LABEL+" = ? "
            val.append(word_type)
        elif word_type is not None and type(word_type) is ListType:
            stmt += " AND "+self.word_meta.TYPE_LABEL+" in (" + ("?,"*len(word_type)-1)+"?) "
            val.extend(word_type)

        if field is not None:
            dat =  self.sql_execute(stmt, val, fetch=True)[0]
            if dat is None:
                return None
            dat = list(dat)
            if field == self.word_meta.word:
                if type(dat[0]) is BytesType:
                    dat[0] = dat[0].decode('utf-8')
                dat[0] = str(dat[0])
            if field2 == self.word_meta.word:
                if type(dat[1]) is BytesType:
                    dat[1] = dat[1].decode('utf-8')
                dat[1] = str(dat[1])
            if field3 == self.word_meta.word:
                if type(dat[2]) is BytesType:
                    dat[2] = dat[2].decode('utf-8')
                dat[2] = str(dat[2])
            if field in self.word_meta.ID_ARRAY_FIELDS:
                if dat[0]: 
                    dat[0] = pickle.loads(bytes(dat[0]))
                else:
                    dat[0] = None
            if field2 in self.word_meta.ID_ARRAY_FIELDS:
                if dat[1]: 
                    dat[1] = pickle.loads(bytes(dat[1]))
                else:
                    dat[1] = None
            if field3 in self.word_meta.ID_ARRAY_FIELDS:
                if dat[2]: 
                    dat[2] = pickle.loads(bytes(dat[2]))
                else:
                    dat[2] = None
            return tuple(dat)
        a_rec = self.sql_execute(stmt, val, fetch=True)[0]
        a_rec = list(a_rec)
        if a_rec:
            if type(a_rec[self.word_meta.word]) is BytesType: 
                a_rec[self.word_meta.word] = a_rec[self.word_meta.word].decode('utf-8')
            a_rec[self.word_meta.word] = str(a_rec[self.word_meta.word])
            for f in self.word_meta.ID_ARRAY_FIELDS:
                if a_rec[f]: 
                    a_rec[f] = pickle.loads(bytes(a_rec[f]))
                else:
                    a_rec[f] = None                        
            return a_rec
        return None
            
    def get_word_iter(self, ids=None, field=None, field2=None, field3=None, word_type=None, reverse=False):
        """
        returns a field, tuple of fields or a rec. id_or_word can be an int or a string word. word can be of the form, dog#animal
        """
        rng_step=999
        if reverse and ids is not None:
           raise RuntimeError("Can only reverse when getting all of the kb")
        if ids is None:
            ids = list(range(self.len_()))
        else:
            ids = list(ids)
        if not ids:
           return 
        len_items = len(ids)
        if reverse:
           ids = list(reversed(ids))
        if field3 == None and field in (self.word_meta.word, self.word_meta.ID) and field2 in (self.word_meta.word, self.word_meta.ID, None):
            for id in ids:
                if field2 is not None:
                    if field == self.word_meta.word:
                        yield (self.id_to_word(id), id)
                    else:
                        yield (id, self.id_to_word(id))
                elif field == self.word_meta.ID:
                    yield self.id_to_word(id)
                else:
                    yield id
            return

        for rng in range(0, len_items, rng_step):
            max_rng = min(rng + rng_step, len_items)
            if field3 is not None:
                stmt = "SELECT "+self.word_meta.FIELD_TO_LABEL[field] + "," + \
                    self.word_meta.FIELD_TO_LABEL[field2] + "," + \
                    self.word_meta.FIELD_TO_LABEL[field3] + \
                    " FROM "+self.word_meta.TABLE_NAME+" WHERE "+self.word_meta.ID_LABEL+" in ("+("?,"*(max_rng-rng-1))+"?)"
                val = ids[rng:max_rng]
            elif field2 is not None:
                stmt = "SELECT "+self.word_meta.FIELD_TO_LABEL[field] + "," + \
                    self.word_meta.FIELD_TO_LABEL[field2] + \
                    " FROM "+self.word_meta.TABLE_NAME+" WHERE "+self.word_meta.ID_LABEL+" in ("+("?,"*(max_rng-rng-1))+"?)"
                val = ids[rng:max_rng]
            elif field is not None:
                stmt = "SELECT "+self.word_meta.FIELD_TO_LABEL[field] + \
                    " FROM "+self.word_meta.TABLE_NAME+" WHERE "+self.word_meta.ID_LABEL+" in ("+("?,"*(max_rng-rng-1))+"?)"
                val = ids[rng:max_rng]
            else:
                stmt = "SELECT * FROM "+self.word_meta.TABLE_NAME+" WHERE "+self.word_meta.ID_LABEL+" in ("+("?,"*(max_rng-rng-1))+"?)"
                val = ids[rng:max_rng]

            if word_type is not None and type(word_type) is IntType:
                stmt += " AND "+self.word_meta.TYPE_LABEL+" = ? "
                val.append(word_type)
            elif word_type is not None and type(word_type) is ListType:
                stmt += " AND "+self.word_meta.TYPE_LABEL+" in (" + ("?,"*len(word_type)-1)+"?) "
                val.extend(word_type)
            if field is not None:
                dat =  self.sql_execute(stmt, val, fetch=True)
                if reverse:
                   dat = reversed(dat)
                if not dat:
                    return
                for dat2 in dat:
                    dat2 = list(dat2)
                    if field == self.word_meta.word:
                        if type(dat2[0]) is BytesType:
                            dat2[0] = dat2[0].decode('utf-8')
                        dat2[0] = str(dat2[0])
                    if field2 == self.word_meta.word:
                        if type(dat2[1]) is BytesType:
                            dat2[1] = dat2[1].decode('utf-8')
                        dat2[1] = str(dat2[1])
                    if field3 == self.word_meta.word:
                        if type(dat2[2]) is BytesType:
                            dat2[2] = dat2[2].decode('utf-8')
                        dat2[2] = str(dat2[2])
                    if field in self.word_meta.ID_ARRAY_FIELDS:
                        if dat2[0]: 
                            dat2[0] = pickle.loads(bytes(dat2[0]))
                        else:
                            dat2[0] = None
                    if field2 in self.word_meta.ID_ARRAY_FIELDS:
                        if dat2[1]: 
                            dat2[1] = pickle.loads(bytes(dat2[1]))
                        else:
                            dat2[1] = None
                    if field3 in self.word_meta.ID_ARRAY_FIELDS:
                        if dat2[2]: 
                            dat2[2] = pickle.loads(bytes(dat2[2]))
                        else:
                            dat2[2] = None
                    if field2 is None:
                        yield dat2[0]
                    else:
                        yield tuple(dat2)
            else:
                dat = self.sql_execute(stmt, val, fetch=True)
                if reverse:
                   dat = reversed(dat)
                for a_rec in dat:
                    a_rec = list(a_rec)
                    if a_rec:
                        if type(a_rec[self.word_meta.word]) is BytesType: 
                            a_rec[self.word_meta.word] = a_rec[self.word_meta.word].decode('utf-8')
                        a_rec[self.word_meta.word] = str(a_rec[self.word_meta.word])
                        for f in self.word_meta.ID_ARRAY_FIELDS:
                            if a_rec[f]: 
                                a_rec[f] = pickle.loads(bytes(a_rec[f]))
                            else:
                                a_rec[f] = None                                   
                        yield a_rec


    def set_word(self, id_or_words, a_rec=None, embeds=None, check_kb_item_exists=False, force_kb_insert=False):
        """ 
        Update the kb and array with the words.  id_or_words is an
        id or words list or a single rec. words can be an array of full rec or
        tuples. If a tuple, then the information will be updated
        against what is on disk. if of rec type, the information
        will be replace what is on disk.
        *Changing a word for a specific id will not give warnings.*
        Resetting a word to another id will remap the id in the
        whole kb. 
        If a word is already in the kb, the new information will
        replace the old information.
        """
        id = None
        if type(id_or_words) in (np.int32, np.int64, np.uint32, np.uint64, IntType):
            id = int(id_or_words)
            if not self.is_word(a_rec) and a_rec in ((), None):
                raise RuntimeError("trying to set an item to empty values. use del_word instead")
            else:
                if self.is_word(a_rec):
                    a_rec[self.word_meta.ID] = id
                else:
                    a_rec = list(a_rec)
                    len_a_rec = len(a_rec)
                    if len_a_rec <= self.word_meta.ID:
                        a_rec = a_rec+([0]*(self.word_meta.ID-len_a_rec+1))
                    a_rec[self.word_meta.ID] = id


        ret_single = False
        if type(id_or_words) is TupleType or self.is_word(id_or_words):
            ret_single=True
            words = [id_or_words]
        elif a_rec is not None:
            ret_single=True
            words = [a_rec]
        else:
           words = id_or_words
        if not words:
            return []

        with self.get_lock():
            words = [list(a_rec) for a_rec in words]
            for a_rec in words:
                len_a_rec = len(a_rec)
                if len_a_rec > self.word_meta.ID:
                    a_rec[self.word_meta.ID] = self.word_to_id(a_rec[self.word_meta.word], a_rec[self.word_meta.ID])
                else:
                   a_rec = a_rec+[None]*(self.word_meta.ID-len_a_rec+1)
                   a_rec[self.word_meta.ID] = -1
            max_id = [a_rec[self.word_meta.ID] for a_rec in words if a_rec[self.word_meta.ID] != -1] + [-1]
            max_id = max(max_id)
            if max_id >= self.len_():
               self.resize(max_id+1) 
            need_id = [a_rec for a_rec in words if a_rec[self.word_meta.ID] == -1]
            if need_id:
               id = self.len_()
               self.resize(id+len(need_id))
               max_id = id - 1
               for a_rec in need_id:
                  a_rec[self.word_meta.ID] = id
                  id +=1
            need_id = None
            words2 = {}
            if not force_kb_insert:
                words2_id = [a_rec[self.word_meta.ID] for a_rec in words if not self.is_word(a_rec) and a_rec[self.word_meta.ID] <= max_id]
                if words2_id:
                   words2 = dict([(a_rec[self.word_meta.ID], a_rec) for a_rec in self.get_word_iter(words2_id)])
            for i in range(len(words)):
                b_rec = words[i]
                if self.is_word(b_rec):
                    a_rec = b_rec
                else:
                    a_rec = None
                    if not force_kb_insert:
                        id = b_rec[self.word_meta.ID]
                        if id in  words2:
                            a_rec = words2[id]
                            del words2[id]
                    if a_rec is None:
                        a_rec = self.new_word()
                    for field, dat in enumerate(b_rec):
                        if field in self.word_meta.ID_ARRAY_FIELDS:
                            self.set_word_array_field(a_rec,field,dat)
                        elif field == self.word_meta.word:
                            a_rec[field] = str(dat)
                        else:
                            a_rec[field] = dat
                words[i] = a_rec
            if words2:
                raise RuntimeError("problem with getting words in update rec")
            if ret_single:
                self.set_word_by_id(words[0][self.word_meta.ID], words[0], check_kb_item_exists=check_kb_item_exists, force_kb_insert=force_kb_insert)
            else:
                self.set_word_by_id([a_rec[self.word_meta.ID] for a_rec in words], words, check_kb_item_exists=check_kb_item_exists, force_kb_insert=force_kb_insert)
            all_ids = dict([(int(a_rec[self.word_meta.ID]),1) for a_rec in words])
            remap_hash = {}
            del_ids = []
            for a_rec in words:
                id = a_rec[self.word_meta.ID]
                word = a_rec[self.word_meta.word]
                if self.has_word(word) and self.word_to_id(word) != id:
                    remapped_id = self.word_to_id(word)
                    remap_hash[remapped_id] = id
                    if remapped_id not in all_ids: del_ids.append(remapped_id)
            for  a_rec in words:
                word = a_rec[self.word_meta.word]
                id = int(a_rec[self.word_meta.ID])
                old_word = self.id_to_word(id)
                if old_word is not None and old_word != word:
                    if old_word in self.encoder: del self.encoder[old_word]
                    if old_word in self.base_encoder: self.base_encoder[old_word]
                if (self.has_word(word) and self.word_to_id(word) != id):
                    if self.has_word(word) and word not in self.encoder and word not in self.base_encoder:
                        raise RuntimeError("Trying to change id of a fixed word in tokenizer")
                    if word in self.encoder: self.encoder[word] = id
                    if word in self.base_encoder: self.base_encoder[word] = id
                    if id in self.decoder: self.decoder[id] = word
                    if id in self.base_decoder: self.base_decoder[id] = word
                elif (not self.has_word(word)):
                   self.encoder[word] = id
                   self.decoder[id] = word
            if del_ids:
                self.del_word(del_ids)
            if remap_hash:
                self.remap_words(remap_hash)

        if ret_single:
           if embeds is not None:
              self.embed_weight_data()[words[0][self.word_meta.ID]] = embeds#[0]
        else:
           if embeds is not None:
              self.embed_weight_data()[[a_rec[self.word_meta.ID] for a_rec in words]] = embeds
        self.flush()
        if ret_single:
           return words[0]
        return words

    def set_word_by_id(self, id, a_word_or_words, check_kb_item_exists=False, force_kb_insert=True):
        if type(id) == SliceType:
            start = (id.start or 0)
            stop = (id.stop or self.shape[0])
            id = range(start, stop)
        if type(id) in (ListType, RangeType):
            update_vals = []
            insert_vals = []
            for i, a_rec in zip(id, a_word_or_words):
                a_rec = copy.copy(a_rec)
                for field in self.word_meta.ID_ARRAY_FIELDS:
                    if a_rec[field]: 
                        a_rec[field] = sqlite3.Binary(pickle.dumps(a_rec[field], pickle.HIGHEST_PROTOCOL))
                    else:
                        a_rec[field] = None
                if not force_kb_insert and self.word_exists(i) and (not check_kb_item_exists or self.db_item_exists(i)):
                    update_vals.append(a_rec +[i])
                else:
                    insert_vals.append(a_rec)
            with self.get_lock():
                if insert_vals:
                   self.sql_execute(self.word_meta.INSERT_STMT, insert_vals, many=True, commit=True)
                if update_vals:
                   self.sql_execute(self.word_meta.UPDATE_STMT, update_vals, many=True, commit=True)
        else:
            i = id
            a_rec = copy.copy(a_word_or_words)
            for field in self.word_meta.ID_ARRAY_FIELDS:
                if a_rec[field]: 
                    a_rec[field] = sqlite3.Binary(pickle.dumps(a_rec[field], pickle.HIGHEST_PROTOCOL))
                else:
                    a_rec[field] = None
            with self.get_lock():
                logger.info((self.word_meta.UPDATE_STMT,))
                logger.info((a_rec,))
                if not force_kb_insert and self.word_exists(i) and (not check_kb_item_exists or self.db_item_exists(i)):
                    self.sql_execute(self.word_meta.UPDATE_STMT, a_rec+[i], many=False, commit=True)
                else:
                    self.sql_execute(self.word_meta.INSERT_STMT, a_rec, many=False, commit=True)

    def del_word_by_id(self, id):
        if type(id) == SliceType:
            start = (id.start or 0)
            stop = (id.stop or self.shape[0])
            id = range(start, stop)
        if type(id) == ListType:
           rng_step=999
           len_items = len(id)
           for rng in range(0, len_items, rng_step):
              max_rng = min(rng + rng_step, len_items)
              self.sql_execute("DELETE FROM "+self.word_meta.TABLE_NAME+" WHERE "+self.word_meta.ID_LABEL+" in ("+("?,"*(max_rng-rng-1))+"?)", id[rng:max_rng], commit=True)
        elif type(id) == RangeType:
             self.sql_execute("DELETE FROM "+self.word_meta.TABLE_NAME+" WHERE "+self.word_meta.ID_LABEL+" >= ? and "+self.word_meta.ID_LABEL+" < ?", (id[0], id[-1]), commit=True)
        else:
             self.sql_execute("DELETE FROM "+self.word_meta.TABLE_NAME+" WHERE  "+self.word_meta.ID_LABEL+" == ? ", id, commit=True)

    def mirror_kb_and_tokenizer(self, force_kb_insert=False):
        logger.info("mirroring kb, embed and tokenizer")
        with self.get_lock():
            #del_ids = []
            max_id = self.len_()
            dat = {}
            for id in range(len(self.tokenizer)):
                word = self.id_to_word(id)
                if not word or word.startswith("[unused"):
                    #del_ids.append(id)
                    continue
                cnt = (max_id - id + 1) * 100
                alpha = 1.
                dat[word] = ((word, id, cnt, alpha)) # ASSUMES THIS ORDERING
            for a_rec in self.get_word_iter():
                word = a_rec[self.word_meta.word]
                a_rec2 = dat.get(word)
                if a_rec2 and a_rec2[1] == a_rec[self.word_meta.ID]:
                    del dat[word]
            dat = list(dat.values())
            logger.info(("adding", len(dat), dat))
            if dat and len(dat) > 0: 
               for a_rec in self.set_word(dat, force_kb_insert=force_kb_insert):
                  pass # logger.info((self.word_to_id(a_rec[self.word_meta.word]), self.id_to_word(a_rec[self.word_meta.ID]), a_rec))
            dat = None
            #if del_ids: self.del_word(del_ids)
            #del_ids = None
            for a_rec in self.get_word_iter():
                id = a_rec[self.word_meta.ID]
                word = a_rec[self.word_meta.word]
                if (self.has_word(word) and self.word_to_id(word) != id) or (not self.has_word(word)):                        
                    if self.has_word(word):
                        if word not in self.encoder and word not in self.base_encoder:
                            raise RuntimeError("Trying to change id of a fixed word in tokenizer")
                        if word in self.encoder: self.encoder[word] = id
                        if word in self.base_encoder: self.base_encoder[word] = id
                        if id in self.decoder: self.decoder[id] = word
                        if id in self.base_decoder: self.base_decoder[id] = word
                    else:
                        self.encoder[word] = id
                        self.decoder[id] = word
            new_words = []
            for id in range(self.len_()):
               if self.id_to_word(id) is None and sum(self.embed_weight_data()[id]) != 0.0:
                  new_words.append(("[unused"+str(id)+"]", id))
            if new_words:
               self.set_word(new_words)
        self.save()

    def set_word_array_field(self, a_rec, field, dat):
        if not dat:
            dat = []
        a_len = len(dat)
        size=self.word_meta.ID_ARRAY_FIELDS_SIZE[field]
        if a_len < size:
            a_rec[field] = dat
        else:
            a_rec[field] = dat[:size]

    def flush(self):
        if True: # with self.get_lock():
            self.embed.flush_all_views()
            #self.conn.commit()

    def resize(self, len_kb, defrag=False):
        with self.get_lock():
            old_len_kb = self.len_()
            if old_len_kb > len_kb and not defrag:
                raise RuntimeError("Trying to shrink KB. Defragment instead.")
            self.embed.resize((len_kb, self.embed.shape[1]))


    def len_(self):
        return self.embed.shape[0]

    def defragment(self, rng_step = 1000000):
        """ 
        Defragement the storage of the kb array to remove deleted
        items.  
        
        WARNING: This will invalidate the cluster indexer and
        search results.
        """
        with self.get_lock():
            self.flush()
            do_defragment=[id for id in range(self.len_()) if not self.word_exists(id)]
            len_kb = self.len_()
            if not do_defragment:
                return False
            self.del_word(do_defragment)
            newwords = []
            oldwordsId = []
            cnt = 1
            a_rec = None
            for id2, a_rec in zip(do_defragment, self.get_word_iter(reverse=True)):
                if a_rec[self.word_meta.ID] <= id2:
                    break
                do_defragment = do_defragment[1:]
                a_rec = copy.copy(a_rec)
                oldwordsId.append(a_rec[self.word_meta.ID])
                a_rec[self.word_meta.ID] = id2
                newwords.append(a_rec)
                cnt +=1
                if cnt % rng_step == 0:
                    # copying in chunks is more efficient than multiple disk access
                    # copy the vector before adding because we are really moving the data
                    self.set_word(newwords, embeds=self.embed_weight_data()[oldwordsId])
                    newwords = []
                    oldwordsId = []
                if not do_defragment:
                    break
            resizeLen=a_rec[self.word_meta.ID]+1
            if newwords:
                # copy the vector before adding because we are really moving the data
                self.set_word(newwords, embeds=self.embed_weight_data()[oldwordsId])
                v = None
                newwords = []
                oldwordsId = []
            self.resize(resizeLen, defrag=True)
            self.save()
        return True

    def remap_words(self, remap_hash, do_parent_child=True, ids=None):
        """ convience method for remapping {old_id:new_id} """
        with self.get_lock():
            if ids is None:
                id2 = range(self.len_())
            else:
                id2 = ids
            words = []
            for a_rec in self.get_word_iter(ids=ids):
                changed = False
                for field in self.word_meta.ID_ARRAY_FIELDS:
                    a_change = False
                    if a_rec[field]:
                        new_words = []
                        for key in a_rec[field]:
                            new_id = remap_hash.get(key, key)
                            if new_id != key:
                                a_change = True
                            if new_id != -1:
                                new_words.append(new_id)
                        if a_change:
                            self.set_word_array_field(a_rec,field,new_words)
                            changed = True
                if changed:
                   rec.append(a_rec)
            if words:
               self.set_word(words)


    def del_word(self, ids):
        """ delete words for all the ids """
        if not ids:
            return
        with self.get_lock():
            if type(ids) in (np.int32, np.int64, IntType):
                ids = [int(ids)]
            elif self.is_word(ids):
                ids = [int(ids[self.word_meta.ID])]
            for id in ids:
                word = self.id_to_word(id)
                if not word or word.startswith("[unused"):
                    continue
                if word in self.encoder: 
                    del self.encoder[word]
                if id in self.decoder:
                    del self.decoder[id]
                if word in self.base_encoder: 
                    del self.base_encoder[word]
                if id in self.base_decoder:    
                    del self.base_decoder[id]
            if ids:
                len_kb = self.len_()
                del_items2 = [i for i in ids if i < len_kb]
                if len(del_items2) == 0:
                    return
                else:
                    embed_weight_data = self.embed_weight_data()
                    embed_weight_data[del_items2] = np.zeros(shape=(len(del_items2), embed_weight_data.shape[1]), dtype=embed_weight_data.dtype)
                    self.del_word_by_id(del_items2)
                    self.flush() # if we don't save here, we will need to do a sanity check when we start up the daabase to check deleted items

    def cleanup_kb(self, min_cluster_size=-1, updated_items=[], recompute_means=True, rng_step=10000, word_type=None):
        """ Create the parent vectors as the mean of the children
        vectors. Do some cleanup if a cluster in the ontology is
        below a certain size.  If we are recomputing specific
        updated_items, the mean calulation will just happen for those
        items and their parents
        returns (all deleted ids, all declustered ids)
        """        
        with self.get_lock():
            logger.info("recompute parents " + str(recompute_means))
            self.flush()
            declustered_items = []
            if updated_items:
                recompute_means = False
            all_del_ids = []
            updated_parent_embeds={}
            cnt = 0
            if not updated_items:
                ids = list(range(self.len_()))
            else:
                ids = updated_items
            ids = list(self.get_word_iter(field=self.word_meta.ID, field2=self.word_meta.LEVEL, ids=ids, word_type=word_type))
            if _pyVer==2:
                ids.sort(lambda a, b: cmp(b[1], a[1]))
            else:
                ids.sort(key=lambda a: a[1], reverse=True)
            while True:
                ids2 = []
                for a_rec in self.get_word_iter(ids=[i[0] for i in ids]):
                    cnt += 1
                    id = a_rec[self.word_meta.ID]
                    level = a_rec[self.word_meta.LEVEL]
                    (parents, children) = [i for i in a_rec[self.word_meta.PARENTS] if i >= 0], [i for i in a_rec[self.word_meta.CHILDREN] if i >= 0], 
                    for a in children or []:
                        if not self.word_exists(a):
                            children.remove(a)
                    if level > 0 and min_cluster_size > 0 and len(children) < min_cluster_size:
                        all_del_ids.append(id)
                        declustered_items.extend(children)
                        self.del_word(id)
                        ids2.extend([(i, level-1) for i in children])
                        ids2.extend([(i, level+1) for i in parents])
                    else:
                        if parents and not self.word_exists(parents[0]):
                            a_rec = self.get_word(id)
                            self.set_word_array_field(a_rec,self.word_meta.PARENTS,[])
                            self.set_word(id, a_rec)
                        elif parents:
                            parents = [parents[0]] + self.get_word(parents[0])[self.word_meta.PARENTS]
                            parents = [a for a in parents if self.word_exists(a) and a >= 0]
                            if tuple(parents) != tuple([i for i in a_rec[self.word_meta.PARENTS] if i >= 0]):
                                a_rec = self.get_word(id)
                                self.set_word_array_field(a_rec,self.word_meta.PARENTS,parents)
                                self.set_word(id, a_rec)
                        if tuple(children) != tuple([i for i in a_rec[self.word_meta.CHILDREN] if i >= 0]) and recompute_means:
                            a_rec = self.get_word(id)
                            # should this be all children or just the immediate children. this could get very expensive at the top.
                            if children != []:
                                updated_parent_embeds[a_rec[self.word_meta.ID]] = np.mean(self.embed_weight_data()[children], axis=0)
                            if cnt % rng_step == 0:
                                for id3 in list(updated_parent_embeds.keys()):
                                    if not self.word_exists(id3):
                                        del updated_parent_embeds[id3]
                                updated_parent_items = updated_parent_embeds.items()
                                self.embed_weight_data()[[i[0] for i in updated_parent_items]] = np.array([i[1] for i in updated_parent_items])
                                self.flush()
                                updated_parent_embeds={}
                                updated_parent_items = None
                            self.set_word_array_field(a_rec,self.word_meta.CHILDREN,children)
                            if children: 
                                a_rec[self.word_meta.CNT] = self.get_word(children[0], field=self.word_meta.CNT)
                            self.set_word(id, a_rec)
                if not ids2:
                    break
                ids = ids2
                ids = remove_duplicates([id for id in ids if self.word_exists(id[0])])
                if _pyVer==2:
                    ids.sort(lambda a, b: cmp(b[1], a[1]))
                else:
                    ids.sort(key=lambda a: a[1], reverse=True)
            for id3 in list(updated_parent_embeds.keys()):
                if not self.word_exists(id3):
                    del updated_parent_embeds[id3]
            updated_parent_items = updated_parent_embeds.items()
            declustered_items = remove_duplicates([id for id in declustered_items if self.word_exists(id)])
            self.embed_weight_data()[[i[0] for i in updated_parent_items]] = np.array([i[1] for i in updated_parent_items])
            updated_parent_embeds={}
            updated_parent_items = None
            for id in all_del_ids:
                a_rec = self.get_word(id)
                if a_rec:
                    logger.info(("deleting ", a_rec))
            self.del_word(all_del_ids)
            self.save()
            return (all_del_ids, [i for i in declustered_items if self.word_exists(i)])

    def load_word2vec_glove_format(self, file_name, binary=True, limit=None, all_words=None, min_cnt=4, find_all_compounds=False, find_all_compounds_strict=False, 
                             collapse_cutoff=0.5, collapse_all_cases=True, prefer_short_words=False, rng_step=250000, 
                             min_alpha=0.25, max_word_size=100, max_cnt=0, unique_upper_case_words=("I", "AM", "May", "March")):


        def cleanup_word(word):
            word = word.replace("#", "1")
            word = word.replace("-", "_")
            word = word.replace("|", "_")
            word = word.replace("=", "_")
            #word = word.replace("+", "_")
            word = word.replace("__", "_")
            word = word.replace("__", "_")
            word = word.replace("__", "_")
            word = word.replace("__", "_")
            word = word.replace("__", "_")
            word = word.replace("__", "_")
            word = word.replace("....", "...")
            word = word.replace("....", "...")
            word = word.replace("....", "...")
            word = word.replace("....", "...")
            word = word.replace("....", "...")
            word = word.replace("....", "...")
            word = word.strip("_")
            if len(word) > 4 and word[0] in "0123456789" and word[-1] in "0123456789":
                word = word.translate(trannum)
            return word

        def add_to_new_embed_word(word, weights, new_hash, line_no, limit, all_words, from_word_list=False):
            if line_no > limit:
                return
            word = word.strip()
            word = cleanup_word(word)
            word2 = None
            if len(word) > max_word_size:
                word = word[:max_word_size]
            if not word or ("@" in word  and "." in word) or ".co" in word or ".org" in word or ".gov" in word or ".edu" in word or "www" in word or "http:" in word or ".net" in word or ".uk" in word or ".ca" in word:
                return
            if collapse_all_cases:
                word = word.lower()
            if not from_word_list:
                all_words[word.lower()] = 1
            elif not ((collapse_all_cases and word.lower() in all_words) or (find_all_compounds and ((len(word.split("_")[0]) > 3 and word.split("_")[0].lower() not in basic_word_to_tag and  word.split("_")[0].lower() in all_words) or (len(word.split("_")[-1]) > 3 and  word.split("_")[-1].lower() not in basic_word_to_tag and  word.split("_")[-1].lower() in all_words)))):
                return
            if find_all_compounds_strict and  (word.split("_")[0].lower() not in all_words or word.split("_")[0].lower() not in all_words):
                return
            if not collapse_all_cases:
                capitalized ="_".join([(len(w) <= 2 and w) or (len(w) > 2 and w[0].upper()+w[1:].lower()) or '' for w in word.split("_")])
                all_caps = word.upper()
                all_lower = word.lower()
                if word not in unique_upper_case_words and word.lower() in self.en_common_verbs:
                    # let's collapse all common verbs together into a lower
                    # case version. 
                    word2 = all_lower
                elif word not in unique_upper_case_words and ((all_lower in new_hash and np_cosine(weights, new_hash[all_lower][0][0]) > collapse_cutoff) or (self.has_word(all_lower) and sum(self.v(all_lower)) != 0.0 and np_cosine(weights, self.v(all_lower)) > collapse_cutoff)):
                    word2 = all_lower
                elif word not in unique_upper_case_words and ((capitalized in new_hash and np_cosine(weights, new_hash[capitalized][0][0]) > collapse_cutoff) or (self.has_word(capitalized) and sum(self.v(capitalized)) != 0.0 and np_cosine(weights, self.v(capitalized)) > collapse_cutoff)):
                    word2 = capitalized
                elif word not in unique_upper_case_words and word != word.lower() and ((all_caps in new_hash and np_cosine(weights, new_hash[all_caps][0][0]) > collapse_cutoff) or (self.has_word(all_caps) and sum(self.v(all_caps)) != 0.0 and np_cosine(weights, self.v(all_caps)) > collapse_cutoff)):
                    word2 = all_caps
                elif word not in unique_upper_case_words and word.upper() == word and (len(word) > 2 and (capitalized in new_hash or self.has_word(capitalized))):
                    # let's collapse upper case words into previous case
                    # words because google has a whole set of parrallel
                    # word sets that are all upper case.
                    word2 = capitalized
                elif word not in unique_upper_case_words and word.upper() == word and (word.lower() in new_hash or self.has_word(word.lower())):
                    word2 = all_lower
            if word == "</s>":
                word = self.pad_word
            if word in new_hash:
                new_hash[word].append([weights, max(min_cnt, vocab_size - line_no + 1)])
            else:
                if self.has_word(word):
                    id = self.word_to_id(word)
                    weights2 = self.embed_weight_data()[id]
                    if sum(weights2) != 0.0:
                        new_hash[word] = [[weights2, counts[id]]]
                        new_hash[word].append([weights, max(min_cnt, vocab_size - line_no + 1)])
                    else:
                        new_hash[word] = [[weights, max(min_cnt, vocab_size - line_no + 1)]]
                else:
                    new_hash[word] = [[weights, max(min_cnt, vocab_size - line_no + 1)]]
            if word2 is not None:
               if word2 in new_hash:
                   new_hash[word2].append([weights, max(min_cnt, vocab_size - line_no + 1)])
               else:
                   if self.has_word(word2):
                       id = self.word_to_id(word2)
                       weights2 = self.embed_weight_data()[id]
                       if sum(weights2) != 0.0:
                           new_hash[word2] = [[weights2, counts[id]]]
                           new_hash[word2].append([weights, max(min_cnt, vocab_size - line_no + 1)])
                       else:
                           new_hash[word2] = [[weights, max(min_cnt, vocab_size - line_no + 1)]]
                   else:
                       new_hash[word2] = [[weights, max(min_cnt, vocab_size - line_no + 1)]]

        def save_part(self, new_hash, max_cnt2, counts, alphas):
            logger.info("save part")
            new_hash2 = {}
            for word, arr in new_hash.items():
                y = sum([a[1] for a in arr])
                vec = sum([a[0]*float(a[1]/y) for a in arr])
                y = max([a[1] for a in arr])
                if prefer_short_words: 
                    if vocab_size - y <= max(0.01*vocab_size,10000):
                        y = int(math.sqrt(max(1, 5-len(word)))*y)
                if word in self.stop_words:
                    new_hash2[word] = [vec, max(min_cnt, int(2*y))]
                else:
                    new_hash2[word] = [vec, max(min_cnt, int(y/((word.count("_")+1))))]
                #logger.info(("shortened cnt", word, new_hash[word][1]))
            new_hash.clear()
            new_hash = None
            max_cnt = max(max_cnt2[0], int(max([vecCnt[1] for vecCnt in new_hash2.values()])+1))
            if self.pad_word in new_hash2:
                new_hash2[self.pad_word][1] = max_cnt
            word_embeds = list(new_hash2.items())
            do_print = True
            if _pyVer==2:
                word_embeds.sort(lambda b, a: cmp(a[1][1], b[1][1]))
            else:
                word_embeds.sort(key=lambda a: a[1][1], reverse=True)
            all_words = []
            for word, vecCnt in word_embeds:
                id = self.word_to_id(word)
                if id is not None:
                    cnt = max(counts[id], vecCnt[1])
                    alpha = min(alphas[id], min(1.0, max(min_alpha, math.sqrt(max(1, max_cnt - cnt + 1))/math.sqrt(max_cnt))))
                else:
                    id = -1
                    cnt = vecCnt[1]
                    alpha = min(1.0, max(min_alpha, math.sqrt(max(1, max_cnt - cnt + 1))/math.sqrt(max_cnt)))
                all_words.append((word, id, cnt, alpha)) # ASSUMES THIS IS THE RIGHT ORDERING

            if all_words:
               for a_rec in self.set_word(all_words, embeds=[tvc[1][0] for tvc in word_embeds]):
                  counts[a_rec[self.word_meta.ID]] = max(counts[a_rec[self.word_meta.ID]], a_rec[self.word_meta.CNT])
                  alphas[a_rec[self.word_meta.ID]] = min(alphas[a_rec[self.word_meta.ID]], a_rec[self.word_meta.ALPHA])
            all_words = None
            word_embeds = None
            self.flush()
            max_cnt2[0] = max(max_cnt2[0], max_cnt)

        def to_unicode(text): 
            if isinstance(text, unicode): return text
            return unicode(text, encoding='utf8', errors='replace')

        # main

        max_cnt2 = [max_cnt]
        if find_all_compounds_strict:
            find_all_compounds=True
        from_word_list=False
        if not all_words:
            all_words = {}
        else:
            from_word_list=True
        new_hash = {}
        fin =  open(file_name, 'rb')
        header = to_unicode(fin.readline())
        orig_vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
        vocab_size = orig_vocab_size
        if not limit:
            limit = vocab_size
        counts = [0]*(vocab_size+self.len_()+1000)
        alphas = [1.]*(vocab_size+self.len_()+1000)
        for id, cnt, alpha in self.get_word_iter(field=self.word_meta.ID, field2=self.word_meta.CNT, field3=self.word_meta.ALPHA):
            counts[id] = cnt
            alphas[id] = alpha
        workers = []
        if binary:
            binary_len = np.dtype(np.float32).itemsize * vector_size
            for line_no in range(orig_vocab_size):
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n': 
                        word.append(ch)
                word = to_unicode(b''.join(word))
                weights = np.fromstring(fin.read(binary_len), dtype=np.float32) # vector_size
                if vector_size < self.embed.embedding_dim:
                   a = int(self.embed.embedding_dim/vector_size)
                   b = self.embed.embedding_dim%vector_size
                   weights = np.concatenate([weights]*a + [weights[:b]])
                elif vector_size > self.embed.embedding_dim:
                   weights = weights[:self.embed.embedding_dim]
                if sum(weights) != 0:
                    add_to_new_embed_word(word, weights, new_hash, line_no, limit, all_words, from_word_list=from_word_list)
                    if line_no >= limit-1:
                        if  not collapse_all_cases and not find_all_compounds:
                            break
                        else:
                            limit = orig_vocab_size
                            from_word_list = True
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
                word, weights = parts[0], np.array([np.float32(x) for x in parts[1:]])
                if sum(weights) != 0:
                    add_to_new_embed_word(word, weights, new_hash, line_no, limit, all_words, from_word_list=from_word_list)
                    if line_no >= limit-1:
                        if  not collapse_all_cases and not find_all_compounds:
                            break
                        else:
                            limit = orig_vocab_size
                            from_word_list = True
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
        self.create_compound_word_count()
        self.save()
        return self

    def create_indexer(self, items=None, rng_step = 10000,  level=0, do_hiearchical=True, num_threads=None, num_parents=None, cluster_size=None, saved_obj=None, lock=True, kb=None):
        """ 
        creates a hiearchical parent index of the underlying embeddings
        finds random N nodes as the head
        nodes to the clusters of leaves and randomly re-clusters the
        clusters to create relatively balanced leaves
        input:
        - items = None for level 0 - indexing all values. 
        - items = [] for level 0 - refining the index. 
        - items = a list to index for any level - indexing just the list (or reindexing at level 0 just the list)
        
        default behviour is to lock while indexing (calling at level 0, mainprocess). 
        state variables:
        - indexer_nodes: level X nodes
        - indexer_nodes_matrix: level X 2d numpy array of nodes X embedding.shape[1]. This is a copy of the nodes items from the embedding. 
        - indexer_nodes_matrix_norm: level X numpy array that is the norm of the nodes matrix above. 
        - indexer_leaves_for_nodes: level X leaf python arrays, each array corresponding to a node in the indexer_nodes.  
          **** indexer_nodes[X][Y] is in indexer_leaves_for_nodes[X][Y]
        - indexer_leaves:
        """
        view_data = self.get_view(self.filename)
        if saved_obj is None:
            saved_obj = self
        if lock:
            tlock = view_data.tlock or DummyLock()
        else:
            tlock = DummyLock()
        view_data = None
        with tlock:
            if level == 0 and lock: self.flush_all_views()
            num_embeddings = self.shape[0]
            if items is not None and kb:
                items = [a for a in items if kb.token_exists(a)]
            if items is not None:
                items = remove_duplicates(items)
            #logger.info("got here create indexer 1")
            indexer_leaves = items
            if num_embeddings == 0:
                return
            if not indexer_leaves:
                saved_obj.indexer_leaves_for_nodes.clear()
                saved_obj.indexer_leaves.clear()
                saved_obj.indexer_nodes.clear()
                saved_obj.indexer_nodes_matrix.clear()
                saved_obj.indexer_nodes_matrix_norm.clear()

            if num_threads is None:
                num_threads = max(4, multiprocessing.cpu_count())
                #num_threads = 1
            if num_threads > 1 and level == 0:
                if  items is not None:
                    size = max(items)+1
                else:
                    size = num_embeddings 
                for num_threads2 in range(num_threads, 1, -1):
                    if size/num_threads2 < rng_step:
                        num_threads = max(1, num_threads2-1)
            #logger.warning(('num_threads level', num_threads, level))
            orig_indexer_leaves=None
            if num_threads > 1 and level == 0 and not items:
                #logger.info("multiprocessing indexer")
                #if self.multiprocessing_lock is not None:
                #    raise RuntimeError("can only start multiprocessing indexing from single processing mode")
                workers=[]
                num_tokens = num_embeddings
                if not indexer_leaves:
                    if kb:
                        indexer_leaves = [c for c in range(num_embeddings) if kb.token_exists(c)]
                    else:
                        indexer_leaves = list(range(num_embeddings))
                    num_tokens = len(indexer_leaves)

                rng_step2 = int(num_tokens/num_threads)

                random.shuffle(indexer_leaves)
                id_num = 0
                lenWorkers=0
                matrix_size = int(np.sqrt(num_embeddings))
                cluster_size = int(num_embeddings/matrix_size) 
                num_parents0 = int(num_embeddings/cluster_size)
                all_saved_obj = []
                for rng in range(0, num_tokens, rng_step2):
                    if id_num < num_threads-1:
                        max_rng = min(rng + rng_step2, num_tokens)
                    else:
                        max_rng = num_tokens
                    num_parents = int(num_parents0/(num_threads-1))
                    tmp = Object()
                    tmp.indexer_leaves_for_nodes=[]
                    tmp.indexer_leaves=[]
                    tmp.indexer_nodes=[]
                    tmp.indexer_nodes_matrix=[]
                    tmp.indexer_nodes_matrix_norm=[]
                    all_saved_obj.append(tmp)
                    aThread = threading.Thread(target=OntoEmbedding.create_indexer_multithreading, args=(indexer_leaves[rng:max_rng], self, tmp, num_parents, cluster_size, kb))
                    workers.append(aThread)
                    aThread.start()
                    if id_num == num_threads-1:
                        break
                    id_num += 1
                for aThread in workers:
                    if aThread:
                        aThread.join()
                lenWorkers = len(workers)
                workers = None

                all_indexer_nodes = []
                all_indexer_leaves_for_nodes = []
                logger.info(("finished multipthreading", all_saved_obj))
                for tmp in all_saved_obj:
                    all_indexer_nodes.extend(tmp.indexer_nodes[level])
                    all_indexer_leaves_for_nodes.extend(tmp.indexer_leaves_for_nodes[level])
                all_saved_obj = None
                #logger.info("finished copying " + str(len(all_nodes)))
                saved_obj.indexer_leaves_for_nodes.clear()
                saved_obj.indexer_leaves.clear()
                saved_obj.indexer_nodes.clear()
                saved_obj.indexer_nodes_matrix.clear()
                saved_obj.indexer_nodes_matrix_norm.clear()
                saved_obj.indexer_nodes.append(all_indexer_nodes)
                saved_obj.indexer_leaves_for_nodes.append(all_indexer_leaves_for_nodes)
                saved_obj.indexer_leaves.append(None)
                saved_obj.indexer_nodes_matrix.append(self.get_view(self.filename)[all_indexer_nodes])
                saved_obj.indexer_nodes_matrix_norm.append(norm_matrix(saved_obj.indexer_nodes_matrix[-1]))
                indexer_nodes = all_indexer_nodes
            else:
                #logger.info(("normal indexer", level))
                saved_obj.indexer_leaves_for_nodes.extend([None]*(level+1-len(saved_obj.indexer_leaves_for_nodes)))
                saved_obj.indexer_leaves.extend([None]*(level+1-len(saved_obj.indexer_leaves)))
                saved_obj.indexer_nodes.extend([None]*(level+1-len(saved_obj.indexer_nodes)))                
                saved_obj.indexer_nodes_matrix.extend([None]*(level+1-len(saved_obj.indexer_nodes_matrix)))                
                saved_obj.indexer_nodes_matrix_norm.extend([None]*(level+1-len(saved_obj.indexer_nodes_matrix_norm)))                

                for i in range(len(saved_obj.indexer_nodes)):
                    if saved_obj.indexer_leaves_for_nodes[i] is None:
                        saved_obj.indexer_leaves_for_nodes[i] = []
                    if saved_obj.indexer_nodes[i] is None:
                        saved_obj.indexer_nodes[i] = []
                logger.info("step 0")            
                if level == 0:
                    matrix_size = int(np.sqrt(num_embeddings))
                    if cluster_size is None:
                        cluster_size = int(num_embeddings/matrix_size) # 500
                    if num_parents is None:
                        #logger.info(("setting num_parents", num_parents))
                        num_parents = int(num_embeddings/cluster_size)
                    else:
                        #logger.info(("received num_parents", num_parents))
                        pass
                    if not indexer_leaves:
                        indexer_leaves = range(num_embeddings)
                        num_tokens = len(indexer_leaves)
                        orig_indexer_leaves = None
                    elif saved_obj.indexer_nodes:
                        if isinstance(indexer_leaves, ListType):
                            indexer_leaves = remove_duplicates(indexer_leaves)
                        child_hash = dict([(a, 1) for a in indexer_leaves])
                        indexer_leaves_for_nodes = saved_obj.indexer_leaves_for_nodes[level]
                        indexer_leaves = []
                        indexer_nodes = saved_obj.indexer_nodes[level]

                        # remove the current items being indexed from the indexer_leaves set
                        for i in range(len(indexer_leaves_for_nodes)):
                            indexer_leaves_for_nodes[i] = [a for a in indexer_leaves_for_nodes[i] if child_hash.get(a) is None]
                            #num_tokens += len(indexer_leaves_for_nodes[i])

                        # if the current items is being indexed if they are a parent node, remove all children
                        for j, i in enumerate(indexer_nodes):
                            if child_hash.get(i) is not None:
                                # we assume that each child is in only one
                                # indexer_leaves_for_nodes, so we don't have to cascade
                                # through the sets
                                indexer_leaves.extend(indexer_leaves_for_nodes[j])
                                indexer_leaves_for_nodes[j] = []
                        indexer_leaves.extend(child_hash.keys())
                        child_hash = None
                        num_tokens = len(indexer_leaves)      
                        orig_indexer_leaves = None
                    # do a sanity check and remove all indexer_leaves that have been deleted
                    if kb:
                        indexer_leaves = [c for c in indexer_leaves if kb.token_exists(c)]
                    num_tokens = len(indexer_leaves)

                elif level > 0 and indexer_leaves:
                    saved_obj.indexer_leaves[level] = copy.copy(indexer_leaves)
                    num_tokens = len(indexer_leaves)
                    orig_indexer_leaves = saved_obj.indexer_leaves[level]
                    indexer_leaves = range(num_tokens)
                    matrix_size = int(np.sqrt(num_tokens))
                    cluster_size = int(num_tokens/matrix_size) # 500
                    num_parents = int(num_tokens/cluster_size)
                else:
                    raise RuntimeError("Can't cluster at level > 0 with no indexer_leaves")            

                logger.info("step 1")            
                j = 0
                if level == 0:
                    indexer_nodes2 = saved_obj.indexer_nodes[level]
                    indexer_leaves_for_nodes2 = saved_obj.indexer_leaves_for_nodes[level]
                    indexer_nodes = []
                    indexer_leaves_for_nodes = []
                    clear_hiearchy = False
                    # let's check to see if any of the indexer_nodes have been deleted
                    for parent, cset in zip(indexer_nodes2, indexer_leaves_for_nodes2):
                        if (kb and not kb.token_exists(parent)):
                            indexer_leaves.extend([c for c in cset if kb.token_exists(c)])
                            clear_hiearchy=True
                        else:
                            indexer_nodes.append(parent)
                            indexer_leaves_for_nodes.append([c for c in cset if not kb or kb.token_exists(c)])
                    if clear_hiearchy:
                        saved_obj.indexer_leaves_for_nodes=[None]
                        saved_obj.indexer_leaves = [None]
                        saved_obj.indexer_nodes = [None]

                else:
                    indexer_nodes = saved_obj.indexer_nodes[level]
                    indexer_leaves_for_nodes = saved_obj.indexer_leaves_for_nodes[level]

                logger.info("step 2")            
                num_tokens = len(indexer_leaves)
                indexer_nodes_hash = dict([(s, 1) for s in indexer_nodes])
                children_no_parents = [c for c in indexer_leaves if indexer_nodes_hash.get(c) is None]
                new_parents_len = num_parents - len(indexer_nodes)
                if new_parents_len > 0 and new_parents_len  <= len(children_no_parents):
                    indexer_nodes.extend(_random.sample(children_no_parents, num_parents - len(indexer_nodes)))
                children_no_parents = None
                len_parents = len(indexer_nodes)
                indexer_leaves_for_nodes.extend([None]*(len_parents - len(indexer_leaves_for_nodes)))
                parents_hash = dict([(s, 1) for s in indexer_nodes])
                for s, r in enumerate(indexer_leaves_for_nodes):
                    if indexer_leaves_for_nodes[s] is None:
                        indexer_leaves_for_nodes[s] = []
                logger.info ("start indexing " + str(len(indexer_nodes)) + " num_tokens " + str( num_tokens))        
                if orig_indexer_leaves is None:
                    b_matrix = self.get_view(self.filename)[indexer_nodes]
                    b_norm = norm_matrix(b_matrix)
                else:
                    b_matrix = self.get_view(self.filename)[[orig_indexer_leaves[i] for i in indexer_nodes]]
                    b_norm =  norm_matrix(b_matrix)
                max_times = 20
                t0 = 0
                for times in range(max_times):
                    searched = 0
                    added = 0
                    logger.info ("indexer_nodes " + str(len(indexer_nodes)) + " num_tokens " + str(num_tokens))
                    if indexer_leaves != []: # hack - setting indexer_leaves == None will even out the sets
                        for rng in range(0, num_tokens, rng_step):
                            t0 += 1
                            max_rng = min(rng + rng_step, num_tokens)
                            logger.info ("rng " + str(rng) + " max_rng " +  str(max_rng))
                            if orig_indexer_leaves is None:
                                a_children = indexer_leaves[rng:max_rng]
                            else:
                                a_children = [orig_indexer_leaves[i] for i in indexer_leaves[rng:max_rng]]

                            a_matrix = self.get_view(self.filename)[a_children]
                            if True:
                                a_norm = norm_matrix(a_matrix)
                                a_norm = a_norm.reshape(a_norm.shape[0], 1)                        
                                ns = cosine_search(a_matrix, b_matrix, a_norm, b_norm, k=1)
                                a_matrix = None
                                a_norm = None
                                for r, vals in enumerate(ns):
                                    for id in range(0, vals.shape[0], 2):
                                        ki = int(vals[id])
                                        searched += 1
                                        if indexer_leaves[rng+r] not in indexer_nodes_hash:
                                            added+= 1
                                            indexer_leaves_for_nodes[ki].append(indexer_leaves[rng+r])
                                            #logger.info(len(indexer_leaves_for_nodes[ki]))
                            ns = None
                        logger.info ("searched " + str(searched) +  " added " +  str(added))
                    # now see if the selection is a good distribution
                    if times < max_times-1:
                        indexer_leaves2 = []
                        indexer_nodes2 = []
                        logger.info("adding removing " + str(times) + " max size " + str(int(1.5*cluster_size)) + " min size " + str(int(cluster_size*(1.0-times/max_times)*0.2)))
                        for s, new_rec in enumerate(indexer_leaves_for_nodes):
                            len_new_rec= len(new_rec)  
                            new_cluster_size = len_new_rec/cluster_size
                            #logger.info(("considering ", len_new_rec, new_cluster_size))
                            if new_cluster_size < (1.0-times/max_times)*0.2: # remove very small clusters
                                #logger.info ("problematic cluster of size " + str(len_new_rec) + " removing")
                                indexer_leaves2.extend(new_rec)
                                indexer_leaves2.append(indexer_nodes[s])
                                indexer_nodes[s] = None
                                indexer_leaves_for_nodes[s]=None
                            elif new_cluster_size >= 1.5:
                                new_cluster_size = int(new_cluster_size) + 1
                                #logger.info ("problematic cluster of size " + str(len_new_rec) + " adding " + str(new_cluster_size))
                                #if self.kb:
                                #    logger.info("problem with cluster taking majority "+ self.get_token(s][TOKEN_REC])
                                samp = _random.sample(new_rec, new_cluster_size)
                                new_s = samp[0]
                                samp = samp[1:]
                                indexer_nodes2.extend(samp)
                                indexer_leaves2.extend(new_rec)
                                indexer_leaves2.append(indexer_nodes[s])
                                indexer_nodes[s]=new_s
                                indexer_leaves_for_nodes[s]=[]
                        for p in indexer_nodes2:
                            indexer_leaves_for_nodes.append([])
                            indexer_nodes.append(p)
                        indexer_leaves_for_nodes = [cs for cs in indexer_leaves_for_nodes if cs is not None]
                        indexer_nodes = [p for p in indexer_nodes if p is not None]
                        indexer_leaves=indexer_leaves2
                        num_tokens = len(indexer_leaves)
                        if kb and not orig_indexer_leaves:                        #IS THIS RIGHT?
                            indexer_leaves = [c for c in indexer_leaves if kb.token_exists(c)]
                            num_tokens = len(indexer_leaves)
                        if num_tokens <= 0:
                            break
                        len_indexer_nodes = len(indexer_nodes)
                        indexer_nodes_hash = dict([(s, 1) for s in indexer_nodes])
                        if orig_indexer_leaves is None:
                            b_matrix = self.get_view(self.filename)[indexer_nodes]
                            b_norm = norm_matrix(b_matrix)
                        else:
                            b_matrix = self.get_view(self.filename)[[orig_indexer_leaves[i] for i in indexer_nodes]]
                            b_norm = norm_matrix(b_matrix)
                        logger.info ("reclustering with more tokens " + str(num_tokens) + " " + str(len(indexer_nodes)) + " " + str(len(indexer_leaves_for_nodes)))
                logger.info(("step 4", level))
                saved_obj.indexer_nodes[level] = indexer_nodes
                for p in range(len(indexer_nodes)):
                    if indexer_nodes[p] not in indexer_leaves_for_nodes[p]:
                        indexer_leaves_for_nodes[p].append(indexer_nodes[p])
                    indexer_leaves_for_nodes[p] = remove_duplicates(indexer_leaves_for_nodes[p])
                saved_obj.indexer_leaves_for_nodes[level] = indexer_leaves_for_nodes
                if orig_indexer_leaves is not None:
                    mapped_indexer_nodes = [orig_indexer_leaves[s] for s in saved_obj.indexer_nodes[level]]
                    saved_obj.indexer_nodes_matrix[level] = self.get_view(self.filename)[mapped_indexer_nodes]
                    saved_obj.indexer_nodes_matrix_norm[level] = norm_matrix(saved_obj.indexer_nodes_matrix[level])
                else:
                    saved_obj.indexer_nodes_matrix[level] = self.get_view(self.filename)[indexer_nodes]
                    saved_obj.indexer_nodes_matrix_norm[level] = norm_matrix(saved_obj.indexer_nodes_matrix[level])

      torch.save(self, self.filename.replace(".mmap", ".pt"))

    def search_for_parents(self):
       pass

    def merge_with(self, model_name_or_model):
       src_to_self = {}
       if type(model_name_or_model) is str:
          src = AutoModel.from_pretrained(model_name_or_model)

       for id in range(src.len_()):
          token = src.id_to_token(id)
          if not token or token.startswith("[unused"):
             unused[id] = 1
       unused2 = list(unused.keys())
       self_len = self.len_()
       all_recs = []
       id3 = self.len_()
       logger.info("starting to create src_to_self")
       sufs = {}
       for (pre, suf) in src.bpe_ranks.keys():
          pre = src_txt(pre)
          suf = src_txt(suf)
          sufs[suf] = 1

       for id in range(src.len_()):
          token = src_txt(src.id_to_token(id))
          not_suf = False
          token2 = None
          if token == "<|endoftext|>":
             token = self.tokenizer.sep_token
          elif token[0] == " ":
             token = token.strip()
          else:
             token = token.strip()
             if not self.has_token(token):
                token = "##"+token
             elif token in sufs:
                token2 = "##"+token                   
          if self.has_token(token):
             src_to_self[id] = self.token_to_id(token)
          else:
             if unused2:
                id2 = unused2[0]
                unused2 = unused2[1:]
                all_recs.append((token, id2, max(10, self_len-id), 1.0))
                src_to_self[id] = id2
             else:
                all_recs.append((token, id3, max(10, self_len-id), 1.0))
                src_to_self[id] = id3
                id3 += 1       
          if not token2:
             continue
          if self.has_token(token2):
             src_to_self[id] = self.token_to_id(token2)
          else:
             if unused2:
                id2 = unused2[0]
                unused2 = unused2[1:]
                all_recs.append((token2, id2, max(10, self_len-id), 1.0))
                src_to_self[id] = id2
             else:
                all_recs.append((token2, id3, max(10, self_len-id), 1.0))
                src_to_self[id] = id3
                id3 += 1       

       if all_recs:
          logger.info(("starting to save new self", len(all_recs)))
          if all_recs: self.set_token(all_recs)

       with torch.no_grad():
          logger.info("starting to create new embed")
          for id in range(min(self.len_(), 2*src.len_()+src.len_())):
             token = self.id_to_token(id)
             weight = self.embed_weight_data()[id]
             if sum(weight) == 0.0:
                not_in_self[id] =  1
          logger.info (('not in self', len(not_in_self)))
          for iter_ in range(2): # 3
             logger.info(("iter", iter_))
             # now let's create embeddings for those items not in self
             gpt_set_only = {}
             search_results = src.search_for_parents()
             for id, result in enumerate(search_results):
                main_id = src_to_self[id]
                result2 = [(src_to_self[id2], score) for id2, score in result[1:] if src_to_self[id2] not in not_in_self and score >=0.01]
   #             if len(result2) > 25:
   #                result2 = result2[:25]
                if len(result2) <=1:
                   logger.info(("problem with search item", self.id_to_token(main_id), len(result2)))
                   continue
                elif len(result2) > 0:
                   total_score = sum([score for id2, score in result2])
                   if sum(self.embed_weight_data()[main_id]) == 0.0:
                      gpt_set_only[main_id] = 1
                      self.embed_weight_data()[main_id] = sum(self.embed_weight_data()[[id2 for id2, score in result2]]*(np.atleast_2d([score/total_score for id2, score in result2])).T)
                   else:
                      self.embed_weight_data()[main_id] = self.embed_weight_data()[main_id]*0.9 + 0.1*sum(self.embed_weight_data()[[id2 for id2, score in result2]]*(np.atleast_2d([score/total_score for id2, score in result2])).T)
                   if main_id in not_in_self: del not_in_self[main_id]

             neg_search_file=src.embed.filename.replace(".mmap", ".search_neg.mmap")
             neg_search_results = src.search(search_file=neg_search_file, vs_fn=get_neg_items, items=list(range(src.len_())), search_indexer_nodes_only=True)
             for id, result in enumerate(neg_search_results):
                main_id = src_to_self[id]
                result2 = [(src_to_self[id2], score) for id2, score in result[1:] if src_to_self[id2] not in not_in_self]
   #             if len(result2) > 25:
   #                result2 = result2[:25]
                if len(result2) <=0:
                   logger.info(("problem with neg search item", self.id_to_token(main_id), len(result2)))
                   continue
                elif len(result2) > 0:
                   total_score = sum([score for id2, score in result2])
                   if total_score != 0:
                      if sum(self.embed_weight_data()[main_id]) == 0.0:
                         logger.info(("no items in neg", self.id_to_token(main_id), len(result2)))
                         continue
                      else:
                         self.embed_weight_data()[main_id] = self.embed_weight_data()[main_id]*1.1 + -0.1*sum(self.embed_weight_data()[[id2 for id2, score in result2]]*(np.atleast_2d([score/total_score for id2, score in result2])).T)
                   if main_id in not_in_self: del not_in_self[main_id]
             search_results = src.search()
             for id, result in enumerate(search_results):
                if id in unused:
                   continue
                result2 = [(id2, score) for id2, score in result[1:] if id2 not in not_in_self and id2 not in unused and score >=0.01]
                main_id = id
                has_embed = sum(self.embed_weight_data()[main_id]) != 0.0
   #             if len(result2) > 25:
   #                result2 = result2[:25]
                if len(result2) <=0:
                   if has_embed:
                      continue
                   logger.info(("problem with search item", self.id_to_token(main_id), len(result2)))
                   continue
                elif len(result2) > 0:
                   total_score = sum([score for id2, score in result2])
                   if has_embed:
                      if main_id in gpt_set_only:
                         self.embed_weight_data()[main_id] = 0.5*self.embed_weight_data()[main_id] + 0.5*sum(self.embed_weight_data()[[id2 for id2, score in result2]]*(np.atleast_2d([score/total_score for id2, score in result2])).T)
                      else:
                         self.embed_weight_data()[main_id] = 0.9*self.embed_weight_data()[main_id] + 0.1*sum(self.embed_weight_data()[[id2 for id2, score in result2]]*(np.atleast_2d([score/total_score for id2, score in result2])).T)                         
                   else:
                      self.embed_weight_data()[main_id] = sum(self.embed_weight_data()[[id2 for id2, score in result2]]*(np.atleast_2d([score/total_score for id2, score in result2])).T)
                   if main_id in not_in_self: del not_in_self[main_id]
             
