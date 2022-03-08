import torch
from transformers import T5EncoderModel, MT5EncoderModel, AutoTokenizer, AutoModel
import itertools

def get_word_embeddings(model, tokenizer, sentence, words, background_text=""):
  encoded_input = [tokenizer.decode(a) for a in tokenizer.encode(sentence)]
  #print (encoded_input)
  len_encoded_input = len(encoded_input)
  lcToKey = dict([(a.lower().replace(" ", ""), a) for a in words])
  model_input = tokenizer(sentence+" "+background_text if background_text else sentence, padding=True, truncation=True, return_tensors='pt')
  # Compute token embeddings  
  with torch.no_grad():
    model_output = model(**model_input, return_dict=True)
  aHash = {}
  for ent in words:
    ent = tuple([tokenizer.decode(a)  for a in tokenizer.encode(ent, add_special_tokens=False)])
    len_ent = len(ent)
    search_results = ([slice(i, i+len_ent) for i, a in zip(range(len_encoded_input), encoded_input) if a == ent[0] and ent == tuple(encoded_input[i:min(len_encoded_input,i+len_ent)])])
    #print (ent, search_results)
    aHash[ent] = aHash.get(ent, []) + search_results
  for key in aHash.keys():
    val =  sum([torch.sum(model_output.last_hidden_state [0][i], axis=0)/(max(1.0, len(model_output.last_hidden_state [0][i]))) for i in aHash[key]])/max(1.0, len(aHash[key]))
    aHash[key] =torch.functional.F.normalize(val, dim=0)
    
  return dict([(lcToKey[''.join([a.lstrip('##') for a in key]).replace(" ", "").lower()], val) for key, val in aHash.items()])

###  "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
##"vocab-transformers/dense_encoder-msmarco-distilbert-word2vec256k-MLM_785k_emb_updated", "nicoladecao/msmarco-word2vec256000-distilbert-base-uncased", "xlm-roberta-base", "xlm-roberta-large",
lang = "en"
is_cjk = lang in {"ja", "ko", "th"} or lang.startswith("zh")
for model_type in ["sentence-transformers/distiluse-base-multilingual-cased-v2",  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "sentence-transformers/LaBSE" ]:
  #\
  #                 "google/t5-small-ssm", "t5-small", "google/mt5-small", "google/t5-large-ssm", "t5-large",  "google/mt5-large",  "sentence-transformers/sentence-t5-base", \
  #                 ]:
  tokenizer = AutoTokenizer.from_pretrained(model_type) 
  model = MT5EncoderModel.from_pretrained(model_type) if "mt5-"  in model_type else T5EncoderModel.from_pretrained(model_type) if "t5-" in model_type else AutoModel.from_pretrained(model_type) 
  for background_text in ["", " BACKGROUND: Michelle Robinson is married to Barack Obama, and they have two daugthers. The family are all African Americans. "]:

    aHash = get_word_embeddings(model, tokenizer, """Barack Hussein Obama II's tenure as the 44th president of the United States began with his first inauguration on January 20, 2009 after defeating John McCain, and ended on January 20, 2017. 
    Obama, a Democrat from Illinois, took office following a decisive victory over Republican nominee John McCain in the 2008 presidential election.""", \
      ["Barack Hussein Obama II", "John McCain", "44th president", "Obama"], background_text=background_text)
    bHash = get_word_embeddings(model, tokenizer, "الرئيس الرابع والأربعين يقف مع زوجته ميشيل أوباما وابنتيه ساشا وماليا، 2009", \
                                ["الرئيس الرابع والأربعين", "ميشيل أوباما", "أوباما", "ساشا"], background_text=background_text)
    cHash = get_word_embeddings(model, tokenizer, "The 44th President poses with wife Michelle Obama and daughters Sasha and Malia, 2009", \
                                ["The 44th President", "Michelle Obama", "Obama", "Sasha", ], background_text=background_text)#"Malia"
    
    #merge aHash and bHash. Should probably be a weighted sum.
    for lang, mHash, idx in [('ar', bHash, 1), ('en', cHash, 2)]:
      for key in mHash.keys():
        if key not in aHash:
          aHash[key] = mHash[key]
        elif len(key) > 6 or (lang in {'zh', 'ja', 'ko', 'th'} and len(key) > 4):
          aHash[key] = (aHash[key] + mHash[key])/2.0
        else:
          key2 = key+"#"+str(idx)
          aHash[key2] = mHash[key] if key2 not in aHash else (aHash[key2] + mHash[key])/2.0 

    print (f'## {model_type} with background: {background_text}')
    print ('## entities', list(aHash.keys()))

    #Compute dot score between query and all ents embeddings
    ents = ["44th president", "Obama", "John McCain","الرئيس الرابع والأربعين", "ميشيل أوباما", "أوباما", "ساشا", "The 44th President", "Michelle Obama", "Obama#2", "Sasha"]
    scores = torch.mm(aHash['Barack Hussein Obama II'].unsqueeze(-1).T, torch.vstack([aHash[s] for s in ents]).transpose(0,1))[0].cpu().tolist()
    ent_score_pairs = list(zip(['Barack Hussein Obama II']*len(ents), ents, scores))
    for a in ent_score_pairs: print (a)

    ents = ["Barack Hussein Obama II", "44th president", "Obama", "John McCain", "ميشيل أوباما", "أوباما", "ساشا", "The 44th President", "Michelle Obama", "Obama#2", "Sasha"]
    scores = torch.mm(aHash["الرئيس الرابع والأربعين"].unsqueeze(-1).T, torch.vstack([aHash[s] for s in ents]).transpose(0,1))[0].cpu().tolist()
    ent_score_pairs = list(zip(["لرئيس الرابع والأربعين"]*len(ents), ents, scores))
    for a in ent_score_pairs: print (a)

    ents =  ["Barack Hussein Obama II", "44th president", "Obama", "John McCain","الرئيس الرابع والأربعين", "ميشيل أوباما", "أوباما", "ساشا",  "Michelle Obama", "Obama#2", "Sasha"]
    scores = torch.mm(aHash["The 44th President"].unsqueeze(-1).T, torch.vstack([aHash[s] for s in ents]).transpose(0,1))[0].cpu().tolist()
    ent_score_pairs = list(zip(["The 44th President"]*len(ents), ents, scores))
    for a in ent_score_pairs: print (a)
