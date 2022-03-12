# coding=utf-8
# Copyright, 2021-2022 Ontocord, LLC, and the authors of this repository. All rights reserved.
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

import os
import re
import glob
import json
import pandas as pd
from tqdm import tqdm
from google.colab import drive
from collections import defaultdict
from datasets import load_dataset
from ontology.ontology_manager import OntologyManager

ontology_langs = ['ace', 'af', 'als', 'am', 'an', 'ang', 'ar', 'arc', 'arz', 'as', 'ast', 'ay', 'az', 'ba', 'bar', 'bat-smg', 'be', 'be-x-old', 'bg', 'bh', 'bn', 'bo', 'br', 'bs', 'ca', 'cbk-zam', 'cdo', 'ce', 'ceb', 'ckb', 'co', 'crh', 'cs', 'csb', 'cv', 'cy', 'da', 'de', 'diq', 'dv', 'el', 'eml', 'en', 'eo', 'es', 'et', 'eu', 'ext', 'fa', 'fi', 'fiu-vro', 'fo', 'fr', 'frr', 'fur', 'fy', 'ga', 'gan', 'gd', 'gl', 'gn', 'gu', 'hak', 'he', 'hi', 'hr', 'hsb', 'hu', 'hy', 'ia', 'id', 'ig', 'ilo', 'io', 'is', 'it', 'ja', 'jbo', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ksh', 'ku', 'ky', 'la', 'lb', 'li', 'lij', 'lmo', 'ln', 'lt', 'lv', 'map-bms', 'mg', 'mhr', 'mi', 'min', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'mwl', 'my', 'mzn', 'nap', 'nds', 'ne', 'nl', 'nn', 'no', 'nov', 'oc', 'or', 'os', 'pa', 'pdc', 'pl', 'pms', 'pnb', 'ps', 'pt', 'qu', 'rm', 'ro', 'ru', 'rw', 'sa', 'sah', 'scn', 'sco', 'sd', 'sh', 'si', 'simple', 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'szl', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vec', 'vep', 'vi', 'vls', 'vo', 'wa', 'war', 'wuu', 'xmf', 'yi', 'yo', 'zea', 'zh', 'zh-classical', 'zh-min-nan', 'zh-yue']

# update to add ORGs, LOC as necessary
relevant_tags = ["PER"]

# number of test cases to run per lang. set None to iterate through all available
num_instances = 10000
results = dict()

ontology_langs = [
    'ace', 'af', 'als', 'am', 'an', 'ang', 'ar', 'arc', 'arz', 'as', 'ast',
    'ay', 'az', 'ba', 'bar', 'bat-smg', 'be', 'be-x-old', 'bg', 'bh', 'bn', 
    'bo', 'br', 'bs', 'ca', 'cbk-zam', 'cdo', 'ce', 'ceb', 'ckb', 'co', 'crh', 
    'cs', 'csb', 'cv', 'cy', 'da', 'de', 'diq', 'dv', 'el', 'eml', 'en', 'eo', 
    'es', 'et', 'eu', 'ext', 'fa', 'fi', 'fiu-vro', 'fo', 'fr', 'frr', 'fur', 
    'fy', 'ga', 'gan', 'gd', 'gl', 'gn', 'gu', 'hak', 'he', 'hi', 'hr', 'hsb', 
    'hu', 'hy', 'ia', 'id', 'ig', 'ilo', 'io', 'is', 'it', 'ja', 'jbo', 'jv', 
    'ka', 'kk', 'km', 'kn', 'ko', 'ksh', 'ku', 'ky', 'la', 'lb', 'li', 'lij',
    'lmo', 'ln', 'lt', 'lv', 'map-bms', 'mg', 'mhr', 'mi', 'min', 'mk', 'ml',
    'mn', 'mr', 'ms', 'mt', 'mwl', 'my', 'mzn', 'nap', 'nds', 'ne', 'nl', 'nn',
    'no', 'nov', 'oc', 'or', 'os', 'pa', 'pdc', 'pl', 'pms', 'pnb', 'ps', 'pt', 
    'qu', 'rm', 'ro', 'ru', 'rw', 'sa', 'sah', 'scn', 'sco', 'sd', 'sh', 'si', 
    'simple', 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'szl', 'ta', 
    'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vec', 
    'vep', 'vi', 'vls', 'vo', 'wa', 'war', 'wuu', 'xmf', 'yi', 'yo', 'zea', 
    'zh', 'zh-classical', 'zh-min-nan', 'zh-yue'
    ]

def parse_entities(x, lang, do_wikiann, do_remove_whitespace):
    gold_entities = defaultdict(list)
    if do_wikiann:
        # extract entities substrings from wikiann format
        for span in x["spans"]:
            m = re.match("(\w+):(.+)", span)
            ent_type, ent = m.group(1), m.group(2)
            if ent_type not in gold_relevant_tags:
                continue
            if do_remove_whitespace:
                ent = ent.replace(" ", "")
            ent = ent.strip().lower()
            gold_entities[ent_type].append(ent)
    else:
        # pii hackathon format
        for (ent, _, _), ent_types_dict in x[f"{lang}_ner"].items():
            for ent_type in [_ for _ in ent_types_dict]:
                if ent_type in gold_relevant_tags:
                    gold_entities[ent_type].append(ent.strip().lower())
    return gold_entities
    
def predict_with_ontology(text, onto, do_remove_whitespace, **kwargs):
    pred_entities = defaultdict(list)
    onto_output = onto.tokenize(text)
    for ent, ent_type in onto_output['chunk2ner'].items():
        if ent_type not in onto_relevant_tags:
            continue
        ent_span = ent[0].strip().lower()
        ent_span = ent_span.replace("_", " ")
        if do_remove_whitespace:
            ent_span = ent_span.replace(" ", "")
        pred_entities[ent_type].append(ent_span)

    return pred_entities

def compare_entities(gold_entities, pred_entities):
    TP_i, FN_i, FP_i = [], [], []
    gold_entities_list = list(set([_ for v in gold_entities.values() for _ in v]))
    pred_entities_list = list(set([_ for v in pred_entities.values() for _ in v]))
    for entity_type, entities in gold_entities.items():
        for entity in set(entities):
            # we consider entity as match as long as the string matches;
            # we do not enforce entity type requirement;
            # we also do not check for partial matches
            if entity in pred_entities_list:
                TP_i.append({entity_type:entity})
            else:
                FN_i.append({entity_type:entity})
    for entity_type, entities in pred_entities.items():
        for entity in set(entities):
            if entity not in gold_entities_list:
                FP_i.append({entity_type:entity})
    return TP_i, FN_i, FP_i, (gold_entities_list, pred_entities_list)
    
def precision_recall_F1_helper(TP_i, FP_i, FN_i):
    TP = len(TP_i)
    FP = len(FP_i)
    FN = len(FN_i)

    precision = 0. if TP == 0 else TP / (TP + FP) 
    recall = 0. if TP == 0 else TP / (TP + FN)
    if (precision + recall) > 0:
        F1 = 2*precision*recall / (precision + recall)
    else:
        F1 = 0.
    
    return precision, recall, F1

def is_whitespace_lang(lang):
    return (lang.startswith("zh")) or (lang in ["ja", "th", "ko"])
 


split = "test"
for lang in tqdm(ontology_langs) :  # ["zh", "ar", "as", "bn", "ca", "en", "es", "eu", "fr", "gu", "hi", "id", "ig", "mr", "pa", "pt",  "sw", "ur", "vi", "yo", ]):
# "ny", "sn", "st", "xh", "zu" "ko"]): #tqdm(["en"]): #
    onto = OntologyManager(lang)
    do_remove_whitespace = (lang.startswith("zh")) or (lang in ('ja', 'th', "ko"))

    try:
        dataset = load_dataset("wikiann", lang)
    except:
        print(f"{lang} not found in wikiann. skipping...")
        continue
        
    # can iterate through splits for coverage - here we use train

    dataset = dataset[split]  
    results[lang] = dict(TP=[], FN=[], FP=[], ents=[])
    
    data = dataset if num_instances is None else dataset.select(range(num_instances))
    
    for x in data:
        gold_entities = defaultdict(list)
        for span in x["spans"]:
            m = re.match("(\w+):(.+)", span)
            ent_type, ent = m.group(1), m.group(2)
            if ent_type not in relevant_tags:
                continue
            if ent == '-{zh': continue
            if do_remove_whitespace:
                ent = ent.replace(" ", "")
            ent = ent.strip().lower()
            gold_entities[ent_type].append(ent)

        # de-tokenize to feed into ontology pipeline
        onto_entities = defaultdict(list)
        text = " ".join(x["tokens"])
        # get ontology module output
        onto_output = onto.tokenize(text)
        for ent, ent_type in onto_output['chunk2ner'].items():
            ent_span = ent[0].strip().lower()
            ent_span = ent_span.replace("_", " ")
            if do_remove_whitespace:
                ent_span = ent_span.replace(" ", "")
            onto_entities[ent_type].append(ent_span)
        
        # compare expected matches
        TP_i, FN_i, FP_i = [], [], []
        gold_entities_list = list(set([_ for v in gold_entities.values() for _ in v]))
        onto_entities_list = list(set([_ for v in onto_entities.values() for _ in v]))

        for entity_type, entities in gold_entities.items():
            for entity in set(entities):
                # we DONT enforce entity type requirements;
                # we also DONT check for partial matches at this stage
                if entity in onto_entities_list:
                    TP_i.append({entity_type:entity})
                else:
                    FN_i.append({entity_type:entity})
        for entity_type, entities in onto_entities.items():
            for entity in set(entities):
                if entity not in gold_entities_list:
                    FP_i.append({entity_type:entity})
        
        results[lang]["ents"].append(
            {
                "expected": gold_entities_list,
                "predicted": onto_entities_list
            }
        )

        results[lang]["TP"].append(TP_i)
        results[lang]["FN"].append(FN_i)
        results[lang]["FP"].append(FP_i)

        
# replace pred_func with arbitrary method that 
# (1) takes a string as input + kwargs; and
# (2) outputs a dict with entity_type:[entity_strings] as key:values

#def example(text, **kwargs):
#    return {"PER": ["john", "smith"]}

results_df = run_predictions(
    langs=langs, 
    num_instances=num_instances, 
    results_path=results_path,
    pred_func=predict_with_ontology
    )


# overall scores
results_df.precision.mean(), results_df.recall.mean(), results_df.F1.mean()

# table of results - the interactive "Filter" function could be helpful
column_order = ["precision", "recall", "F1", "TP", "FP", "FN"]
results_df.pivot_table(
    index=["lang", "domain"],
    aggfunc={
        "precision": np.mean,
        "recall": np.mean,
        "F1": np.mean,
        "TP": lambda x: x.str.len().sum(),
        "FP": lambda x: x.str.len().sum(),
        "FN": lambda x: x.str.len().sum(),
    }
).reindex(column_order, axis=1)


# look at gold entities vs predictions
results_df[["lang", "gold_entities", "pred_entities"]]


# plot results
plot_table = results_df.pivot_table(
    index=["lang"],
    aggfunc={
        "TP": lambda x: x.str.len().sum(),
        "FP": lambda x: x.str.len().sum(),
        "FN": lambda x: x.str.len().sum(),
    }
)
figsize = (36,3) if do_wikiann else (6,3)
plot_table.plot(kind='bar', figsize=figsize)


# plot specific languages 
selected_langs = ["en", "pt"]
plot_table[plot_table.index.isin(selected_langs)].plot(kind='bar', figsize=(6,3))
