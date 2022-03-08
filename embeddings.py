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
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model, XLNetModel, XLNetTokenizer, RobertaTokenizer, RobertaModel, AlbertModel, AlbertTokenizer
import sys, gc
import sqlite3
import traceback
NAME_2_MODEL = {'bert': BertModel, 'gpt2': GPT2Model, 'xlnet': XLNetModel, 'roberta': RobertaModel, 'albert': AlbertModel}
NAME_2_TOKENIZER = {'bert': BertTokenizer, 'gpt2': GPT2Tokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer, 'albert': AlbertTokenizer}

try:
   import cPickle as pickle
except:
   import pickle


_use_torch = True
   
_pyVer = 2
try:
    from types import IntType, FloatType, IntType, LongType, ListType, StringType, TupleType, DictType, BooleanType, SliceType, RangeType
    BytesType = bytes
    from string import atoi
    range = xrange
except ImportError:
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



if _pyVer==2:
    trannum = string.maketrans("0123456789", "1111111111")
else:
    trannum = str.maketrans("0123456789", "1111111111")


import logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s : %(processName)s : %(threadName)s : %(levelname)s : %(message)s',
    level=logging.DEBUG)


##

################## UTILITIY FUNCTIONS AND CLASSES  ################

_SP = " "

def bSearch(arr, id0, id1, key, doBookend, isList=False):
    for i in range(10000):
        if id1 < id0:
            return (None, None)
        # bookend test if either end of the list includes this
        # value. Sometimes useful for domains that are very large
        # where values are likely outside the range.
        if doBookend: 
            id = id0
            keyVal = arr[id]
            isList = isList or isinstance(keyVal, ListType) or isinstance(keyVal, TupleType)
            if isList:
                val = keyVal
            elif keyVal == None:
                val = (None, None)
            else:
                val = keyVal.split(_SP, 1)
            key2 = val[0]
            match = cmp(key2, key)
            if match == 0:
                return (val[1], id)
            elif id0 == id1:
                return (None, None)
            elif match > 0:
                if id <= id0: return (None, None)
            elif match < 0:
                if id >= id1: return (None, None)                

            id = id1
            keyVal = arr[id]
            isList =  isList or isinstance(keyVal, ListType) or isinstance(keyVal, TupleType)
            if isList:
                val = keyVal
            elif keyVal == None:
                val = (None, None)
            else:
                val = keyVal.split(_SP, 1)
            key2 = val[0]
            match = cmp(key2, key)
            #print match, key2, key
            if match == 0:
                return (val[1], id)
            elif id0 == id1:
                return (None, None)
            elif match > 0:
                if id <= id0: return (None, None)
            elif match < 0:
                if id >= id1: return (None, None)                

        id = id0 + int((id1 - id0)/2)
        #print "*", id0, id1, id
        if id > id1:
            return (None, None)
        if id < id0:
            return (None, None)
        keyVal = arr[id]
        isList = isList or isinstance(keyVal, ListType) or isinstance(keyVal, TupleType) 
        if isList:
            val = keyVal
        elif keyVal == None:
            val = (None, None)
        else:
            val = keyVal.split(_SP, 1)
        #print "*", key, keyVal
        key2 = val[0]
        match = cmp(key2, key)
        #print id0, id1, id, key, key2, match
        #print match, key2, key, val

        if match == 0:
            return (val[1], id)
        elif id0 == id1:
            return (None, None)
        elif match > 0:
            if id <= id0: return (None, None)
            id1 = id-1
            doBookend=False
            continue
        elif match < 0:
            if id >= id1: return (None, None)                
            id0 = id+1
            doBookend=False
            continue
        else:
            return (None, None)
    # we should never get to this point. this should never happen
    return (None, None)

def starts_with(str1, prefix):
    return str1[:len(prefix)] == prefix

def ends_with(str1, suffix):
    return str1[-len(suffix):] == suffix

def combo(u):
    lenu1 = len(u)
    ret= []
    for i in range(lenu1):
       for j in range(0, i):
            ret.append((u[j], u[i]))
    return ret

def product(u, v):
    return flatten1(list(map(lambda a, v=v:list(map(lambda b, a=a:(a,b), v)), u)))

def all_sequences(lst):
    if len(lst) == 0: return []
    if len(lst) == 1: return [[s,] for s in lst[0]]
    if len(lst) == 2: return [list(s) for s in product(lst[0], lst[1])]
    return [[s[0]]+s[1] for s in product(lst[0], all_sequences(lst[1:]))]

def remove_duplicates(sequence):
    a_hash = {}
    accumulator = []
    for a in sequence:
        b = a
        if isinstance(a, ListType):
            b = tuple(a)
        if a_hash.get(b, None) == None:
            accumulator.append(a)
            a_hash[b] = 1
    return accumulator


def flatten1(sequence):
    accumulator = []
    for item in sequence:
        if isinstance(item,TupleType):
            accumulator.extend(list(item))            
        elif isinstance(item, ListType):
            accumulator.extend(item)
        else:
            accumulator.append(item)
    return accumulator

OUTLIER_MAX = 0
OUTLIER_MEAN = 1
OUTLIER_MEDIAN = 2

def remove_outliers(a_hash, numStdDev=1.0, bottomOnly=True, type=OUTLIER_MEAN, minCutOff=5):
    aList = None
    if isinstance(a_hash, ListType) :
        if hasattr(a_hash[0], 'score'):
            aList = a_hash
            a_hash = {}
            for result in aList:
                a_hash[result] = result.score
        else:
            return a_hash
        
    items = list(a_hash.keys())
    if len(items) == 0: return a_hash    
    if len(items) <= minCutOff : return a_hash
    mean = 0.0
    max = 0.0
    if _pyVer==2:
        items.sort(lambda a, b: cmp(a_hash[a], a_hash[b]))
    else:
        items.sort(key=lambda a: a_hash[a])        
    for item in a_hash:
        mean += a_hash[item]
        if max < a_hash[item]: max = a_hash[item]
    mean = mean / len(list(a_hash.keys()))
    midPoint = mean
    if type==OUTLIER_MAX:
        midPoint = max
    elif type==OUTLIER_MEDIAN:
        mid = int(len(items)/2)
        median = a_hash[items[mid]]
        midPoint = median
    sum = 0.0
    for item in a_hash:
        sum += (a_hash[item] - midPoint) * (a_hash[item] - midPoint)
    stdDev = math.sqrt(float(sum)/float(len(list(a_hash.keys()))))
    bottomCutOff = midPoint - (stdDev * numStdDev)
    topCutOff = midPoint + (stdDev * numStdDev)
    iCnt = len(items)
    #items.reverse()
    #print items
    for item in items:
        if iCnt <= minCutOff: break
        if  a_hash[item] < bottomCutOff:
            a_hash[item] = None
            #print "removing", item
            del a_hash[item]
            iCnt -= 1
        if not bottomOnly and a_hash[item] > topCutOff:
            a_hash[item] = None
            #print "removing", item            
            del a_hash[item]
            iCnt -= 1            
    if not aList:
        return a_hash
    else:
        return list(a_hash.values())

def max_key_from_hash(a_hash, maxFn=max):
    if not a_hash:
        return None
    maxVal = maxFn(list(a_hash.values()))
    for key in a_hash:
        if a_hash[key] == maxVal:
            return key
    return None

def extend_ret(list1, list2, list3=None, list4=None, list5=None, list6=None, list7=None):
    list1 = copy.copy(list1)
    list1.extend(list2)
    if list3:
        list1.extend(list3)
    if list4:
        list1.extend(list4)
    if list5:
        list1.extend(list5)
    if list6:
        list1.extend(list6)
    if list7:
        list1.extend(list7)
    return list1


def merge_hashes(origHashSet, addFn=sum):
    hashArrays = []
    if isinstance(origHashSet, DictType):
        hashArrays = list(origHashSet.values())
    else:
        hashArrays = list(origHashSet)
    if not hashArrays: return {}
    a_hash = hashArrays[0]
    allHashes = [s for s in hashArrays if s]
    if not allHashes:
        return {}
    typeHash = copy.copy(allHashes[0])
    hashArrays = hashArrays[1:]
    for a_hash2 in hashArrays:
        for key in a_hash2:
            val2 =  a_hash2[key]
            if a_hash.get(key, None) == None:
                a_hash[key] = copy.copy(val2)
                continue
            elif a_hash2.get(key, None) == None:
                continue
            else:
                val = a_hash.get(key)
                if isinstance(typeHash.get(key), BooleanType):
                    a_hash[key] = val and val2
                elif (isinstance(typeHash.get(key), IntType) or isinstance(typeHash.get(key), FloatType) or isinstance(typeHash.get(key), LongType)):
                    a_hash[key] = addFn([val, val2])
                elif isinstance(typeHash.get(key), StringType):
                    a_hash[key] += " " + val2                
                elif isinstance(typeHash.get(key), TupleType):
                    a_hash[key] = tuple(remove_duplicates(list(val) + list(val2)))
                elif (isinstance(typeHash.get(key), ListType) or isinstance(typeHash.get(key), TupleType)):
                    a_hash[key] = list(val) + list(val2)
                elif isinstance(typeHash.get(key), DictType):
                    a_hash[key] = merge_hashes([a_hash[key], a_hash2[key]])
                else:
                    #ignore errors
                    pass
    return a_hash

def create_inv_hash(a_hash1):
    """
    convert a hash of from {a:b, c:b} -> {b:[a,c]}
    """
    a_hash2 = {}
    for key, val in a_hash1.items():
        a_hash2[val] = a_hash2.get(val, []) + [key]
    return a_hash2

def intersection(u, v):
    """
    find intersaction between two arrays
    """
    v2 = {}
    for e in v:
        v2[e] = 1
    w = []
    for e in u:
        if v2.get(e,None) != None:
            w.append(e)
    return w


__indx__ = 0
_trie_ignorecase = __indx__
__indx__ += 1
_trie_strTable = __indx__
__indx__ += 1
_trie_dimSizes = __indx__
__indx__ += 1
_trie_firstChar = __indx__
__indx__ += 1
_trie_lastChar = __indx__
__indx__ += 1
_trie_size = __indx__

def trie_init(prefixList=[], suffixList=[], wordList=[], default=None, ignorecase=True):
    """
    A trie used for searching strings patterns.
    """
    _self = [None]*_trie_size
    _self[_trie_ignorecase] = ignorecase
    _self[_trie_strTable] = None
    if _self[_trie_ignorecase]:
        _self[_trie_firstChar] = min(ord('a'), ord('z'))
        _self[_trie_lastChar] =  max(ord('a'), ord('z'))
    else:
        _self[_trie_firstChar] = min(min(ord('a'), ord('z')), min(ord('A'), ord('Z')))
        _self[_trie_lastChar] =  max(max(ord('a'), ord('z')), max(ord('A'), ord('Z')))
    _self[_trie_dimSizes] = [(_self[_trie_lastChar] - _self[_trie_firstChar]) + 1, (_self[_trie_lastChar] - _self[_trie_firstChar]) + 1, (_self[_trie_lastChar] - _self[_trie_firstChar]) + 1]
    ignorecase = _self[_trie_ignorecase]
    if suffixList and isinstance(suffixList[0], TupleType):
        suff2 = [((''.join(reversed(a[0])), a[1]), 0, len(a[0])) for a in suffixList]
    else:
        suff2 = [((''.join(reversed(a)), default), 0, len(a)) for a in suffixList]
    if _pyVer == 2:
        suff2.sort(lambda a, b: cmp(a[0][0], b[0][0]))
    else:
        suff2.sort(key=lambda a: a[0][0])
#    suff2 = map(lambda a: ((''.join(a[0][0]), a[0][1]), a[1], a[2]), suff2)

    if prefixList and isinstance(prefixList[0], TupleType):
        pref2 = [(a, 1, len(a[0])) for a in prefixList]
    else:
        pref2 = [((a, default), 1, len(a)) for a in prefixList]
    if _pyVer == 2:
        pref2.sort(lambda a, b: cmp(a[0][0], b[0][0]))
    else:
        pref2.sort(key=lambda a: a[0][0])
    
    if wordList and isinstance(wordList[0], TupleType):
        wordList = [(a, 2, len(a[0])) for a in wordList]            
    else:
        wordList = [((a, default), 2, len(a)) for a in wordList]            
    if _pyVer == 2:
        wordList.sort(lambda a, b: cmp(a[0][0], b[0][0]))
    else:
        wordList.sort(key=lambda a: a[0][0])
    if not _self[_trie_strTable]:
        _self[_trie_strTable] = [None]*3
    strTable = _self[_trie_strTable]
    firstChar = _self[_trie_firstChar]
    dimSizes = _self[_trie_dimSizes]
    allWords = suff2 + pref2 +  wordList

    for aComplex in  allWords:
        wordAndVal=aComplex[0]
        stringType=aComplex[1]
        lenWord=aComplex[2]
        if strTable[stringType] == None:
            strTable[stringType] = [None] * (dimSizes[0]+1) # add one extra slot for other chars
        val = wordAndVal[1]
        if ignorecase:
            word = wordAndVal[0].lower()
        else:
            word = wordAndVal[0]
        minLen = lenWord
        j = 0
        k = 0
        l = 0
        minLen = lenWord
        j = ord(word[0])- firstChar
        if j < 0 or j > dimSizes[0]-1:
            j = 0
        else:
            j = j + 1
        if lenWord == 1:
            k = 0 
        else:
            k = ord(word[1])- firstChar
            if k < 0 or k > dimSizes[1]-1:
                k = 0
            else:
                k = k + 1
        if lenWord <= 2:
            l = 0 
        else:
            l = ord(word[2])- firstChar
            if l < 0 or l > dimSizes[2]-1:
                l = 0
            else:
                l = l + 1
        if strTable[stringType][j] == None:
            strTable[stringType][j] = [None]*(dimSizes[1]+1)
        if strTable[stringType][j][k] == None:
            strTable[stringType][j][k] = [None]*(dimSizes[2]+1)
        if strTable[stringType][j][k][l] == None:
            strTable[stringType][j][k][l] = [None] * 10
        maxLen = 10
        # fix dim < 3
        if minLen > maxLen:
            minLen = maxLen
        for i in range(minLen, maxLen+1):
            if strTable[stringType][j][k][l][i-1] == None:
                strTable[stringType][j][k][l][i-1] = [[lenWord]]
            else:
                minMax = strTable[stringType][j][k][l][i-1][0]
                needSort = False
                if lenWord not in minMax:
                    minMax.append(lenWord)
                    minMax.sort()
                    minMax.reverse()
                    strTable[stringType][j][k][l][i-1][0] = minMax
            strTable[stringType][j][k][l][i-1].append((word, val, lenWord)) # goal is to minimize this list
    return _self

def trie_get_pref_suff_exact_list_from_table(_self):
    strTable=_self[_trie_strTable]
    suffixTable = strTable[0]
    prefixTable = strTable[1]    
    exactTable = strTable[2]    
    suffixList = []
    prefixList = []
    exactList = []
    for j in suffixTable or []:
        for k in j or []:
            for l in k or []:
                if l:
                    l = l[1:]
                    suffixList.extend([(a and (a[1][0][0], a[1][1])) or a for a in l])


    for j in prefixTable or []:
        for k in j or []:
            for l in k or []:
                if l:
                    l = l[1:]
                    prefixList.extend([(a and (a[1][0][0], a[1][1])) or a for a in l])


    for j in exactTable or []:
        for k in j or []:
            for l in k or []:
                if l:
                    l = l[1:]
                    exactList.extend([(a and (a[1][0][0], a[1][1])) or a for a in l])


    return [remove_duplicates([a for a in prefixList if a]), remove_duplicates([a for a in suffixList if a]), remove_duplicates([a for a in exactList if a])]

#find 0 for suffix, 1 for prefix, 2 for exact
def trie_find(_self, word, default=None, lenWord=-1, prefixSuffix=0, needReverse=True, condTest=None, condState=None):
    strTable = _self[_trie_strTable]
    if prefixSuffix == 0:
        if needReverse:
            word = ''.join(reversed(word))
        
    if not strTable[prefixSuffix]:
        return default
    firstChar = _self[_trie_firstChar]
    dimSizes = _self[_trie_dimSizes]
    if lenWord < 0:
        lenWord = len(word)
    if lenWord == 0:
        return default
    i = lenWord
    j = 0
    k = 0
    l = 0
    j = ord(word[0]) - firstChar		
    if j  < 0 or j > dimSizes[0]-1:
        j = 0
    else:
        j = j + 1
    if lenWord == 1:
        k = 0 
    else:
        k = ord(word[1]) - firstChar		
        if k  < 0 or k > dimSizes[1]-1:
            k = 0
        else:
            k = k + 1
    if lenWord <= 2:
        l = 0 
    else:
        l = ord(word[2]) - firstChar		
        if l  < 0 or l > dimSizes[2]-1:
            l = 0
        else:
            l = l + 1
    if i > 10: 
        i = 10
    if not strTable[prefixSuffix][j]:
        return default
    if not strTable[prefixSuffix][j][k]:
        k = 0
        l = 0
        if not strTable[prefixSuffix][j][k]:
            return default
    if not strTable[prefixSuffix][j][k][l]:
        l = 0
        if not strTable[prefixSuffix][j][k][l]:
            k = 0
            if not strTable[prefixSuffix][j][k]:
                return default
    ret = strTable[prefixSuffix][j][k][l]
    if ret:
        ret = ret[i-1]
        aRet = None
        if ret:
            for lenWord2 in ret[0]:
                if lenWord2 > lenWord:
                    continue
                if prefixSuffix != 2:
                    word2 = word[:lenWord2]
                else:
                    # Check if exact matches here is correct. the first ret has all of the lengths
                    if lenWord2 != lenWord:
                        continue
                    word2 = word
                lenRet = len(ret)
                ret2 = bSearch(ret, 1, lenRet-1, word2, False, True)
                if ret2[0] != None:
                    ret_id = ret2[1]
                    if not condTest:
                        val = ret[ret_id][1] # ret2[0]
                        lenPat = ret[ret_id][2]
                        return (val, lenPat)
                    aRet = condTest(ret[ret_id], condState)
                    # search backwards and forwards in a linear
                    # fashion for a matching rule. can be interepreted
                    # as: match(X) AND conditional test -> val.
                    ret_id0 = ret_id-1
                    while ret_id0 > 0:
                        if word2 ==  ret[ret_id0][0]:
                            aRet = condTest(ret[ret_id0], condState)
                            ret_id0 = ret_id0-1
                        else:
                            break
                    ret_id0 = ret_id+1
                    while ret_id0 < lenRet:
                        if word2 ==  ret[ret_id0][0]:
                            aRet = condTest(ret[ret_id0], condState)
                            ret_id0 = ret_id0+1
                        else:
                            break
                    if aRet:
                        val = aRet[1] # ret2[0]
                        lenPat = aRet[2]
                        return (val, lenPat) # (val, lenPat)
    return default


# apply prefix compression and suffix compression.  for the last step,
# do an exact match. if the compression has reached the end then just
# return the match.
def trie_apply_prefix_suffix_compression(fsmdict, word, lenW, chPrefix=chr(30), chSuffix=chr(21), lenCh=1, condTest=None, condState=None, suffixOnly=False, prefixOnly=False, exactOnly=False):
    if not fsmdict:
        return (word, lenW)
    revWord = ''.join(reversed(word))
    prevLenPrefix = -1
    prevLenSuffix = -1
    doPref = False
    doSuf = False
    retPref = None
    retSuff = None
    cont = True
    # preserve some ordering to the steps, assuming that the longest
    # match is preferred when there is a tie in the ordering.  save on
    # computing prefix/suffix finding where appropriate.
    while cont:
        cont=False
        retExact = None
        if not prefixOnly and not suffixOnly:
            retExact = trie_find(fsmdict, word, lenWord=lenW, prefixSuffix=2, needReverse=False, condTest=condTest, condState=condState)
        if retExact: # transExact and lenExact > prevLenExact+lenCh:
            lenExact = retExact[1]
            transExact = retExact[0]
            lenTrans = transExact[1]
            word = transExact[0]+chPrefix+chSuffix+transExact[0]
            lenW = len(word)
            return (word, lenW)            
        if not retPref and not suffixOnly and not exactOnly:
            retPref = trie_find(fsmdict, word, lenWord=lenW, prefixSuffix=1, needReverse=False, condTest=condTest, condState=condState)
        if not retSuff and not prefixOnly and not exactOnly:
            retSuff = trie_find(fsmdict, revWord, lenWord=lenW, prefixSuffix=0, needReverse=False, condTest=condTest, condState=condState)
        doPref = retPref != None
        doSuf = retSuff != None

        if retPref and retSuff:
            tooLong = (retPref[1]+retSuff[1] > lenW)
            if retPref[0][2] < retSuff[0][2]: # priority 
                doPref = True
                doSuf = False
                if tooLong:
                    retSuf = None
            elif retPref[0][2] > retSuff[0][2]:
                doSuf = True
                doPref = False
                if tooLong:
                    retPref = None
            elif retPref[1] >= retSuff[1] and tooLong:
                doPref = True
                doSuf = False
                retSuff = None
            elif tooLong:
                doPref = False
                doSuf = True
                retPref = None
        if doPref:
            lenPrefix = retPref[1]
            transPrefix = retPref[0]
            lenTrans = 0
            if transPrefix and lenPrefix > prevLenPrefix+lenCh:
                lenTrans = transPrefix[1]
                prevLenPrefix = lenTrans
                transPrefix = transPrefix[0]
                word2 = transPrefix+chPrefix+word[lenPrefix:]
                word = word2
                lenW = lenW-lenPrefix+lenTrans+lenCh
                revWord = ''.join(reversed(word))
                cont=True
                retPref = None
        if doSuf:
            lenSuffix = retSuff[1]
            transSuffix = retSuff[0]
            lenTrans = 0
            if transSuffix and lenSuffix > prevLenSuffix+lenCh:
                lenTrans = transSuffix[1]
                prevLenSuffix = lenTrans
                transSuffix = transSuffix[0]
                word = word[:lenW-lenSuffix]+chSuffix+transSuffix
                #print lenW, lenSuffix, lenTrans, lenCh
                lenW = lenW-lenSuffix+lenTrans+lenCh
                revWord = ''.join(reversed(word))
                cont=True
                retSuff = None
        if word[-1] == chPrefix or word[0] == chSuffix:
            return (word, lenW)
    return (word, lenW)


## Basic search function

def remove_duplicates(lst):
    return list(OrderedDict.fromkeys(lst))

@jit(nopython=True, cache=True, nogil=True)
def np_cosine(v1, v2):  return np_dot(v1, v2) / (np_norm(v1) * np_norm(v2))

@jit(nopython=True, cache=True, nogil=True)
def norm_matrix(a):
    a_norm = np.zeros(shape=(a.shape[0],), dtype=np.float32)
    for i in range(a.shape[0]):
        a_norm[i] = np_norm(a[i])
        if a_norm[i] == 0:
            a_norm[i] = 0.00001
    return a_norm

@jit(nopython=True, cache=True, nogil=True)
def cosine_search(a, b, a_norm=None, b_norm=None, k=5):
    """
    given a and b, find the k highest cosine matches
    """
    if a_norm is None:
        a_norm0 = norm_matrix(a)
        a_norm = a_norm0.reshape(a_norm0.shape[0], 1)

    if b_norm is None:
        b_norm = norm_matrix(a)
    if k > b.shape[0]:
        k = b.shape[0] 
    dist_matrix = 1.0 - np_dot(a, b.transpose())/(a_norm*b_norm)
    len_dist_matrix = dist_matrix.shape[0]
    ret = np.zeros(shape=(len_dist_matrix, k*2), dtype=np.float32)

    for i in range(len_dist_matrix):
        scores = dist_matrix[i]
        if k == 1:
            index = np.argmin(scores)
            j = 0
            score = 1.0-scores[index]
            ret[i][j] = index
            ret[i][j+1] = score
        else:
            arg_index = np.argsort(scores)[:k]
            j = 0
            for id in arg_index:
                score = 1.0-scores[id]
                ret[i][j] = id
                ret[i][j+1] = score
                j+=2
    return ret
    

def np_memmap(f, mode, dtype, shape):
    return np.memmap(f, mode=mode, dtype=dtype, shape=shape)

class Object(object):
    pass

class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# TODO - shard out the search results
class OntoSearchResults:
    """
    Interator class that yields search results in pairs of [(index,
    score) ...] for each row corresponding to the match. Search
    results are not guranteed to be kept up to date. There is no
    gurantee that the search results won't change as you interate
    through because other processes or threads may update the search
    results.
    """
    def __init__(self, match=None, search_file=None, k=100,  max_tokens=0, items=None, filter_fns=None, rng_step = 1000, kb=None, token_type=None, search_space=None, search_space_score_fn=None, name=None, dtype=np.float32):
        """
        WARNING: setting the items does not check if the items is a token_type
        """
        self.dtype = dtype
        self.match = match
        self.search_file = search_file
        if search_file is None and match is None:
            self.match = []
        self.k = k
        self.max_tokens = max_tokens
        self.items = items
        self.filter_fns = filter_fns
        self.rng_step = rng_step
        self.kb = kb
        self.token_type = token_type
        if isinstance(search_space, ListType) or isinstance(search_space, RangeType):
          search_space = dict([(a,1) for a in search_space])                
        self.search_space = search_space
        self.search_space_score_fn = search_space_score_fn

    def __len__(self):
        if self.match:
            num_tokens = len(self.match)
        else:
            num_tokens = self.max_tokens
        if self.items:
            num_tokens = len(self.items)
        return num_tokens


    def set_items(self, items):
        """
        WARNING: setting the items does not check if the items is a token_type
        WARNING: Do not change items while you are iterating through the results
        """
        self.items = items
        return self

    def set_kb(self, kb):
        self.kb = kb
        return self


    def set_token_type(self, token_type=None):
        self.token_type = token_type
        return self

    def set_filter_fns(self, fiter_fns=None):
        self.filter_fns = filter_fns
        return self

    def set_search_space(self, search_space=None):
       if isinstance(search_space, ListType) or isinstance(search_space, RangeType):
          search_space = dict([(a,1) for a in search_space])                
       self.search_space = search_space
       return self

    def set_search_space_score_fn(self, search_space_score_fn=None):
        self.search_space_score_fn = search_space_score_fn
        return self


    def set_search_file(self, search_file, delete_old_search_file=False):
        dtype = self.dtype
        if self.match is None: 
            match = np_memmap(self.search_file, mode='r+', dtype=dtype, shape=(self.max_tokens, 2*self.k))                
        else:
            match = self.match
        search_dat = np_memmap(search_file, mode='w+', dtype=dtype, shape=(self.max_tokens, 2*self.k))                
        search_dat[:] = match[:]
        search_dat = None
        match = None
        if self.search_file and delete_old_search_file:
            os.unlink(self.search_file)            
        self.search_file = search_file
        self.match = None
        return self

    def clone(self):
        self2 = OntoSearchResults()
        self2.match = self.match
        self2.search_file = self.search_file
        self2.k = self.k
        self2.max_tokens = self.max_tokens
        self2.items = copy.copy(self.items)
        self2.filter_fns = self.filter_fns
        self2.rng_step = self.rng_step
        self2.kb = self.kb
        self2.token_type = self.token_type
        self2.search_space = self.search_space
        return self2

    def __iter__(self):
        # read in parrallel to speed up search results
        dtype = self.dtype
        def reader(match, items, rng2, max_rng2, search_file):
            search_dat = np_memmap(search_file, mode='r+', dtype=dtype, shape=(self.max_tokens, 2*self.k))                
            match[rng2:max_rng2] = search_dat[items]
            search_dat = None

        items = self.items
        kb = self.kb
        token_type = self.token_type
        search_space_score_fn = None
        if self.search_space:
            search_space = self.search_space                
            search_space_score_fn = self.search_space_score_fn
        else:
            search_space=None
        
        filter_fns=self.filter_fns

        if self.match is not None:
            match = self.match
            if items is None:
                items = range(len(match))
            for b in items:
                row = match[b]
                if not row:
                    continue
                if (kb and not kb.token_exists(b)):
                   yield []
                   continue
                if filter_fns:
                    for filter_fn in filter_fns:
                        if not row:
                            break
                        row = filter_fn(self, b,row)

                if kb:
                    if search_space_score_fn:
                        ret = [(j, search_space_score_fn(score, search_space[j])) for j, score in row if j >= 0 and kb.token_exists(j) and (not search_space or j in search_space)] #  and (token_type is None or (kb.token_type(j) == token_type))
                    else:
                        ret = [(j, score) for j, score in row if j >= 0 and kb.token_exists(j) and (not search_space or j in search_space)] #   and (token_type is None or (kb.token_type(j) == token_type))
                else:
                    if search_space_score_fn:
                        ret = [(j, search_space_score_fn(score, search_space[j])) for j,score in row if j >= 0 and (not search_space or j in search_space)]
                    else:
                        ret = [(j, score) for j,score in row if j >= 0 and (not search_space or j in search_space)]
                yield ret
        else:
            # TODO, work properly with items that are ranges
            rng_step2 = self.rng_step
            rng_step = rng_step2*5
            #i = 0 
            k = self.k
            num_tokens = self.max_tokens
            if items:
                num_tokens = len(items)
            search_file = self.search_file
            id = 0
            for rng in range(0, num_tokens, rng_step):
                max_rng = min(rng + rng_step, num_tokens)
                if items is None:
                    items0 = range(rng,max_rng)
                else:
                    items0 = items[rng:max_rng]
                num_tokens2 = (max_rng-rng)
                match = [None]*num_tokens2
                if num_tokens2 < rng_step2:
                    search_dat = np_memmap(search_file, mode='r+', dtype=dtype, shape=(self.max_tokens, 2*self.k))                
                    if items is None:
                        match[:] = search_dat[rng:max_rng]                    
                    else:
                        match[:] = search_dat[items[rng:max_rng]]
                    search_dat = None
                else:
                    workers=[]
                    for rng2 in range(0, num_tokens2, rng_step2):                    
                        max_rng2 = min(rng2 + rng_step2, num_tokens2)
                        if items is None:
                            worker = threading.Thread(target=reader, args=(match, range(rng+rng2,rng+max_rng2), rng2, max_rng2, search_file))
                        else:
                            worker = threading.Thread(target=reader, args=(match, items[rng+rng2:rng+max_rng2], rng2, max_rng2, search_file))
                        workers.append(worker)
                        worker.start()
                    for worker in workers:
                        worker.join()
                if kb:
                    match = [[(int(row[ki*2]), row[ki*2+1]) for ki in range(k) if row[ki*2] >= 0 and kb.token_exists(int(row[ki*2]))] for row in match] #  and (token_type is None or (kb.token_type(int(row[ki*2])) == token_type))
                else:
                    match = [[(int(row[ki*2]), row[ki*2+1]) for ki in range(k) if row[ki*2] >= 0] for row in match]

                for b, row in zip(items0, match):
                    if filter_fns:
                        for filter_fn in filter_fns:
                            if not row:
                                break
                            row = filter_fn(self, b,row)
                    if search_space_score_fn:
                        ret= [(j, search_space_score_fn(score, search_space[j])) for j, score in row  if (search_space is None or j in search_space)]
                    else:
                        ret= [(j, score) for j, score in row  if (search_space is None or j in search_space)]
                    yield ret
                    id += 1

        search_space=None


class OntoMemmap(np.memmap):
    """
    Extension of numpy's memmap class which provides for a cached
    subset of the items in a local in-memory cache. The local cache
    is suitable for use with pytorch as a parameter.  There are three
    ways to update the on disk memmap, using the 3 flush_types. The
    local cache can be used as an on device (cuda) view of the out of
    core array.

    NOTE: There is minimal support for multiprocessing and threading
    safe operations: there is only locking when write_lock is true and
    locking occurs only for flushing and writing the data.
    
    NOTE: We use python's shared memory arrays as underlying storage
    instead of shared pytorch tensors because they are more stable in
    Windows when sharing between processes and threads.

    """
    SYNC_TYPE_FLUSH_ONLY = 0
    SYNC_TYPE_FLUSH_AND_UPDATE = 1
    SYNC_TYPE_UPDATE_ONLY = 2

    FLUSH_TYPE_OVERWRITE = 0
    FLUSH_TYPE_AVG = 1
    FLUSH_TYPE_DELTA = 2

    np_dtype_to_torch_dtype = {'int16': torch.int16, 'int32':torch.int32, 'int64': torch.int64, 'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64}

    def __new__(subtype, filenameOrState, dtype=np.float32,
                shape=None, local_map=None, flush_type=FLUSH_TYPE_OVERWRITE, device=None, padding_idx=-1, write_lock=False, tlock=None):

        if isinstance(filenameOrState, TupleType):
            filename, dtype, shape, flush_type, tlock, viewname, device, padding_idx, write_lock, local_map_len, local, local_orig, local_map, local_view, local_map_tensor, need_flush = filenameOrState
            if tlock is None:
                tlock = threading.RLock()
        else:            
            filename = filenameOrState
            viewname = None
            if tlock is None:
                tlock = threading.RLock()
            local_map_len = 0
            local_view = None
            local_map_tensor = None
            local = None
            local_orig = None
            need_flush = None
            
        if isinstance(dtype, ListType):
            flush_type = OntoMemmap.FLUSH_TYPE_OVERWRITE
            
        mode='r+' 
        offset=0
        order='C'
        if not os.path.exists(filename):
            mode = 'w+'
        if shape[0] == -1:
            if not os.path.exists(filename):
                if len(shape) > 1:
                    shape = (1, shape[1])
                else:
                    shape = (1,)
            else:
                if len(shape) > 1:
                    size = np.dtype(dtype).itemsize*shape[1]
                    shape = (int(os.path.getsize(filename)/size), shape[1])
                else:
                    shape = (int(os.path.getsize(filename)/np.dtype(dtype).itemsize),)
        self = super(OntoMemmap, subtype).__new__(subtype=subtype, filename=filename, dtype=dtype, mode=mode, offset=offset, shape=tuple(shape), order=order)
        dir_path = os.getcwd()
        if "/" in dir_path:
            dir_path = dir_path + "/"
        else:
            dir_path = dir_path + "\\"
        if self.filename.startswith(dir_path):
            filename = self.filename.replace(dir_path, "")
        else:
            filename = self.filename
        self.base_filename = filename
        self.viewname = viewname
        self.flush_type = flush_type
        self.tlock = tlock
        self.write_lock = write_lock
        self.device = device
        self.padding_idx = padding_idx
        if type(local_map) is IntType:
           if local_map == -1:
              local_map = list(range(shape[0]))
           else:
              local_map = list(range(local_map))
        if local_map is None or local_map_len is None:
            local_map_len = 0
        self.local = local
        self.local_orig = local_orig
        self.local_map = local_map
        self.local_map_len = local_map_len
        with torch.no_grad():            
            self.local_view = local_view
            self.local_map_tensor = local_map_tensor
        self.need_flush = need_flush
        if self.local_map_len in (0, None) and local_map is not None and type(local_map) is not ListType:
            self.local_map_len = 0
            self.need_flush = need_flush
            self.local_map = local_map
            self.local = local
            print (self.local)
            self.local_orig = local_orig
            self.local_map_len = len(self.local_map)
            if str(self.dtype) in self.np_dtype_to_torch_dtype:
                with torch.no_grad():            
                    if self.device is None or self.device == torch.device('cpu'):
                        if self.local_view is None:
                            self.local_view = nn.Parameter(torch.from_numpy(self.local))
                        if self.local_map_tensor is None:
                            self.local_map_tensor = torch.from_numpy(self.local_map)
                    else:
                        if self.local_view is None:
                            self.local_view = nn.Parameter(torch.from_numpy(self.local)).to(self.device)
                        if self.local_map_tensor is None:
                            self.local_map_tensor = torch.from_numpy(self.local_map).to(self.device)
        elif self.local_map_len in (0, None) and local_map is not None:
            self.set_local_map(local_map, padding_idx)
        elif self.local_map_len in (0, None) and self.local_view is None:
            if str(self.dtype) in self.np_dtype_to_torch_dtype:
                with torch.no_grad():            
                    self.local_view = Parameter(torch.from_numpy(self)).to(self.device)
                    self.local_map_tensor = None
        self.set_viewname()
        return self

    def resize(self, shape):
        if shape[0] < self.shape[0]: 
            if not threading.current_thread() is threading.main_thread():
              raise RuntimeError("Can't shrink OntoMemmap in multhreading mode")
            if len(self.shape) == 2:
                assert shape[1] == self.shape[1]
                size = self.dtype.itemsize
                self.base.resize(shape[0]*shape[1]*size)
            else:
                size = self.dtype.itemsize
                self.base.resize(shape[0]*size)
            #super(OntoMemmap, self).resize(shape, refcheck=False)
        return self.clone(shape=shape)

    def clone(self, shape=None):
        return OntoMemmap(self.get_state(shape=shape))
        
    def get_state(self, shape=None):
        if self.device is not None and self.device != torch.device ('cpu'):
            logger.info("WARNING: You are trying to get a state for a cuda view. Converting the state to a cpu view. Views can only be shared if they are cpu views.")
            self.flush()
            return (self.base_filename, self.dtype, (shape is None and self.shape) or shape, self.flush_type,  self.tlock, None, None, self.padding_idx, self.write_lock, None, None, None, None, None, None, None)
        return (self.base_filename, self.dtype, (shape is None and self.shape) or shape, self.flush_type, self.tlock, self.viewname, self.device, self.padding_idx, self.write_lock, self.local_map_len, self.local, self.local_orig, self.local_map, self.local_view, self.local_map_tensor, self.need_flush)

    def to(self, device=None, dtype=None, non_blocking=False):
        if self.device == device:
            return 
        if self.local_map is None or self.local_map_len == 0:
            if str(self.dtype) in self.np_dtype_to_torch_dtype:
                with torch.no_grad():            
                    self.local_view = self.local_view.to(self.device)
            self.device = device
            return
        if str(self.dtype) in self.np_dtype_to_torch_dtype:
            with torch.no_grad():
                if device in (torch.device('cpu'), None) and self.device not in (torch.device('cpu'), None):
                    self.local[:] = self.local_view.detach().cpu()
                    self.local_view = nn.Parameter(torch.from_numpy(self.local))
                    self.local_map_tensor = torch.from_numpy(self.local_map)
                else:
                    self.local_view = self.local_view.to(device=device, dtype=dtype, non_blocking=non_blocking)
                    self.local_map_tensor = self.local_map_tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.device = device


    def set_local_map(self, local_map, padding_idx=-1):
        self.flush()
        if local_map is None:
            self.local_map_len = 0
            self.padding_idx = padding_idx
            self.local_map = None
            self.local = None
            self.local_orig = None
            self.need_flush = None
            self.local_view = None
            self.local_map_tensor = None
            if str(self.dtype) in self.np_dtype_to_torch_dtype:
                with torch.no_grad():            
                    self.local_view = Parameter(torch.from_numpy(self)).to(self.device)
            return

        if type(local_map) is IntType:
           if local_map == -1:
              local_map = list(range(self.shape[0]))
           else:
              local_map = list(range(local_map))
        ids = [i for i in local_map if i > -1]
        ids.sort()
        local_map_len = max(ids)+1
        self.padding_idx = padding_idx
        self.local_map_len = 0
        self.local_map = np.zeros(shape=(local_map_len,), dtype=np.int64)
        last_idx = len(ids)
        if self.padding_idx not in local_map:
           self.padding_idx = last_idx
        self.local_map.fill(last_idx)
        #self.local_to_id
        self.need_flush = np.zeros(shape=(last_idx,), dtype=np.int32)
        self.need_flush.fill(0)
        for j, i in enumerate(ids):
            self.local_map[i] = j
        self.local_orig = None
        self.local_view = None
        self.local_map_tensor = None
        # one item at the end for the out of range item for error cases
        if len(self.shape) == 2:
            local = np.zeros(shape=(last_idx+1,self.shape[1]),dtype=self.dtype)
        else:
            local = np.zeros(shape=(last_idx+1,),dtype=self.dtype) 
        self.local = local
        self.local[:len(self.local)-1] = self.get_items_from_disk(ids)
        self.local[-1] = 0.0 # set the out of range item to 0.0
        if self.flush_type == OntoMemmap.FLUSH_TYPE_DELTA:
            if len(self.shape) == 2:
                local_orig = np.zeros(shape=(last_idx+1,self.shape[1]), dtype=self.dtype)
            else:
                local_orig = np.zeros(shape=(last_idx+1,), dtype=self.dtype)
            self.local_orig = local_orig
            self.local_orig[:] = self.local
        self.local_map_len = local_map_len
        self.set_viewname()
        if str(self.dtype) in self.np_dtype_to_torch_dtype:
            with torch.no_grad():
                if self.device in (None, torch.device('cpu')):
                    self.local_view = nn.Parameter(torch.from_numpy(self.local))
                    self.local_map_tensor = torch.from_numpy(self.local_map)
                else:
                    self.local_view = nn.Parameter(torch.from_numpy(self.local)).to(self.device)
                    self.local_map_tensor = torch.from_numpy(self.local_map).to(self.device)

    def set_write_lock(self, wlock=False):
        self.write_lock = wlock

    def set_viewname(self):
        if self.viewname is not None:
            return
        if self.local_map is not None:
            self.viewname = self.base_filename +  ":" + str(random.randint(0, 1000000))
        else:
            self.viewname = self.base_filename

    def id_to_local_id(self, i):
        if i < self.local_map_len:
            return self.local_map[i]
        return -1

    def __array_finalize__(self, obj):
        super(OntoMemmap, self).__array_finalize__(obj)
        if hasattr(obj, 'viewname'):
            self.base_filename = obj.base_filename
            self.viewname = obj.viewname
            self.device = obj.device
            self.flush_type = obj.flush_type
            self.tlock = obj.tlock
            self.write_lock = obj.write_lock
            self.padding_idx = obj.padding_idx
        if hasattr(obj, 'local_map'):
            self.local = obj.local
            self.local_view = obj.local_view
            self.local_map_tensor = obj.local_map_tensor
            self.local_map = obj.local_map
            self.local_map_len = obj.local_map_len
            self.need_flush = obj.need_flush
            self.local_orig = obj.local_orig

    def __getitem__(self, i):
        getitem = super(OntoMemmap, self).__getitem__
        if hasattr(self, 'local_map') and self.local_map_len > 0:
            if isinstance(i, SliceType):
                start = (i.start or 0)
                stop = (i.stop or self.shape[0])
                return self.get_items(range(start, stop))
            elif  isinstance(i, ListType) or isinstance(i, RangeType):
                return self.get_items(i)
            elif isinstance(i, TupleType) and len(i) == 2:
                i2 = i[0]
                j = i[1]
                if i2 < 0:
                    i2 = self.shape[0]+i2
                if j < 0:
                    j = self.shape[1]+j
                if i2 < self.shape[0] and j < self.shape[1]:
                    if self.local_map_len >i2 and self.local_map[i2] >=0 and self.local_map[i2] != self.padding_idx:
                        if self.device is not None and self.device != torch.device ('cpu'):
                            with torch.no_grad():
                                local = self.local_view.data
                                ret = local[self.local_map[i2], j]
                                ret = ret.cpu().numpy()
                        else:
                            local = self.local
                            ret = local[self.local_map[i2], j]
                        return ret
            elif isinstance(i, IntType):
                i = int(i)
                if i < 0: i = self.shape[0]+i
                if self.local_map_len >i and self.local_map[i] >=0 and self.local_map[i] != self.padding_idx:
                    if self.device is not None and self.device != torch.device ('cpu'):
                        with torch.no_grad():
                            local = self.local_view.data
                            ret = local[self.local_map[i]]
                            ret = ret.cpu().numpy()
                    else:
                        local = self.local
                        ret = local[self.local_map[i]]
                    return ret
        if type(i) in (np.int32, np.int64, np.uint32, np.uint64, IntType) or (isinstance(i, TupleType) and len(i) == 2):
            return getitem(i) # np.array(, copy=True)
        else:
            if isinstance(i, SliceType):
                start = (i.start or 0)
                stop = (i.stop or self.shape[0])
                i = range(start, stop)
            return self.get_items_from_disk(i)

    def get_items_from_disk(self, items, rng_step=1000):
        getitem = super(OntoMemmap, self).__getitem__   
        def reader(ret, j_items, i_items):
            ret[i_items] = getitem(j_items)
        num_tokens = len(items)
        if len(self.shape) == 2:
            ret = np.zeros(shape=(num_tokens, self.shape[1]), dtype=self.dtype)        
        else:
            ret = np.zeros(shape=(num_tokens,), dtype=self.dtype)        
        num_tokens = len(items)
        workers = []
        for rng in range(0, num_tokens, rng_step):
            max_rng = min(rng + rng_step, num_tokens)
            worker = threading.Thread(target=reader, args=(ret, items[rng:max_rng], range(rng, max_rng)))
            workers.append(worker)
            worker.start()
        for worker in workers:
            worker.join()
        return ret


    def get_items(self, items, rng_step=1000):
        """ get copy of array for the items """
        getitem = super(OntoMemmap, self).__getitem__
        def reader(ret, i_items, j_items):
            ret[j_items] = getitem(i_items)

        # main
        jArr = []
        iArr = [] 
        if not items:
            if len(self.shape) == 2:
                return np.zeros(shape=(1, self.shape[1]), dtype=self.dtype)        
            else:
                return np.zeros(shape=(1,), dtype=self.dtype)        
        num_tokens = len(items)
        if len(self.shape) == 2:
            ret = np.zeros(shape=(num_tokens, self.shape[1]), dtype=self.dtype)        
        else:
            ret = np.zeros(shape=(num_tokens,), dtype=self.dtype)        
        for j, i in enumerate(items):
            if i >= self.shape[0]:
                raise RuntimeError("index out of bound when setting multiple items")
            if (i < self.local_map_len and self.local_map[i]>=0 and self.local_map[i] != self.padding_idx):
                ret[j] = self[i] # this will call __getitem__
            else:
                if iArr != []:
                    prev_i_range = iArr[-1]
                    prev_j_range = jArr[-1]
                    if i != prev_i_range[1] or (prev_i_range[1]-prev_i_range[0] +1) >= rng_step:
                        iArr.append([i,i+1])
                        jArr.append([j,j+1])
                    else:
                        prev_i_range[1]+=1
                        prev_j_range[1]+=1
                else:
                    iArr.append([i,i+1])
                    jArr.append([j,j+1])

        workers = []
        prev_j_items = None
        prev_i_items = None
        for j_items, i_items in zip(jArr, iArr):
            if prev_j_items is None:
                if j_items[1] > j_items[0]+1:
                    prev_j_items = range(j_items[0], j_items[1])
                    prev_i_items = range(i_items[0], i_items[1])
                else:
                    prev_j_items = [j_items[0]]
                    prev_i_items = [j_items[0]]
            else:
                if type(prev_j_items) == RangeType or j_items[1] > j_items[0]+1:
                    worker = threading.Thread(target=reader, args=(ret, prev_i_items, prev_j_items))
                    workers.append(worker)
                    worker.start()
                    if j_items[1] > j_items[0]+1:
                        prev_j_items = range(j_items[0], j_items[1])
                        prev_i_items = range(i_items[0], i_items[1])
                    else:
                        prev_j_items = [j_items[0]]
                        prev_i_items = [j_items[0]]
                else:
                    prev_j_items.append(j_items[0])
                    prev_i_items.append(i_items[0])
        if prev_j_items is not None:
            worker = threading.Thread(target=reader, args=(ret, prev_i_items, prev_j_items))
            workers.append(worker)
            worker.start()
        for worker in workers:
            worker.join()
        return ret

    def __setitem__(self, i, val):
        setitem = super(OntoMemmap, self).__setitem__
        if self.write_lock:
            tlock = self.tlock or DummyLock()
        else:
            tlock = DummyLock()
        with tlock:
            if hasattr(self, 'local_map') and self.local_map_len > 0:
                if isinstance(i, SliceType):
                    start = (i.start or 0)
                    stop = (i.stop or self.shape[0])
                    self.set_items(list(range(start, stop)), val)
                    return
                elif isinstance(i, ListType):
                    self.set_items(i, val)
                    return
                elif isinstance(i, RangeType):
                    self.set_items(i, val)
                    return
                elif isinstance(i, TupleType) and len(i) == 2:
                    i2 = i[0]
                    j = i[1]
                    if i2 < 0:
                        i2 = self.shape[0]+i2
                    if j < 0:
                        j = self.shape[1]+j
                    if i2 < self.shape[0] and j < self.shape[1]:
                        if self.local_map_len >i2  and self.local_map[i2] >=0 and self.local_map[i2] != self.padding_idx:
                            self.local[self.local_map[i2], j]= val
                            if self.device not in (None, torch.device('cpu')):
                                with torch.no_grad():
                                    self.local_view.data[self.local_map[i2], j]= torch.tensor(val).to(self.device)
                            self.need_flush[self.local_map[i2]] = 1
                            return
                elif isinstance(i, IntType):
                    if i < 0: i = self.shape[0]+i
                    if self.local_map_len >i  and self.local_map[i] >=0 and self.local_map[i] != self.padding_idx:
                        self.local[self.local_map[i]]= val
                        if self.device not in (None, torch.device('cpu')):
                            with torch.no_grad():
                                self.local_view.data[self.local_map[i]]= torch.tensor(val).to(self.device)
                        self.need_flush[self.local_map[i]] = 1
                        return
            if type(i) in (np.int32, np.int64, np.uint32, np.uint64, IntType) or (isinstance(i, TupleType) and len(i) == 2):
                setitem(i, val)
            else:
                if isinstance(i, SliceType):
                    start = (i.start or 0)
                    stop = (i.stop or self.shape[0])
                    i = range(start, stop)
                self.set_items_to_disk(i, val)

    def set_items_to_disk(self, items, vals, rng_step=1000):
        setitem = super(OntoMemmap, self).__setitem__
        if self.write_lock:
            tlock = self.tlock or DummyLock()
        else:
            tlock = DummyLock()
        def writer(vals, j_items, i_items):
            v = vals[i_items]
            if type(v) is torch.Tensor:
                v = v.data.cpu().numpy()
            with torch.no_grad():
                with tlock:
                    setitem(j_items, v)
            v = None

        num_tokens = len(items)
        workers = []
        for rng in range(0, num_tokens, rng_step):
            max_rng = min(rng + rng_step, num_tokens)
            worker = threading.Thread(target=writer, args=(vals, items[rng:max_rng], slice(rng, max_rng)))
            workers.append(worker)
            worker.start()
        for worker in workers:
            worker.join()


    def set_items(self, items, vals, rng_step=1000, need_flush=1):
        """ set vals to items """
        setitem = super(OntoMemmap, self).__setitem__
        if self.write_lock:
            tlock = self.tlock or DummyLock()
        else:
            tlock = DummyLock()

        def writer(vals, i_items, j_items):
            v = vals[j_items]
            if type(v) is torch.Tensor:
                v = v.data.cpu().numpy()
            with torch.no_grad():
                with tlock:
                    setitem(i_items, v)
            v = None


        # main
        vals = np.asarray(vals)
        jArr = []
        iArr = [] 
        for j, i in enumerate(items):
            if i >= self.shape[0]:
                raise RuntimeError("index out of bound when setting multiple items")
            if (i < self.local_map_len and self.local_map[i]>=0 and self.local_map[i] != self.padding_idx):
                self.need_flush[self.local_map[i]] = need_flush
                self[i] = vals[j] # this will call __setitem__
            else:
                if iArr != []:
                    prev_i_slice = iArr[-1]
                    prev_j_slice = jArr[-1]
                    if i != prev_i_slice[1] or (prev_i_slice[1]-prev_i_slice[0] +1) >= rng_step:
                        iArr.append([i,i+1])
                        jArr.append([j,j+1])
                    else:
                        prev_i_slice[1]+=1
                        prev_j_slice[1]+=1
                else:
                    iArr.append([i,i+1])
                    jArr.append([j,j+1])
        workers = []
        prev_j_items = None
        prev_i_items = None
        for j_items, i_items in zip(jArr, iArr):
            if prev_j_items is None:
                if j_items[1] > j_items[0]+1:
                    prev_j_items = slice(j_items[0], j_items[1])
                    prev_i_items = slice(i_items[0], i_items[1])
                else:
                    prev_j_items = [j_items[0]]
                    prev_i_items = [j_items[0]]
            else:
                if type(prev_j_items) == SliceType or j_items[1] > j_items[0]+1:
                    worker = threading.Thread(target=writer, args=(vals, prev_i_items, prev_j_items))
                    workers.append(worker)
                    worker.start()
                    if j_items[1] > j_items[0]+1:
                        prev_j_items = slice(j_items[0], j_items[1])
                        prev_i_items = slice(i_items[0], i_items[1])
                    else:
                        prev_j_items = [j_items[0]]
                        prev_i_items = [j_items[0]]
                else:
                    prev_j_items.append(j_items[0])
                    prev_i_items.append(i_items[0])
        if prev_j_items is not None:
            worker = threading.Thread(target=writer, args=(vals, prev_i_items, prev_j_items))
            workers.append(worker)
            worker.start()
        for worker in workers:
            worker.join()



    def __array_wrap__(self, arr, context=None):
        return super(OntoMemmap, self).__array_wrap__(arr, context)

    def supress_flush(self, supress=1):
        self._supress_flush = supress

    def flush(self, sync_type=SYNC_TYPE_FLUSH_ONLY, rng_step=1000, force=0):
        """
        sync_type == SYNC_TYPE_FLUSH_ONLY will only flush and update local changed values, and not update other local values from disk.
        sync_type == SYNC_TYPE_FLUSH_AND_UPDATE will flush updated local changed values, and update all other local values from disk.
        sync_type == SYNC_TYPE_UPDATE_ONLY will ignore changed local values and sync everything from disk to local cache.

        NOTE: If we do not operate in write_lock mode, it is possible
        that we will flush while changing local values.

        NOTE: We might be able to do the flushing in parallel with
        multi-threads to improve performance.
        """
        """ set vals to items """

        if isinstance(self.base, mmap.mmap) and hasattr(self, 'local_map') and self.local_map_len > 0 and (force==0 or not hasattr(self, '_supress_flush') or self._supress_flush == 0):
            try:
                need_flush = self.need_flush.any()
            except:
                need_flush = True
            if need_flush or sync_type:
                padding_idx = self.padding_idx
                if self.write_lock:
                    tlock = self.tlock or DummyLock()
                else:
                    tlock = DummyLock()
                with tlock:
                    with torch.no_grad():
                        flush_from_cuda = False
                        local = self.local
                        local_map = self.local_map
                        local_orig = self.local_orig
                        ids0 = None
                        if self.device not in (None, torch.device('cpu')):
                            local = self.local_view.data
                            flush_from_cuda = True
                            # this view is a cuda view. get the data from cuda local cache
                        if need_flush and sync_type != OntoMemmap.SYNC_TYPE_UPDATE_ONLY:
                            ids = [(j,i) for j,i in enumerate(self.local_map) if i != padding_idx and self.need_flush[i]==1]
                            ids0 = [(j,i) for j,i in enumerate(self.local_map) if i != padding_idx and self.need_flush[i]==0]
                            if ids:
                                self.need_flush[[i for j,i in ids]] = 0
                                if self.flush_type == OntoMemmap.FLUSH_TYPE_DELTA:
                                    if not flush_from_cuda:
                                        val = self.get_items_from_disk([j for j,i in ids]) + (local[[i for j, i in ids]]-local_orig[[i for j, i in ids]])
                                    else:
                                        val = self.get_items_from_disk([j for j,i in ids]) + (local[[i for j, i in ids]].cpu().numpy()-local_orig[[i for j, i in ids]])
                                    self.set_items_to_disk([j for j,i in ids], val)
                                    if not flush_from_cuda:
                                        local[[i for j,i in ids]] = val
                                    else:
                                        local[[i for j,i in ids]] = torch.from_numpy(val).to(self.device)
                                        self.local[[i for j,i in ids]] = val
                                    local_orig[[i for j,i in ids]] = val
                                    val = None
                                elif self.flush_type == OntoMemmap.FLUSH_TYPE_AVG:
                                    if flush_from_cuda:
                                        val = (self.get_items_from_disk([j for j,i in ids]) + local[[i for j,i in ids]])/2.0 
                                    else:
                                        val = (self.get_items_form_disk([j for j,i in ids]) + local[[i for j,i in ids]].cpu().numpy())/2.0 
                                    self.set_items_to_disk([j for j,i in ids], val)
                                    local[[i for j,i in ids]] = val
                                    if not flush_from_cuda:
                                        local[[i for j,i in ids]] = val                                            
                                    else:
                                        local[[i for j,i in ids]] = torch.from_numpy(val).to(self.device)
                                        self.local[[i for j,i in ids]] = val
                                    val = None
                                else:
                                    if flush_from_cuda:
                                        val = local[[i for j,i in ids]].cpu().numpy()
                                    else:
                                        val = local[[i for j,i in ids]]
                                    self.set_items_to_disk([j for j,i in ids], val)
                                    if flush_from_cuda:
                                        self.local[[i for j,i in ids]] = val
                        if sync_type != OntoMemmap.SYNC_TYPE_FLUSH_ONLY:
                            if ids0 is not None:
                                ids = [j for j,i in ids0]
                                ids2 = [i for j,i in ids0]
                            else:
                                ids = [j for j,i in enumerate(local_map) if i != padding_idx]
                                ids2 = [i for j,i in enumerate(local_map) if i != padding_idx]
                            val = self.get_items_from_disk(ids)
                            if not flush_from_cuda:
                                local[ids2] = val
                            else:
                                local[ids2] = torch.from_numpy(val).to(self.device)
                            if self.flush_type == OntoMemmap.FLUSH_TYPE_DELTA:
                                local_orig[ids2] = val
                            if flush_from_cuda:
                                self.local[ids2] = val
                            self.need_flush[:] = 0
        super(OntoMemmap, self).flush()        

    def fill(self, val):
        if hasattr(self, 'local_map') and self.local_map_len > 0:
            if self.write_lock:
                tlock = self.tlock or DummyLock()
            else:
                tlock = DummyLock()
            with tlock:
                self.local.fill(val)
                if self.local_orig is not None:
                    self.local_orig.fill(val)
                super(OntoMemmap, self).fill(val)       
        else:
            super(OntoMemmap, self).fill(val)      
        self.flush()


    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        we flush before we do a whole array ufunc. but this doesn't
        gurantee that the array might not change due to another
        process/thread writing to the array while performing the
        ufunc.
        """

        args = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, OntoMemmap):
                input_.flush()
                args.append(input_.view(np.memmap))
            else:
                args.append(input_)
        outputs = kwargs.pop('out', None)
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, OntoMemmap):
                    out_no.append(j)
                    output.flush()
                    out_args.append(output.view(np.memmap))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout
        results = super(OntoMemmap, self).__array_ufunc__(ufunc, method,
                                                 *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented
        if method == 'at':
            return
        if ufunc.nout == 1:
            results = (results,)
        results2 = []
        for result, output in zip(results, outputs):
            if isinstance(output, OntoMemmap):
                if output is not result and hasattr(self, 'local_map') and self.local_map_len > 0:
                    padding_idx = self.padding_idx
                    ids = [j for j,i in enumerate(output.local_map) if i != padding_idx]
                    output[ids] = result[ids]
                    output.need_flush.fill(0)
                elif output.base != result.base:
                    output[:] = result
                results2.append(output)
            else:
                results2.append(np.asarray(result))

        results = results2
        return results[0] if len(results) == 1 else results

    def view(self, dtype=None, type_=None):
        if dtype is None and type_ is None:
            return OntoMemmap(self.get_state())
        return super(OntoMemmap, self).view(dtype or type_)
    
        

class OntoEmbedding(nn.Embedding):
    """
    An extension of pytorch's embeddings that maps to a numpy memmap
    file. The actual data is not serialized nor stored in the
    state_dict. Best practice is to use a relative path or to save the
    memmap file in the same directory as the pytorch pickel file.

    The .weight parameter (and its underlying .data) is not used, so
    do not access it directly. Instead, use .get_view() to get
    the underlying data view.

    TODO: Consider whether we should use the tlock from the
    underlying OntoMemmap or create a lock specifically for the
    OntoEmbedding class.
    """

    view_to_data = {}  # keep the actual mmap handles as a class variable to prevent torch.save from serializing

    def __init__(self, path_or_state, num_embeddings=None, embedding_dim=None, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False, 
                 indexer_leaves_for_nodes=None, indexer_leaves=None, indexer_nodes=None, indexer_nodes_matrix=None, indexer_nodes_matrix_norm=None,
                 local_map=None, flush_type=None, _weight=None, views=None, shape=None):
        """warning! for the weight parameter, We do not tie parameters to the weight. Instead, this is a copy."""


        assert sparse==False, 'OntoEmbeddings do not support sparse mode'

        if _weight is not None:
           # check if it is an embedding or a tensor
           #num_embeddings,embedding_dim =  _weight.weight.data.size()
           num_embeddings=_weight.num_embeddings
           embedding_dim=_weight.embedding_dim
           padding_idx=_weight.padding_idx
           max_norm=_weight.max_norm
           norm_type=_weight.norm_type
           scale_grad_by_freq=_weight.scale_grad_by_freq

        if indexer_leaves_for_nodes is None: indexer_leaves_for_nodes = [] 
        if indexer_leaves is None: indexer_leaves = [] 
        if indexer_nodes is None: indexer_nodes = [] 
        if indexer_nodes_matrix is None: indexer_nodes_matrix = [] 
        if indexer_nodes_matrix_norm is None: indexer_nodes_matrix_norm = [] 

        self.indexer_leaves_for_nodes = indexer_leaves_for_nodes
        self.indexer_leaves = indexer_leaves
        self.indexer_nodes = indexer_nodes
        self.indexer_nodes_matrix = indexer_nodes_matrix
        self.indexer_nodes_matrix_norm = indexer_nodes_matrix_norm

        if views is None:
            self.views = {}
        else:
            self.views = views
        self.view_cnt = 0
        if type(path_or_state) is TupleType:
            path_or_state = OnotMemmap(path_or_state)
        if type(path_or_state) is OntoMemmap:
            view_data = path_or_state
            self.path = view_data.base_path
            viewname = self.viewname = view_data.viewname
            if viewname not in self.view_to_data:
                self.view_to_data[viewname] = view_data
                self.views[viewname] = 1
            if shape is None:
                self.shape = list(copy.copy(view_data.shape))
            else:
                self.shape = list(shape)
            num_embeddings = self.shape[0]
            embedding_dim = self.shape[1]
            if self.path not in self.view_to_data: self.get_view(self.get_view_name(self.path))
        else:
            path = path_or_state
            if shape is None:
                self.shape = [num_embeddings, embedding_dim]
            else:
                self.shape = list(shape)
            dir_path = os.getcwd()
            if "/" in dir_path:
                dir_path = dir_path + "/"
            else:
                dir_path = dir_path + "\\"
            if path.startswith(dir_path):
                path = path.replace(dir_path, "")
            self.path = path
            if not os.path.exists(path):
                 os.mkdir(path)
            filename = path + "/embedding.mmap"
            self.filename = self.viewname = filename
            view_data = self.get_view(self.get_view_name(self.filename)) 
            self.shape = list(copy.copy(view_data.shape))
            num_embeddings = self.shape[0]
            embedding_dim = self.shape[1]
            if local_map is not None:
                view_data =  OntoMemmap(self.filename, np.float32, shape=self.shape, local_map=local_map, flush_type=flush_type, device=None, tlock=view_data.tlock)
                self.viewname = viewname = view_data.viewname
                self.view_to_data[viewname] = view_data
                self.views[viewname] = 1
        super(OntoEmbedding, self).__init__(num_embeddings=0, embedding_dim=0, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_id = num_embeddings + padding_idx
        self.padding_idx = padding_idx 
        if _weight is not None:
            self.copy_into_view(_weight)
        self.weight = None # we don't use the weight variable to store the weights. local_view's weights are used dynamically instead.

    def clone(self):
        return OntoEmbedding(self.get_view(), shape=self.shape, padding_idx=self.padding_idx,
                             max_norm=self.max_norm, norm_type=self.norm_type, scale_grad_by_freq=self.scale_grad_by_freq,
                             sparse=self.sparse, flush_type=self.flush_type, views=self.views, 
                             indexer_leaves_for_nodes=self.indexer_leaves_for_nodes, indexer_leaves=self.indexer_leaves, indexer_nodes=self.indexer_nodes, 
                             indexer_nodes_matrix=self.indexer_nodes_matrix, indexer_nodes_matrix_norm=self.indexer_nodes_matrix_norm)

    def init_after_torch_load(self):
        pass

    def resize(self, new_len, supress_flush=1):
        # resize will work properly for multithreading, but multiprocessing may be an issue
        if type(new_len) is TupleType:
            assert self.shape[1] == new_len[1]
            new_len = new_len[0]
        view_data = self.get_view(self.filename)
        tlock = view_data.tlock or DummyLock()
        with tlock:
            # only do resizing for the main view. the other views will just be cloned and get a new shape. use renew_view to set the shape of all other views.
            self.shape[0] = new_len
            view_data = view_data.resize((new_len, self.shape[1]))
            self.view_to_data[self.filename] = view_data
            self.renew_view(self.filename)
            view_data = None
            for viewname in self.views:
                if viewname != self.filename:
                    self.renew_view(viewname)
            for obj in gc.get_referrers(self.shape):
                if hasattr(obj, 'embedding_dim'):
                    obj.shape[0] = self.shape[0]
                    obj.num_mebeddings = self.shape[0]
        return self

    def copy_into_view(self, _weight, viewname=None, rng_step=1000000):
        if viewname is None:
           viewname = self.filename
        view_data = self.get_view(viewname)
        if type(_weight) is OntoEmbedding:
            data = _weight.get_view()[:]            
        elif type(_weight) is nn.Embedding:
            data = _weight.weight.data.cpu().numpy()
        elif type(_weight) is torch.Tensor:
            data = _weight.data.cpu().numpy()
        else:
            data = _weight
        len_data = len(data)
        if len_data > self.shape[0]:
            self.resize((len_data, self.shape[1]))
            view_data = self.get_view(viewname)
        with torch.no_grad():
            for rng in range(0, data.shape[0], rng_step):
                max_rng = min(rng + rng_step, data.shape[0])
                view_data[rng:max_rng] = data[rng:max_rng]
                self.flush()
                view_data = self.renew_view(viewname)
        return self

    def reset_parameters(self, rng_step=1000000, init_fn=None, items=None):
        if (self.num_embeddings == 0 and self.embedding_dim == 0):
            return 
        view_data = self.get_view(self.filename)
        if init_fn is None:
            init_fn = nn.init.normal_
        tlock = view_data.tlock or DummyLock()
        with tlock:
            if items is None:
                num_items = self.shape[0]
                items = list(range(self.shape[0]))
            else:
                num_items = len(items)
            with torch.no_grad():
                for rng in range(0, num_items, rng_step):
                    max_rng = min(rng + rng_step, num_items)
                    # if we work in pytorch.tensor space instead of numpy space there seems to be some memory issues.
                    view_data[items[rng:max_rng]]  = init_fn(torch.from_numpy(np.zeros(shape=(max_rng-rng, self.shape[1]), dtype=view_data.dtype))).numpy()
                    view_data.flush()
                    view_data = self.get_view(self.filename)
                if self.padding_idx is not None: # this doesn't clear the padding values in the local map. TODO. 
                   view_data[self.padding_idx].fill(0)
            self.flush_all_views(sync_type=OntoMemmap.SYNC_TYPE_UPDATE_ONLY)


    def __del__(self):
        self.del_all_views()
        s = super(self)
        if hasattr(s, '__del__'): s.__del__()
        
    def extra_repr(self):
        return "'"+ self.filename + "', " + super(OntoEmbedding, self).extra_repr() 


    def save_pretrained(self):
         self.flush()
         torch.save(self, self.filename.replace(".mmap", ".bin"))

    @classmethod
    def from_pretrained(cls, filename_or_path, freeze=True):
         pass

    def get_view(self, viewname_or_local_map=None, supress_flush=1, flush_type=OntoMemmap.FLUSH_TYPE_OVERWRITE, device=None):
        """ get a OntoMemmap view using a viewname or a local_map """
        if viewname_or_local_map is not None and type(viewname_or_local_map) is not StringType:
           viewname = self.get_view_name(viewname_or_local_map, flush_type=flush_type, device=device)
        else:
           viewname = viewname_or_local_map
        if viewname is None:
            viewname = self.viewname
            if viewname not in self.view_to_data:
                if viewname is not None: logger.info("WARNING! This OntoMemmp is referencing an old view. Using default view for the whole mmap file")
                if viewname in self.views: del self.views[viewname] 
                viewname = self.filename
                viewname = self.get_view_name(viewname)
        elif viewname not in self.view_to_data:
            if viewname == self.filename:
                return self.view_to_data[self.get_view_name(viewname)]                
            raise RuntimeError("This OntoMemmap view has not been set or is old.")
        self.view_cnt += 1
        if self.view_cnt > 100000:
            self.view_cnt = 0
            self.renew_view(viewname, supress_flush)
        return self.view_to_data[viewname]

    def renew_view(self, viewname, supress_flush=1):
        """ do some cleanup of the mmap data for the particular view. NOTE:  using .clone as a hack to force Windows to free up mmap data. """
        view_data =self.view_to_data[viewname]
        view_data_clone = view_data.clone(shape=self.shape) 
        if supress_flush == 1 or sys.getrefcount(view_data) <= 2:
            # if the only references to view_data is in the
            # view_to_data table and here, we don't want to flush
            # on delete in every cycle, so supress flush for the
            # old view. the clone will handle flushing.
            view_data.supress_flush() 
        self.view_to_data[viewname] = view_data_clone
        return view_data_clone

    def get_view_name(self, local_map=None, flush_type=OntoMemmap.FLUSH_TYPE_OVERWRITE, device=None):
        """ creates a mmap view and returns the view name for the local map """
        if (type(local_map) is StringType and local_map == self.filename):
            tlock = None
            for view2 in self.views.keys():
                if view2 in self.view_to_data:
                    view_data2 = self.view_to_data[view2]
                    tlock = view_data2.tlock
                    break
            if self.filename not in self.view_to_data:
                if device not in (None, torch.device('cpu')):
                    logging.log("Warning! converting the whole embedding to cuda may take up lots of GPU memory. Try using a view.")  
                view_data = self.view_to_data[self.filename] = OntoMemmap(self.filename, np.float32, shape=self.shape, local_map=None, flush_type=flush_type, device=device, tlock=tlock)
                self.views[self.filename] = 1
            viewname = self.filename
        elif local_map is None:
            viewname = self.viewname
        elif type(local_map) is StringType:
            viewname = local_map
        else:
            try:
                local_map = local_map.clone().cpu().numpy().tolist()
            except:
                pass
            view_data = self.get_view()
            view_data = OntoMemmap(self.filename, np.float32, shape=self.shape, local_map=local_map, flush_type=flush_type, device=device, tlock=view_data.tlock).clone()
            viewname = view_data.viewname
            self.view_to_data[viewname] = view_data
            self.views[viewname] = 1
            viewname = view_data.viewname
        view_data = self.view_to_data[viewname]
        return view_data.viewname

    def set_view(self, local_map, flush_type=OntoMemmap.FLUSH_TYPE_OVERWRITE, device=None):
        """ sets the default mmap view for this OntoEmbedding using local_map """
        self.flush(sync_type=OntoMemmap.SYNC_TYPE_FLUSH_ONLY)
        view_data =  self.get_view(self.get_view_name(local_map, flush_type, device))
        self.viewname = view_data.viewname
        return self

    def del_view(self, viewname_or_mmap = None):        
        """ delete the mmap view """
        if isinstance(viewname_or_mmap, OntoMemmap):
           viewname = viewname_or_mmap.viewname
        else:
           viewname = viewname_or_mmap
        if viewname is None:
            viewname = self.viewname
        if viewname in self.views: 
            if sys.getrefcount(self.views) <= 1:
                try:
                    del self.views[viewname]
                except:
                    pass
        if viewname in self.view_to_data: 
            view_data = self.view_to_data[viewname]
            if sys.getrefcount(self.view_to_data) <= 1:
                try:
                    del self.view_to_data[viewname]        
                except:
                    pass
            view_data.flush(sync_type=OntoMemmap.SYNC_TYPE_FLUSH_ONLY)
            view_data = None

    def del_all_views(self):
        """ delete all mmap view """
        for key in list(self.views.keys()):
            self.del_view(key)
        if sys.getrefcount(self.views) <= 1:
            try:
                self.views.clear()
            except:
                pass

    def flush_all_views(self, sync_type=OntoMemmap.SYNC_TYPE_FLUSH_AND_UPDATE, force=1):
        for key in self.view_to_data:
            if key == self.filename or key.startswith(self.filename+":"):
                self.flush(key, sync_type=sync_type, force=force)

    def flush(self, viewname=None, sync_type=OntoMemmap.SYNC_TYPE_FLUSH_AND_UPDATE, force=1):
        if viewname is None:
            viewname = self.viewname
        view_data = self.view_to_data[viewname]
        view_data.flush(sync_type=sync_type, force=force)


    def cuda(self, device=None, viewname=None):
        if device is None: device = torch.device ('cuda')
        self.get_view(viewname).to(device)
        super(OntoEmbedding, self).cuda(device)
        return self


    def cpu(self, viewname=None):
        self.get_view(viewname).to(torch.device('cpu'))
        super(OntoEmbedding, self).cpu()
        return self

    def add_module(self, name, module):
        raise RuntimeError("Cannot add modoule to OntoEmbedding")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        logger.info("WARNING! state_dict of an OntoEmbedding does not have have data. Data is saved to .mmap file. Access the data directly with .get_view()") 
        return super(OntoEmbedding, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, dict, strict=True):
        #logger.info("WARNING! load_state_dict for OntoEmbedding will set the data in the .mmap file.")
        super(OntoEmbedding, self).load_state_dict_dict(dict, **kargs)
        return self

    def requires_grad_(self, _requires_grad=True):
        self.get_view(viewname).local_view.requires_grad_(_requires_grad)
        super(OntoEmbedding, self).requires_grad_(_requires_grad)
        return self

    def apply(self, fn, viewname=None):
        view = self.get_view(viewname).local_view
        fn(view)
        super(OntoEmbedding, self).apply(fn)
        return self

    def parameters(self, recurse=True, viewname=None):
        yield self.get_view(viewname).local_view
        for param in super(OntoEmbedding, self).parameters(recurse):
            yield param

    def named_parameters(self, prefix='', recurse=True, viewname=None):
        yield prefix+'local_view', self.get_view(viewname).local_view
        for name, param in super(OntoEmbedding, self).named_parameters(prefix, recurse):
            yield name, param

    def eval(self, viewname=None):
        self.get_view(viewname).local_view.eval()
        super(OntoEmbedding, self).eval()
        return self

    def half(self, viewname=None):
        self.get_view(viewname).local_view.half()
        super(OntoEmbedding, self).half()
        return self

    def float(self, viewname=None):
        self.get_view(viewname).local_view.float()
        super(OntoEmbedding, self).float()
        return self

    def double(self, viewname=None):
        self.get_view(viewname).local_view.double()
        super(OntoEmbedding, self).double()
        return self

    def train(self, mode=True, viewname=None):
        self.get_view(viewname).local_view.train(mode=mode)
        super(OntoEmbedding, self).train(mode=mode)
        return self

    def type(self, dst_type, viewname=None):
        self.get_view(viewname).local_view.type(dst_type=dst_type)
        super(OntoEmbedding, self).type(dst_type=dst_type)
        return self

    def zero_grad(self, viewname=None):
        self.get_view(viewname).local_view.zero_grad()
        super(OntoEmbedding, self).zero_grad()
        return self

    def to(self, device=None, viewname=None, dtype=None, non_blocking=False):
        view_data = self.get_view(viewname)
        if viewname == self.filename and device not in (None, torch.device('cpu')):
            logging.log("Warning! converting the whole embedding to cuda may take up lots of memory. Try using a view.")
        view_data.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return super(OntoEmbedding, self).to(device=device, dtype=dtype, non_blocking=non_blocking)

    def forward(self, input, viewname=None):
        view_data = self.get_view(viewname)
        if hasattr(view_data, 'local_map') and view_data.local_map_len > 0:
            padding_idx = view_data.padding_idx
            out_of_bound = (input >= view_data.local_map_tensor.shape[0]).nonzero().flatten()
            if len(out_of_bound) > 0:
                with torch.no_grad():
                    input = input.clone().detach()
                    input.data[out_of_bound] = padding_idx # set everything out of bound to just be a pad
            return F.embedding(
                view_data.local_map_tensor[input], view_data.local_view, padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            return F.embedding(
                input, view_data.local_view, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)

    def init_weight(self, model):
        """ Initialize and prunes weights if needed. """
        if hasattr(model, '_init_weight'):
            # Initialize weights
            model.apply(model._init_weight) # do we need to remove self from the children list there?
            # Prune heads if needed
            if model.config.pruned_heads:
                model.prune_heads(model.config.pruned_heads)
            self.reset_parameters(init_fn=model._init_weight)
        else:
            self.reset_parameters()
        # Tie weights if needed
        self.tie_weight(model)



    # indexing and search methods
    @staticmethod
    def create_indexer_multithreading(items, self, saved_obj, num_parents=None, cluster_size=None, kb=None):
        logger.info("indexing multithreading " + str(len(items))) # 
        self.create_indexer(items=items, level=0, do_hiearchical=False, num_threads=1, num_parents=num_parents, cluster_size=cluster_size, saved_obj=saved_obj, lock=False, kb=kb)

    def create_indexer(self, items=None, rng_step = 10000,  level=0, do_hiearchical=True, num_threads=None, num_parents=None, cluster_size=None, saved_obj=None, lock=True, kb=None):
        """ 
        creates a hiearchical index of the underlying embeddings for
        cosine similarity searches.  finds random N nodes as the head
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

        logger.info("indexer_leaves set count " + str(sum([len(s) for s in saved_obj.indexer_leaves_for_nodes[level]])))
        logger.info("step 5")
        if do_hiearchical and len(indexer_nodes) >= rng_step: # 1000:
            logger.info("indexer hiearchical")
            if orig_indexer_leaves is None:
                self.create_indexer(items=indexer_nodes, level=level+1, rng_step=rng_step, saved_obj=saved_obj, lock=False, kb=kb)
            else:
                self.create_indexer(items=[orig_indexer_leaves[i] for i in indexer_nodes], level=level+1, rng_step=rng_step, saved_obj=saved_obj, lock=False, kb=kb) 
        if saved_obj == self:
           torch.save(self, self.filename.replace(".mmap", ".pt"))

    def default_search_file(self, k=100):
        return self.filename.replace(".mmap", ".search_"+str(k)+".mmap")

    @staticmethod
    def search_multithreading(search_file, vs, vs_fn, vs_callback_data, level, items, k, k_clusters, update_search_file, max_tokens, token_type, search_space, self, kb, search_indexer_nodes_only):
        logger.info("searching multithreading " + " " + str(len(items)) + " " + str(max_tokens))
        self.search(search_file=search_file, vs=vs, vs_fn=vs_fn, vs_callback_data=vs_callback_data, items=items, clear_search_file=True, level=level, k=k, k_clusters=k_clusters, update_search_file=update_search_file, max_tokens=max_tokens, num_threads=1, token_type=token_type, search_space=search_space, kb=kb, search_indexer_nodes_only=search_indexer_nodes_only)


    def search(self, items=None, vs=None, vs_norm=None, k=100, k_clusters=10, rng_step = 10000, search_file=None, indexer_nodes=None, level=None, match=None, update_search_file=False, cnt=None, num_threads=None, use_search_file=False, clear_search_file=False, init_search_file=False,  need_update=None, max_tokens=None, search_space=None, token_type=None, vs_fn=None, vs_callback_data=None, kb=None, search_indexer_nodes_only=False):
        """
        Search the array index in several modes (and optionally cache
        away search results in search_file for fast lookup):
        
        - items contains the index into the mmmap file. items controls
          what is returned as a result either in memory or in a cache
          file. If in a cache file, the cache file may have many more
          entries than just the items, but only items are returned.

        - search_space controls what is searched against. For example,
          if items = [1, 2, 3], search_space = [3, 4, 5], then we find
          search results for 1 and 2 and 3, against similar entries in
          the set [3, 4, 5].

        - (vectors) vs and vs_fn permits searching the array of self
          against the vectors. when vs is None and vs_fn is None, will
          default to searching of the array of self against itself,
          based on everything (if not items list is set) or the items
          list if it is set. 

          - NOTE: searching with both an vs and an items parameter
          will just search the vs parameter. the items are considered
          indexes to the final match returned and not self
          file. however, this can be slow if the number of items being
          updated is large. in some cases, recreating the search cache
          may be faster. For example, if item i, has results (0.2, j),
          then j will also be updated with (0.2, i) if appropriate
          
        - if multiprocessing is performed, the state of the object
          will be saved to disk before doing the multiprocessing. This
          will flush the cache.


        """

        # this function is the most expensive part of search. 
        def sort_and_save(match, search_file, shape, update_search_file, itemsSet, k, self_search):

            tt = time.time()
            if itemsSet is not None:
                itemsSet = dict([(a, 1) for a in itemsSet])
            itemsSet2={}
            if search_file is not None:
                search_dat = np_memmap(search_file, mode='r+', dtype=np.float32, shape=shape)        
            for times in (1,2):
                #logger.info("sort and save times " + str(times) + " " + str(itemsSet.keys()))
                for b in itemsSet.keys():
                    match_item = match[b]
                    if not match_item:
                        continue
                    # combine with disk if we have to
                    unique=False
                    if search_file is not None:
                        #search_dat = np_memmap(search_file, mode='r', dtype=float32, shape=shape)        
                        if search_dat[b][0] != 0.0 and search_dat[b][1] != 0.0: # we should use the score, and not the index? 
                            search_dat2 = np.array(search_dat[b])
                            matchHash = dict(match_item)
                            for ki2 in range(k):
                                if search_dat2[ki2*2] != 0.0 and search_dat2[ki2*2+1] != 0.0:
                                    if matchHash.get(int(search_dat2[ki2*2])) is None:
                                        matchHash[int(search_dat2[ki2*2])] = search_dat2[ki2*2+1] 
                                else:
                                    break
                            match_item = list(matchHash.items())
                            matchHash = None
                            unique=True
                            search_dat2 = None

                    # sort and truncate 
                    if update_search_file:
                        for c, score in match_item:
                            if  b != c and c not in itemsSet:
                                if match[c] is None:
                                    match[c] = []
                                match[c].append((b, score))
                                itemsSet2[c] = 1
                    if not unique:
                        matchHash = dict(match_item)
                        match_item = list(matchHash.items())
                        matchHash = None
                        unique = True
                    if _pyVer==2:
                        match_item.sort(lambda a, b: cmp(b[1], a[1]))
                    else:
                        match_item.sort(key=lambda a: a[1], reverse=True)
                    if self_search:
                        match_item = [(b, 1.0)] + [a for a in match_item if a[0] != b] 
                    len_match_item = len(match_item)
                    if len_match_item > k:                        
                        match_item = match_item[:k]
                        len_match_item = k
                    match[b] = match_item
                    #logger.info("sorted " + str(match[b]))

                    # save the value to disk if we have a file; we expect the data in 1.0-cosine format. 
                    if search_file is not None:
                        search_dat2 = search_dat[b]
                        # save it all at once. 
                        #match3 = [float(a) for a in flatten(match_item)] 
                        #if len_match_item < k: match3.extend([0.0]*((k-len_match_item)*2)) 
                        #search_dat2[:] = match3
                        for ki2 in range(k):
                            if ki2 >= len_match_item:
                                search_dat2[ki2*2] = 0.0
                                search_dat2[ki2*2+1] = 0.0
                            else:
                                search_dat2[ki2*2] = match_item[ki2][0]
                                search_dat2[ki2*2+1] = float(match_item[ki2][1])
                                #c = match_item[ki2][1]
                        search_dat2 = None
                        match[b] = None
                    match_item = None
                if not update_search_file or not search_file:
                    break
                update_search_file = False
                itemsSet = itemsSet2
            return time.time()-tt
        
        ## main
        sort_and_save_time = 0
        if vs is not None and vs_fn is not None:
            raise RuntimeError("either pass in vs or vs_fn, but not both")
        if items is None and vs_fn is not None:
            raise RuntimeError("must pass in an items list if a vs_fn is defined")
        view_data = self.get_view(self.filename)
        if self.indexer_nodes == [] or self.indexer_nodes is None:
            tlock = view_data.tlock or DummyLock()
        else:
            tlock = DummyLock()
        with tlock:
            if self.indexer_nodes == [] or self.indexer_nodes is None:
                self.indexer_nodes.clear()
                self.indexer_leaves_for_nodes.clear()
                self.indexer_leaves.clear()
                self.indexer_nodes_matrix.clear()
                self.indexer_nodes_matrix_norm.clear()
                self.create_indexer(num_threads=num_threads, kb=kb)
        view_data = None
        # do some sanity checks
        #if clear_search_file and vs is None and vs_fn is None and ((not items and token_type is None) or (self.kb and len(items) == self.shape[0])):
        #    init_search_file = True
        if search_file is not None:
            use_search_file = True
        if init_search_file:
            clear_search_file = True
            use_search_file = True
        if update_search_file:
            use_search_file = True
            clear_search_file = True
        if clear_search_file:
            use_search_file = True

        if use_search_file and search_file is None:
            search_file = self.default_search_file(k=k)

        if items is None and update_search_file:
            update_search_file = False

        if not items:
            orig_all_items = True
        else:
            orig_all_items = False            
        num_embeddings = self.shape[0]
        if isinstance(search_space, ListType) or isinstance(search_space, RangeType):
           search_space = dict([(a,1) for a in search_space])                
        else:
           search_space = search_space

        if (vs is None and vs_fn is None) and token_type is not None and items is None and kb:
            if kb and len([a for a in range(num_embeddings) if not kb.token_exists(a)]) == num_embeddings:
                return OntoSearchResults(match=None, search_file=search_file, k=k, max_tokens=max_tokens, items=items, kb=kb, token_type=token_type, search_space=search_space)
        if cnt is None:
            cnt=[0]
        if match is None or level is None:
            # base case: we only get here from an initial search
            #call, including from a multiprocessing call.  

            #logger.info ("search k "+ str(k) + " k_clusters " + str(k_clusters))
                
            if max_tokens is None:
                if vs is None:
                    max_tokens = num_embeddings
                elif vs is not None:
                    max_tokens = vs.shape[0]
                else:
                    raise RuntimeError("need max_tokens if vs_fn is passed in")
            shape = (max_tokens, 2*k)
            if items: 
                iHash = {}
                items2 = []
                for i in items:
                    if (vs is None and vs_fn is None) and kb and not kb.token_exists(i):
                        logger.warning('searching deleted items. ignoring ' + str(i))
                        continue
                    if i in iHash:
                        logger.warning('items being searched has duplicates. removing duplicates ' + str(i))
                    else:
                        iHash[i] = 1
                        items2.append(i)
                items = items2
                # TODO: what happens when we are searching from a vs list using an items list?
            if  not search_file and use_search_file:
                raise RuntimeError("need to pass in search_file if caching")
            #logger.info("searching with file "+str( search_file))
            if isinstance(search_file, StringType) and os.path.exists(search_file) and not clear_search_file:
                if orig_all_items: items = None
                return OntoSearchResults(match=None, search_file=search_file, k=k, max_tokens=max_tokens, items=items, kb=kb, token_type=token_type, search_space=search_space)

            if items and search_file and clear_search_file and os.path.exists(search_file) and not init_search_file:
                # if we are going to refresh the search, clear out the items
                #logger.info("clearing the file for items at level " + str(level) + " " + str(shape))
                search_dat = np_memmap(search_file, mode='r+', dtype=np.float32, shape=shape)                        
                for i in items:
                    # hack to just clear out the first item                    
                    search_dat[i][0] = 0.0 #index
                    search_dat[i][1] = 0.0 #score
                search_dat = None

            # TODO: need a better way to clear the cache, because it
            # is not intuitive that we should init the cache when we
            # pass no items and set clear_search_file
            elif search_file and ((not os.path.exists(search_file)) or init_search_file):# or ((not items) and clear_search_file)
                if os.path.exists(search_file):
                        #logger.info("unlinking " + search_file)
                    os.unlink(search_file)
                np_memmap(search_file, mode='w+', dtype=np.float32, shape=shape)        
            match=[None]*max_tokens
            if level is None:
                level=len(self.indexer_nodes)-1
        if max_tokens is None:
            max_tokens = len(match)
        shape = (max_tokens, 2*k)
        num_tokens = 0
        if vs is None and vs_fn is None:
            if items:
                num_tokens = min(len(items),num_embeddings)
            else:
                num_tokens = num_embeddings
        else:
            if items and vs is None:
                num_tokens = len(items)
            elif items:
                num_tokens = min(len(items),vs.shape[0])
            else:
                num_tokens = vs.shape[0]
        #logger.info("num_tokens in search = " + str(num_tokens))
        # take care of some base cases
        if num_threads is None:
            num_threads = max(4, multiprocessing.cpu_count())
        if num_threads > 1 and (search_file is None):
            num_threads = 1
        for num_threads2 in range(num_threads, 1, -1):
            if num_tokens/num_threads2 < rng_step:
                num_threads = max(1, num_threads2-1)
        self_search = (vs is None and vs_fn is None)

        if vs is None and vs_fn is not None:
            #let's see if we can take advantage of stepping through
            #very large vectors using the vs_fn
            if len(items) > 10*rng_step:
                #logger.info("stepping vs_fn")
                rng_step2 = 10*rng_step
                for rng2 in range(0, num_tokens, rng_step2):
                    max_rng2 = min(rng2 + rng_step2, num_tokens)
                    search_result = self.search(k=k, vs=vs_fn(self, vs_callback_data, items[rng2:max_rng2]), items=items[rng2:max_rng2], search_file=search_file, clear_search_file=True, max_tokens=max_tokens, init_search_file=init_search_file, token_type=token_type, search_space=search_space, kb=kb)
                    init_search_file=False
                return OntoSearchResults(match=None, search_file=search_file, k=k, max_tokens=max_tokens, items=items, kb=kb, token_type=token_type, search_space=search_space)

        if search_file and  num_threads > 1:
            if not items:
                items2 = list(range(num_tokens))
            else:
                items2 = copy.copy(items)
            if vs is None and vs_fn is None:
                items2 = [i for i in items2 if not kb or kb.token_exists(i)]

            #if search_space:
            #    items = [i for i in items if search_space.get(i) is not None]
            if vs is None:
                random.shuffle(items2) # shuffle it to spread out equal access to the self cache across the num_threads
            id_num = 0
            workers = []
            rng_step2 = int(num_tokens/num_threads)
            for rng in range(0, num_tokens, rng_step2):
                if id_num < num_threads-1:
                    max_rng = min(rng + rng_step2, num_tokens)
                else:
                    max_rng = num_tokens
                vs2 = None
                if vs is not None:
                    vs2 = vs[rng:max_rng]
                #logger.info("startig search 2 for " + str(len(items[rng:max_rng])))
                aThread = threading.Thread(target=OntoEmbedding.search_multithreading, args=(search_file, vs2, vs_fn, vs_callback_data, level, items2[rng:max_rng], k, k_clusters, update_search_file, max_tokens, token_type, search_space, self, kb, search_indexer_nodes_only))
                workers.append(aThread)
                aThread.start()
                if id_num == num_threads-1:
                    break
                id_num += 1
            for aThread in workers:
                if aThread:
                    aThread.join()
            workers = None

            if orig_all_items: items = None
            return OntoSearchResults(match=None, search_file=search_file, k=k, max_tokens=len(match), items=items, kb=kb, token_type=token_type, search_space=search_space)

        # another base case where the items list or number of tokens
        # is just very long, we update the search file in chunks in a
        # single process
        if search_file and  num_tokens > rng_step:
            # let's just walk through the steps without save and sort and then seperately save and sort
            if not items:
                for rng in range(0, num_tokens, rng_step):
                    max_rng = min(rng + rng_step, num_tokens)
                    # just search and fill in the match array, don't do any saving
                    vs2 = None
                    if vs is not None:
                        vs2 = vs[rng:max_rng]
                        items2 = range(rng, max_rng)
                    elif vs_fn is not None:
                        raise RuntimeError("must pass in an items list if a vs_fn is defined")
                    else:
                        items2 = [i for i in range(rng, max_rng) if not kb or kb.token_exists(i)]
                    #if search_space:
                    #    items = [i for i in items if search_space.get(i) is not None]
                    self.search(items=items2, level=level, k_clusters=k_clusters, match=match, search_file=None, update_search_file=update_search_file, cnt=cnt, vs=vs2, max_tokens=max_tokens, num_threads=1, token_type=token_type, search_space=search_space, kb=kb, search_indexer_nodes_only=search_indexer_nodes_only)
                    # should this be indented over?
                    sort_and_save_time += sort_and_save(match, search_file, shape, update_search_file, range(0, num_tokens), k, self_search)
            else:
                for rng in range(0, num_tokens, rng_step):
                    max_rng = min(rng + rng_step, num_tokens)
                    # just search and fill in the match array, don't do any saving
                    vs2 = None
                    if vs is not None:
                        vs2 = vs[rng:max_rng]
                    elif vs_fn is not None:
                        vs2=vs_fn(self, vs_callback_data,items[rng:max_rng])
                    self.search(items=items[rng:max_rng], level=level, k_clusters=k_clusters, match=match, search_file=None, update_search_file=update_search_file, cnt=cnt, vs=vs2, max_tokens=max_tokens, num_threads=1, token_type=token_type, search_space=search_space, kb=kb, search_indexer_nodes_only=search_indexer_nodes_only)
                    # should this be indented over?
                    sort_and_save_time += sort_and_save(match, search_file, shape, update_search_file, items, k, self_search)
            #logger.info("sort_and_save_time 1 " + str(sort_and_save_time))
            if orig_all_items: items = None                    
            return OntoSearchResults(match=None, search_file=search_file, k=k, max_tokens=len(match), items=items, kb=kb, token_type=token_type, search_space=search_space)

        # main searching code. *** we only get here if there is no
        # search file or the number of items or search_file length or
        # vs length is less than or equal to range_step.

        if vs_fn is not None:
            vs = vs_fn(self, vs_callback_data, items)
            vs_fn = None
            vs_callback_data = None
        if vs is not None and vs_norm is None:
            vs_norm = norm_matrix(vs)
        #logger.info(("*** search ", level, self.indexer_nodes_matrix))
        if not indexer_nodes:
            indexer_nodes = list(range(len(self.indexer_nodes[level])))
            b_matrix = self.indexer_nodes_matrix[level]
            b_norm = self.indexer_nodes_matrix_norm[level]
        else:
            b_matrix = self.indexer_nodes_matrix[level][indexer_nodes]
            b_norm = self.indexer_nodes_matrix_norm[level][indexer_nodes]
        #logger.info("search1 " + str(num_tokens) + " levels "+ str(level) + " parents " + str(len(parents)))
        #test_zero(None, None, a_matrix, "search1 a_matrix")
        len_parents = len(indexer_nodes)
        ks = [None]*len_parents
        if search_indexer_nodes_only and level == 0:
           indexer_leaves_for_nodes = [[self.indexer_leaves_for_nodes[level][p][0]] for p in indexer_nodes]           
        else:
           indexer_leaves_for_nodes = [self.indexer_leaves_for_nodes[level][p] for p in indexer_nodes]
        for rng in range(0, num_tokens, rng_step):
            max_rng = min(rng + rng_step, num_tokens)
            if vs is None:
                if items is None:
                    if kb:
                        a_items = [i for i in range(rng, max_rng) if kb.token_exists(i)]
                    else:
                        a_items = list(range(rng, max_rng))                        
                    #if search_space:
                    #    a_items = [i for i in a_items if search_space.get(i) is not None]
                    if not a_items:
                        continue
                    a_matrix = self.get_view(self.filename)[a_items]
                    #b_matrix = self.get_view(self.filename).get_range(rng, max_rng)
                    a_norm = norm_matrix(a_matrix)
                    a_norm = a_norm.reshape(a_norm.shape[0], 1)                        
                else:
                    if kb:
                        a_items = [i for i in items[rng:max_rng] if kb.token_exists(i)]
                    else:
                        a_items = items[rng:max_rng]  
                    # when items are passed in, we assume it's already
                    # been filtered for the search space, so we don't
                    # filter here.
                    if not a_items:
                        continue
                    a_matrix = self.get_view(self.filename)[a_items]
                    a_norm = norm_matrix(a_matrix)
                    a_norm = a_norm.reshape(a_norm.shape[0], 1)                        
            else:
                a_items = list(range(rng, max_rng))
                a_matrix = vs[rng:max_rng]
                a_norm = vs_norm[rng:max_rng]
                a_norm = a_norm.reshape(a_norm.shape[0], 1)                        
            #test_zero(None, None, b_matrix, "search1 b_matrix")
            ns = cosine_search(a_matrix, b_matrix, a_norm, b_norm, k=k_clusters)
            for b, vals in enumerate(ns):
                for id in range(0, vals.shape[0], 2):
                    ki = int(vals[id])
                    if ks[ki] is None:
                        ks[ki] = []
                    ks[ki].append(a_items[b])

        a_matrix = None
        a_norm = None
        b_matrix = None
        b_norm = None
        if level > 0:
            # do hiearchical search
            for ki, bs in enumerate(ks):            
                if bs:
                    if vs is None:
                        self.search(indexer_nodes=indexer_leaves_for_nodes[ki], level=level-1, items=bs, k_clusters=k_clusters, match=match, search_file=None, update_search_file=update_search_file,  cnt=cnt, max_tokens=max_tokens, num_threads=1, token_type=token_type, search_space=search_space, kb=kb, search_indexer_nodes_only=search_indexer_nodes_only) 
                    else:
                        self.search(vs=vs[bs], vs_norm=vs_norm[bs], indexer_nodes=indexer_leaves_for_nodes[ki], level=level-1, items=bs, k_clusters=k_clusters, match=match, search_file=None, update_search_file=update_search_file, cnt=cnt, max_tokens=max_tokens, num_threads=1, token_type=token_type, search_space=search_space, kb=kb, search_indexer_nodes_only=search_indexer_nodes_only)
                ks[ki] = None
        else:
            # do simple search
            for indexer_leaves, bs2 in zip(indexer_leaves_for_nodes, ks):
                if kb:
                    indexer_leaves = [i for i in indexer_leaves if kb.token_exists(i)]
                if search_space:
                    indexer_leaves = [i for i in indexer_leaves if search_space.get(i) is not None]                    
                if not indexer_leaves or not bs2: continue
                cnt[0] += 1
                if cnt[0] > 1000:#
                    cnt[0] = 0
                    sort_and_save_time += sort_and_save(match, search_file, shape, update_search_file, items or range(0, num_tokens), k, self_search)
                b_matrix = self.get_view(self.filename)[indexer_leaves]
                b_norm = norm_matrix(b_matrix)
                len_bs2 = len(bs2)
                for rng in range(0, len_bs2, rng_step):
                    max_rng = min(rng + rng_step, len_bs2)
                    if vs is None:
                        a_matrix = self.get_view(self.filename)[bs2[rng:max_rng]]
                        a_norm = norm_matrix(a_matrix)
                        a_norm = a_norm.reshape(a_norm.shape[0], 1)                        
                    else:
                        a_matrix = vs[bs2[rng:max_rng]]
                        a_norm = vs_norm[bs2[rng:max_rng]]
                        a_norm = a_norm.reshape(a_norm.shape[0], 1)                        
                    ns = cosine_search(a_matrix, b_matrix, a_norm, b_norm, k=k)#*2
                    for bi, vals in enumerate(ns):
                        b = bs2[bi+rng]
                        if vs is not None and items:
                            b = items[b]
                        if match[b] is None:
                            if self_search and (not search_space or b in search_space):
                                match[b] = [(b, 1.0)]
                            else:
                                match[b] = []
                        match[b].extend([(indexer_leaves[int(vals[id])], vals[id+1]) for id in range(0, vals.shape[0], 2)])
                ks[ki] = None
                a_matrix = None
                a_norm = None
                b_matrix=None
                b_norm=None
            ks = None
        sort_and_save_time += sort_and_save(match, search_file, shape, update_search_file, items or range(0, num_tokens), k, self_search)
        if level == len(self.indexer_nodes)-1:
            if orig_all_items: items = None                    
            if search_file is not None:
                return OntoSearchResults(match=None, search_file=search_file, k=k, max_tokens=max_tokens, items=items, kb=kb, token_type=token_type, search_space=search_space)
            else:
                return OntoSearchResults(match=match, search_file=None, k=k, max_tokens=max_tokens, items=items, kb=kb, token_type=token_type, search_space=search_space)


    # utils for working with HuggingFace's transfomer modules. adapted from transfomers.modeling_utils                                 
    def tie_weight(self, model):
        """ Tie or clone module weights for input view to output embeddings depending if whether we are using TorchScript or not
        """
        input_embedding = model.get_input_embeddings()
        if self is not input_embedding:
            raise RuntimeError("can't tie weight for a model that doesn't use this embedding as input")
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None:
            return
        if model.config.torchscript: # TODO: we are cloning and we need to make another view using the clone                                 
            raise RuntimeError("cloning of embeddings not yet implemented")
        else:
            output_embeddings.weight = self.get_view().local_view

        if hasattr(output_embeddings, 'bias') and output_embeddings.bias is not None:
            output_embeddings.bias.data = torch.nn.functional.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
                'constant',
                0
            )
        if hasattr(output_embeddings, 'out_features'):
            output_embeddings.out_features = self.local_view.shape[1] # self.num_embeddings

    def resize_token_embeddings(self, model, new_num_tokens=None):
        with self.get_lock():
            base_model = getattr(model, model.base_model_prefix, model)  # get the base model if needed
            if new_num_tokens is None:
                return self
            if new_num_tokens == self.num_embeddings:
                return self

            orig_num_embeddings = self.num_embeddings
            self.resize(new_num_tokens)

            # Update base model and current model config
            model.config.vocab_size = new_num_tokens
            base_model.vocab_size = new_num_tokens

            if orig_num_embeddings < self.num_embeddings:
                self.reset_parameters(init_fn=model._init_weight, items=list(range(orig_num_embeddings, self.num_embeddings)))

            # Tie current view weights again if needed
            self.tie_weight(model)
            model.set_input_embeddings(self)

        return self

            
    def attach_view_to_model(self, model, use_current_view=False, local_map_or_input=None, flush_type=OntoMemmap.FLUSH_TYPE_OVERWRITE, local_map_len=200000):
         
       assert use_current_view and local_map_or_input is not None, "Can either use_current_view or set a new local_map_or_input, but not both"
       if local_map_or_input is None:
            local_map_or_input = -1
       if not use_current_view and isinstance(local_map_or_input, IntType):
            local_map = local_map_or_input
            if isinstance(local_map, IntType):
                 if local_map == -1:
                      local_map = list(range(self.shape[0]))
                 else:
                      local_map = list(range(local_map))
       # todo account for an actual local_map
       elif not use_current_view and isinstance(local_map_or_input, ListType):
            local_map = local_map_len
            try:
                 int (local_map_or_input[0])
            except:
                 raise RuntimeError("could not tie input and output embeddings for " + str(model))
            cnt = Counter(local_map_or_input)
            local_map = [a[0] for a in cnt.most_common(max(int(len(cnt)*.75), local_map_len))] #  we cut off the 25% of rare words
            # todo - always add the first N words 
            # todo - add the indexer nodes
            # todo - add some random words for negative sampling, cross entropy
       if use_current_view:
            self.clone().set_input_embeddings(model)
       else:
            self.set_view(local_map, flush_type=flush_type, device=model.device).clone().set_input_embeddings(model)
       return model


    def set_input_embeddings(self, model):
       embeddings = model.get_input_embeddings()
       if isinstance(embeddings, OntoEmbedding) and embeddings is not self and not use_current_view:
          embeddings.flush()
       # we will flush the final embeddings on close/del
       if hasattr(model, 'orig_set_input_embeddings'):
            model.orig_set_input_embeddings(self)
       else:
            model.set_input_embeddings(self)
       output_embeddings = model.get_output_embeddings()
       # because huggingface does not have a set_output_embedding, we hack it
       if output_embeddings is not None:
          tied_embeddings = False
          for attr, out in [(attr, getattr(model, attr)) for attr in dir(model)]:
             if out is output_embeddings:
                setattr(model, attr, copy.copy(out))
                model.get_input_embeddings().tie_weights(model)
                tied_embeddings = True
                break
             else:
                for attr2, out2 in [(attr2, getattr(out, attr2)) for attr2 in dir(out)]:                
                   if out2 is output_embeddings:
                      setattr(out, attr2, copy.copy(out2))
                      model.get_input_embeddings().tie_weights(model)
                      tied_embeddings = True
                      break
                   else:
                      for attr3, out3 in [(attr3, getattr(out2, attr3)) for attr3 in dir(out2)]:                
                         if out3 is output_embeddings:
                            setattr(out2, attr3, copy.copy(out3))
                            model.get_input_embeddings().tie_weights(model)
                            tied_embeddings = True
                            break
          if not tied_embeddings:
             raise RuntimeError("could not tie input and output embeddings for " + str(model))
       return model
# we augment the PreTrainedModel class to account for onto embddings 

orig_tie_weights = PreTrainedModel.tie_weights
def tie_weights_with_ontoembedding(self):
     embeddings = self.get_input_embeddings()     
     if isinstance(embeddings, OntoEmbedding):
          return embeddings.tie_weights(model)
     return self.orig_tie_weights()

if orig_tie_weights != tie_weights_with_ontoembedding:
     PreTrainedModel.orig_tie_weights = orig_tie_weights
     PreTrainedModel.tie_weights = tie_weights_with_ontoembedding
 
def _generate_no_beam_search_with_local_map(
     self,
     input_ids,
     cur_len,
     max_length,
     min_length,
     do_sample,
     temperature,
     top_k,
     top_p,
     repetition_penalty,
     no_repeat_ngram_size,
     bad_words_ids,
     bos_token_id,
     pad_token_id,
     eos_token_id,
     decoder_start_token_id,
     batch_size,
     encoder_outputs,
     attention_mask,
     use_cache,
 ):
     """ Generate sequences for each example without beam search (num_beams == 1).
         All returned sequence are generated independantly.
     """
     # length of generated sentences / unfinished sentences
     unfinished_sents = input_ids.new(batch_size).fill_(1)
     sent_lengths = input_ids.new(batch_size).fill_(max_length)

     past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

     while cur_len < max_length:
         model_inputs = self.prepare_inputs_for_generation(
             input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache
         )

         outputs = self(**model_inputs)
         next_token_logits = outputs[0][:, -1, :]

         # if model has past, then set the past variable to speed up decoding
         if self._use_cache(outputs, use_cache):
             past = outputs[1]

         # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
         if repetition_penalty != 1.0:
             self.enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids, repetition_penalty)

         if no_repeat_ngram_size > 0:
             # calculate a list of banned tokens to prevent repetitively generating the same ngrams
             # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
             banned_tokens = calc_banned_ngram_tokens(input_ids, batch_size, no_repeat_ngram_size, cur_len)
             for batch_idx in range(batch_size):
                 next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

         if bad_words_ids is not None:
             # calculate a list of banned tokens according to bad words
             banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

             for batch_idx in range(batch_size):
                 next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

         # set eos token prob to zero if min_length is not reached
         if eos_token_id is not None and cur_len < min_length:
             next_token_logits[:, eos_token_id] = -float("inf")

         if do_sample:
             # Temperature (higher temperature => more likely to sample low probability tokens)
             if temperature != 1.0:
                 next_token_logits = next_token_logits / temperature
             # Top-p/top-k filtering
             next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
             # Sample
             probs = F.softmax(next_token_logits, dim=-1)
             next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
         else:
             # Greedy decoding
             next_token = torch.argmax(next_token_logits, dim=-1)

         # update generations and finished sentences
         if eos_token_id is not None:
             # pad finished sentences if eos_token_id exist
             tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
         else:
             tokens_to_add = next_token

         input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

         if eos_token_id is not None:
             eos_in_sents = tokens_to_add == eos_token_id
             # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
             is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
             sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
             # unfinished_sents is set to zero if eos in sentence
             unfinished_sents.mul_((~eos_in_sents).long())

         # stop when there is a </s> in each sentence, or if we exceed the maximul length
         if unfinished_sents.max() == 0:
             break

         # extend attention_mask for new generated input if only decoder
         if self.config.is_encoder_decoder is False:
             attention_mask = torch.cat(
                 [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
             )

         cur_len = cur_len + 1

     # if there are different sentences lengths in the batch, some batches have to be padded
     if sent_lengths.min().item() != sent_lengths.max().item():
         assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
         # finished sents are filled with pad_token
         decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
     else:
         decoded = input_ids

     for hypo_idx, hypo in enumerate(input_ids):
         decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

     return decoded



PreTrainedModel._generate_no_beam_search  = _generate_no_beam_search_with_local_map
def _generate_beam_search_with_local_map(
     self,
     input_ids,
     cur_len,
     max_length,
     min_length,
     do_sample,
     early_stopping,
     temperature,
     top_k,
     top_p,
     repetition_penalty,
     no_repeat_ngram_size,
     bad_words_ids,
     bos_token_id,
     pad_token_id,
     eos_token_id,
     decoder_start_token_id,
     batch_size,
     num_return_sequences,
     length_penalty,
     num_beams,
     vocab_size,
     encoder_outputs,
     attention_mask,
     use_cache,
 ):
     """ Generate sequences for each example with beam search.
     """
     input_embeddings = self.get_input_embeddings()
     if isinstance(input_embeddings, OntoEmbedding):
        view_data = input_embeddings.get_view()
        local_to_id = view_data.local_to_id
     generated_hyps = [
         BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
         for _ in range(batch_size)
     ]

     # scores for each sentence in the beam
     beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

     # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
     if do_sample is False:
         beam_scores[:, 1:] = -1e9
     beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

     # cache compute states
     past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

     # done sentences
     done = [False for _ in range(batch_size)]

     while cur_len < max_length:
         model_inputs = self.prepare_inputs_for_generation(
             input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache
         )
         outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
         next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

         # if model has past, then set the past variable to speed up decoding
         if self._use_cache(outputs, use_cache):
             past = outputs[1]

         # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
         if repetition_penalty != 1.0:
             self.enforce_repetition_penalty_(
                 next_token_logits, batch_size, num_beams, input_ids, repetition_penalty,
             )

         if temperature != 1.0:
             next_token_logits = next_token_logits / temperature

         scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
         if self.config.is_encoder_decoder and do_sample is False:
             # TODO (PVP) still a bit hacky here - there might be a better solutino
             scores = self.prepare_scores_for_generation(scores, cur_len=cur_len, max_length=max_length)

         # set eos token prob to zero if min_length is not reached
         if eos_token_id is not None and cur_len < min_length:
             scores[:, eos_token_id] = -float("inf")

         if no_repeat_ngram_size > 0:
             # calculate a list of banned tokens to prevent repetitively generating the same ngrams
             num_batch_hypotheses = batch_size * num_beams
             # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
             banned_batch_tokens = calc_banned_ngram_tokens(
                 input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
             )
             for i, banned_tokens in enumerate(banned_batch_tokens):
                 #ontocord change
                 banned_tokens = local_map[banned_tokens]
                 scores[i, banned_tokens] = -float("inf")

         if bad_words_ids is not None:
             # calculate a list of banned tokens according to bad words
             banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

             for i, banned_tokens in enumerate(banned_tokens):
                 #ontocord change
                 banned_tokens = local_map[banned_tokens]
                 scores[i, banned_tokens] = -float("inf")

         assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
             scores.shape, (batch_size * num_beams, vocab_size)
         )

         if do_sample:
             _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
             # Top-p/top-k filtering
             _scores = top_k_top_p_filtering(
                 _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
             )  # (batch_size * num_beams, vocab_size)
             # re-organize to group the beam together to sample from all beam_idxs
             _scores = _scores.contiguous().view(
                 batch_size, num_beams * vocab_size
             )  # (batch_size, num_beams * vocab_size)

             # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
             probs = F.softmax(_scores, dim=-1)
             next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
             # Compute next scores
             next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)

             # sort the sampled vector to make sure that the first num_beams samples are the best
             next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
             next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)
             next_tokens = local_to_ids[next_tokens]

         else:
             next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

             # re-organize to group the beam together (we are keeping top hypothesis accross beams)
             next_scores = next_scores.view(
                 batch_size, num_beams * vocab_size
             )  # (batch_size, num_beams * vocab_size)

             next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
             next_tokens = local_to_ids[next_tokens]

         assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

         # next batch beam content
         next_batch_beam = []

         # for each sentence
         for batch_idx in range(batch_size):

             # if we are done with this sentence
             if done[batch_idx]:
                 assert (
                     len(generated_hyps[batch_idx]) >= num_beams
                 ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                 assert (
                     eos_token_id is not None and pad_token_id is not None
                 ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                 next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                 continue

             # next sentence beam content
             next_sent_beam = []

             # next tokens for this sentence
             for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                 zip(next_tokens[batch_idx], next_scores[batch_idx])
             ):
                 # get beam and token IDs
                 beam_id = beam_token_id // vocab_size
                 token_id = beam_token_id % vocab_size

                 effective_beam_id = batch_idx * num_beams + beam_id
                 # add to generated hypotheses if end of sentence or last iteration
                 if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                     # if beam_token does not belong to top num_beams tokens, it should not be added
                     is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                     if is_beam_token_worse_than_top_num_beams:
                         continue
                     generated_hyps[batch_idx].add(
                         input_ids[effective_beam_id].clone(), beam_token_score.item(),
                     )
                 else:
                     # add next predicted token if it is not eos_token
                     next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                 # the beam for next step is full
                 if len(next_sent_beam) == num_beams:
                     break

             # Check if were done so that we can save a pad step if all(done)
             done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                 next_scores[batch_idx].max().item(), cur_len=cur_len
             )

             # update next beam content
             assert len(next_sent_beam) == num_beams, "Beam should always be full"
             next_batch_beam.extend(next_sent_beam)
             assert len(next_batch_beam) == num_beams * (batch_idx + 1)

         # stop when we are done with each sentence
         if all(done):
             break

         # sanity check / prepare next batch
         assert len(next_batch_beam) == batch_size * num_beams
         beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
         beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
         beam_idx = input_ids.new([x[2] for x in next_batch_beam])

         # re-order batch
         input_ids = input_ids[beam_idx, :]
         input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
         # re-order internal states
         if past is not None:
             past = self._reorder_cache(past, beam_idx)

         # extend attention_mask for new generated input if only decoder
         if self.config.is_encoder_decoder is False:
             attention_mask = torch.cat(
                 [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
             )

         # update current length
         cur_len = cur_len + 1

     # finalize all open beam hypotheses and end to generated hypotheses
     for batch_idx in range(batch_size):
         if done[batch_idx]:
             continue

         # test that beam scores match previously calculated scores if not eos and batch_idx not done
         if eos_token_id is not None and all(
             (token_id % vocab_size).item() is not eos_token_id for token_id in next_tokens[batch_idx]
         ):
             assert torch.all(
                 next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
             ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                 next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
             )

         # need to add best num_beams hypotheses to generated hyps
         for beam_id in range(num_beams):
             effective_beam_id = batch_idx * num_beams + beam_id
             final_score = beam_scores[effective_beam_id].item()
             final_tokens = input_ids[effective_beam_id]
             generated_hyps[batch_idx].add(final_tokens, final_score)

     # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
     output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
     output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

     # select the best hypotheses
     sent_lengths = input_ids.new(output_batch_size)
     best = []

     # retrieve best hypotheses
     for i, hypotheses in enumerate(generated_hyps):
         sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
         for j in range(output_num_return_sequences_per_batch):
             effective_batch_idx = output_num_return_sequences_per_batch * i + j
             best_hyp = sorted_hyps.pop()[1]
             sent_lengths[effective_batch_idx] = len(best_hyp)
             best.append(best_hyp)

     # shorter batches are filled with pad_token
     if sent_lengths.min().item() != sent_lengths.max().item():
         assert pad_token_id is not None, "`Pad_token_id` has to be defined"
         sent_max_len = min(sent_lengths.max().item() + 1, max_length)
         decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

         # fill with hypothesis and eos_token_id if necessary
         for i, hypo in enumerate(best):
             decoded[i, : sent_lengths[i]] = hypo
             if sent_lengths[i] < max_length:
                 decoded[i, sent_lengths[i]] = eos_token_id
     else:
         # none of the hypotheses have an eos_token
         assert (len(hypo) == max_length for hypo in best)
         decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

     return decoded

PreTrainedModel._generate_beam_search  = _generate_beam_search_with_local_map


# TODO - modify to use Sqlalchemy or another ORM system

class OntoTokenMetada:
   def __init__(self, token_table_name):
      self.TABLE_NAME = token_table_name

      # the location of the fields
      self.TOKEN = 0
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

      self.TOKEN_LABEL = self.TABLE_NAME+".token"
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
      self.FIELD_TO_LABEL = [self.TOKEN_LABEL, self.ID_LABEL, self.CNT_LABEL, self.ALPHA_LABEL, self.TYPE_LABEL, self.LEVEL_LABEL, self.COMPOUND_WORD_COUNT_LABEL, self.PARENTS_LABEL, self.CHILDREN_LABEL, self.SYNONYMS_LABEL, self.DEF_LABEL, self.SIM_LABEL, self.SUBTYPE_LABEL, self.RECNO_LABEL, self.REL_CUTOFF_LABEL]

      self.TOKEN_SIZE = 100
      self.LEVEL_MAX = 9
      self.PARENTS_SIZE = self.LEVEL_MAX-1
      self.CHILDREN_SIZE = 10
      self.SYNONYMS_SIZE = 5
      self.DEF_SIZE = 5
      self.SIM_SIZE = 5
      
      self.ID_ARRAY_FIELDS = (self.PARENTS, self.CHILDREN, self.SYNONYMS, self.DEF, self.SIM)
      self.ID_ARRAY_FIELDS_SIZE = [0, 0, 0, 0, 0, 0, 0, self.PARENTS_SIZE, self.CHILDREN_SIZE,
                                   self.SYNONYMS_SIZE, self.DEF_SIZE, self.SIM_SIZE, 0, 0, 0]

      # 0 is reserved for deleted tokens. has to be between 0-9
      self.TYPE_DELETED = 0
      self.TYPE_REC = 1
      self.TYPE_REL = 2
      self.TYPE_REL3 = 3
      self.TYPE_REL4 = 4
      self.TYPE_REL5 = 5

      # TODO - modify table if the metada is different
      self.CREATE_TABLE_STMT =  "CREATE TABLE IF NOT EXISTS "+self.TABLE_NAME+" ( "
      self.CREATE_TABLE_STMT += self.TOKEN_LABEL+" TEXT," # VARCAHR ("+str(TOKEN_SIZE)+"), "
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
      self.UPDATE_STMT += self.TOKEN_LABEL+"  = ?, "
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
        

class OntoKB:

    # basic words and stopwords
    en_common_verbs = {
        "consider":1,
        "did":1,
        "says":1,
        "let":1,
        "allows":1,
        "'s":1,
        "would":1,
        "taken":1,
        "tell":1,
        "must":1,
        "can":1,
        "follow":1,
        "get":1,
        "may":1,
        "allow":1,
        "help":1,
        "looks":1,
        "thank":1,
        "might":1,
        "gets":1,
        "went":1,
        "mean":1,
        "got":1,
        "given":1,
        "could":1,
        "think":1,
        "done":1,
        "took":1,
        "regards":1,
        "gotten":1,
        "were":1,
        "be":1,
        "say":1,
        "have":1,
        "take":1,
        "shall":1,
        "saying":1,
        "should":1,
        "do":1,
        "cannot":1,
        "said":1,
        "please":1,
        "wo":1,
        "won":1,
        "became":1,
        "been":1,
        "look":1,
        "will":1,
        "is":1,
        "can":1,
        "had":1,
        "has":1,
        "ought":1,
        "follows":1,
        "'ve":1,
        "become":1,
        "happens":1,
        "does":1,
        "tends":1,
        "be":1,
        "getting":1,
        "was":1,
        "were":1,
        "is":1,
        "are":1,
        "becoming":1,
        "'re":1,
        "am":1,
        "'ll":1,
        "'d":1,
        "'m":1,
        "having":1,
        "ask":1,
        "be":1,
        "become":1,
        "begin":1,
        "call":1,
        "can":1,
        "come":1,
        "could":1,
        "do":1,
        "feel":1,
        "find":1,
        "get":1,
        "give":1,
        "go":1,
        "have":1,
        "hear":1,
        "help":1,
        "keep":1,
        "know":1,
        "leave":1,
        "let":1,
        "like":1,
        "live":1,
        "look":1,
        "make":1,
        "may":1,
        "mean":1,
        "might":1,
        "move":1,
        "need":1,
        "play":1,
        "put":1,
        "run":1,
        "say":1,
        "see":1,
        "seem":1,
        "should":1,
        "show":1,
        "start":1,
        "take":1,
        "talk":1,
        "tell":1,
        "think":1,
        "try":1,
        "turn":1,
        "use":1,
        "want":1,
        "will":1,
        "work":1,
        "would":1,
    }

    en_common_web_verbs = {
        "continue":1,
        "back":1,
        "compare":1,
        "subscribe":1,
        "sign":1,
        "share":1,
        "follow":1,
        "like":1,
        "mail":1,
        "email":1,
        "send":1,
        "save":1,
        "print":1,
        "apply":1,
        "make": 1,
        "reserve":1,
        "find":1,
        "view":1,
        "search":1,
        "contact":1,
        "learn":1,
        "view":1,
    }


    en_stop_words = {
      "'s":1,
      "'mon":1,
      "'ll":1,
      "'d":1,
      "'t":1,
      "'ve":1,
      "'re":1,
      "'m":1,
      "a":1,
      "a's":1,
      "able":1,
      "about":1,
      "above":1,
      "according":1,
      "accordingly":1,
      "across":1,
      "actually":1,
      "after":1,
      "afterwards":1,
      "again":1,
      "against":1,
      "ain't":1,
      "ain":1,
      "all":1,
      "allow":1,
      "allows":1,
      "almost":1,
      "alone":1,
      "along":1,
      "already":1,
      "also":1,
      "although":1,
      "always":1,
      "am":1,
      "among":1,
      "amongst":1,
      "an":1,
      "and":1,
      "another":1,
      "any":1,
      "anybody":1,
      "anyhow":1,
      "anyone":1,
      "anyway":1,
      "anyways":1,
      "anywhere":1,
      "apart":1,
      "appropriate":1,
      "are":1,
      "aren't":1,
      "aren":1,
      "around":1,
      "as":1,
      "aside":1,
      "at":1,
      "available":1,
      "away":1,
      "awfully":1,
      "be":1,
      "became":1,
      "because":1,
      "become":1,
      "becomes":1,
      "been":1,
      "before":1,
      "beforehand":1,
      "behind":1,
      "being":1,
      "below":1,
      "beside":1,
      "besides":1,
      "best":1,
      "better":1,
      "between":1,
      "beyond":1,
      "both":1,
      "brief":1,
      "but":1,
      "by":1,
      "c'mon":1,
      "c's":1,
      "can":1,
      "can't":1,
      "cannot":1,
      "cant":1,
      "certain":1,
      "certainly":1,
      "clearly":1,
      "co":1,
      "com":1,
      "consequently":1,
      "consider":1,
      "corresponding":1,
      "could":1,
      "couldn't":1,
      "couldn":1,
      "currently":1,
      "definitely":1,
      "despite":1,
      "did":1,
      "didn't":1,
      "didn":1,
      "different":1,
      "do":1,
      "does":1,
      "doesn't":1,
      "doesn":1,
      "doing":1,
      "don't":1,
      "don":1,
      "done":1,
      "down":1,
      "downwards":1,
      "during":1,
      "each":1,
      "edu":1,
      "eg":1,
      "eight":1,
      "either":1,
      "else":1,
      "elsewhere":1,
      "enough":1,
      "entirely":1,
      "especially":1,
      "et":1,
      "etc":1,
      "even":1,
      "ever":1,
      "every":1,
      "everybody":1,
      "everyone":1,
      "everything":1,
      "everywhere":1,
      "ex":1,
      "exactly":1,
      "example":1,
      "except":1,
      "far":1,
      "few":1,
      "fifth":1,
      "first":1,
      "five":1,
      "followed":1,
      "following":1,
      "follows":1,
      "for":1,
      "former":1,
      "formerly":1,
      "forth":1,
      "four":1,
      "from":1,
      "further":1,
      "furthermore":1,
      "get":1,
      "gets":1,
      "given":1,
      "got":1,
      "gotten":1,
      "had":1,
      "hadn't":1,
      "hadn":1,
      "happens":1,
      "hardly":1,
      "has":1,
      "hasn't":1,
      "hasn":1,
      "have":1,
      "haven't":1,
      "haven":1,
      "having":1,
      "he":1,
      "he's":1,
      "hello":1,
      "help":1,
      "hence":1,
      "her":1,
      "here":1,
      "here's":1,
      "hereafter":1,
      "hereby":1,
      "herein":1,
      "hereupon":1,
      "hers":1,
      "herself":1,
      "hi":1,
      "him":1,
      "himself":1,
      "his":1,
      "hither":1,
      "hopefully":1,
      "how":1,
      "howbeit":1,
      "however":1,
      "i":1,
      "i'd":1,
      "i'll":1,
      "i'm":1,
      "i've":1,
      "ie":1,
      "if":1,
      "immediate":1,
      "in":1,
      "inasmuch":1,
      "inc":1,
      "indeed":1,
      "inner":1,
      "insofar":1,
      "instead":1,
      "into":1,
      "inward":1,
      "is":1,
      "isn't":1,
      "isn":1,
      "it":1,
      "it'd":1,
      "it'll":1,
      "it's":1,
      "its":1,
      "itself":1,
      "just":1,
      "last":1,
      "lately":1,
      "later":1,
      "latter":1,
      "latterly":1,
      "least":1,
      "less":1,
      "lest":1,
      "let":1,
      "let's":1,
      "likely":1,
      "little":1,
      "ltd":1,
      "mainly":1,
      "many":1,
      "may":1,
      "maybe":1,
      "me":1,
      "mean":1,
      "meanwhile":1,
      "merely":1,
      "might":1,
      "more":1,
      "moreover":1,
      "most":1,
      "mostly":1,
      "much":1,
      "must":1,
      "my":1,
      "myself":1,
      "namely":1,
      "nd":1,
      "near":1,
      "nearly":1,
      "necessary":1,
      "neither":1,
      "never":1,
      "nevertheless":1,
      "new":1,
      "next":1,
      "nine":1,
      "no":1,
      "nobody":1,
      "non":1,
      "none":1,
      "noone":1,
      "nor":1,
      "normally":1,
      "not":1,
      "nothing":1,
      "now":1,
      "nowhere":1,
      "obviously":1,
      "of":1,
      "off":1,
      "often":1,
      "oh":1,
      "ok":1,
      "okay":1,
      "old":1,
      "on":1,
      "only":1,
      "onto":1,
      "or":1,
      "other":1,
      "others":1,
      "otherwise":1,
      "ought":1,
      "our":1,
      "ours":1,
      "ourselves":1,
      "out":1,
      "outside":1,
      "over":1,
      "overall":1,
      "own":1,
      "particular":1,
      "particularly":1,
      "per":1,
      "perhaps":1,
      "please":1,
      "plus":1,
      "possible":1,
      "presumably":1,
      "probably":1,
      "que":1,
      "quite":1,
      "qv":1,
      "rather":1,
      "rd":1,
      "re":1,
      "really":1,
      "reasonably":1,
      "regarding":1,
      "regardless":1,
      "regards":1,
      "relatively":1,
      "respectively":1,
      "said":1,
      "same":1,
      "say":1,
      "says":1,
      "secondly":1,
      "self":1,
      "selves":1,
      "sensible":1,
      "serious":1,
      "seriously":1,
      "several":1,
      "shall":1,
      "she":1,
      "should":1,
      "shouldn't":1,
      "shouldn'":1,
      "since":1,
      "six":1,
      "so":1,
      "some":1,
      "somebody":1,
      "somehow":1,
      "someone":1,
      "something":1,
      "sometime":1,
      "sometimes":1,
      "somewhat":1,
      "somewhere":1,
      "soon":1,
      "sorry":1,
      "still":1,
      "sub":1,
      "such":1,
      "sup":1,
      "sure":1,
      "t's":1,
      "take":1,
      "taken":1,
      "tell":1,
      "tends":1,
      "th":1,
      "than":1,
      "thank":1,
      "thanks":1,
      "thanx":1,
      "that":1,
      "that's":1,
      "thats":1,
      "the":1,
      "their":1,
      "theirs":1,
      "them":1,
      "themselves":1,
      "then":1,
      "thence":1,
      "there":1,
      "there's":1,
      "thereafter":1,
      "thereby":1,
      "therefore":1,
      "therein":1,
      "theres":1,
      "thereupon":1,
      "these":1,
      "they":1,
      "they'd":1,
      "they'll":1,
      "they're":1,
      "they've":1,
      "think":1,
      "third":1,
      "this":1,
      "thorough":1,
      "thoroughly":1,
      "those":1,
      "though":1,
      "three":1,
      "through":1,
      "throughout":1,
      "thru":1,
      "thus":1,
      "to":1,
      "together":1,
      "too":1,
      "took":1,
      "toward":1,
      "towards":1,
      "truly":1,
      "two":1,
      "un":1,
      "under":1,
      "unfortunately":1,
      "unless":1,
      "unlikely":1,
      "until":1,
      "unto":1,
      "up":1,
      "upon":1,
      "us":1,
      "usually":1,
      "various":1,
      "very":1,
      "via":1,
      "viz":1,
      "vs":1,
      "was":1,
      "wasn't":1,
      "wasn":1,
      "way":1,
      "we":1,
      "we'd":1,
      "we'll":1,
      "we're":1,
      "we've":1,
      "welcome":1,
      "please":1,
      "well":1,
      "went":1,
      "were":1,
      "weren't":1,
      "weren":1,
      "what":1,
      "what's":1,
      "whatever":1,
      "when":1,
      "whence":1,
      "whenever":1,
      "where":1,
      "where's":1,
      "whereafter":1,
      "whereas":1,
      "whereby":1,
      "wherein":1,
      "whereupon":1,
      "wherever":1,
      "whether":1,
      "which":1,
      "while":1,
      "whither":1,
      "who":1,
      "who's":1,
      "whoever":1,
      "whole":1,
      "whom":1,
      "whose":1,
      "why":1,
      "will":1,
      "willing":1,
      "with":1,
      "within":1,
      "without":1,
      "won't":1,
      "would":1,
      "wouldn't":1,
      "wouldn":1,
      "yes":1,
      "yet":1,
      "you":1,
      "you'd":1,
      "you'll":1,
      "you're":1,
      "you've":1,
      "your":1,
      "yours":1,
      "yourself":1,
      "yourselves":1,
      }

    PARSE_WORD = 0
    PARSE_WORD_ID = 1


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

    # SUBTYPE FOR TOKENS
    TAG_NOUN = 10
    TAG_VERB = 11
    TAG_ADJ = 12
    TAG_OTHER = 13
    TAG_REL = 14
    TAG_DOMAIN = 15
    TAG_PARSE = 16

    TAG_TO_STR = ["U", "S", "W", "Y", "G", "I", "T", "X", "F", "Q", "N", "V", "A", "O", "R", "D", "P"]
    TAG_TO_LONG_STR = ["UNKOWN", "REC", "ORPHAN" "SYN", "ORTH", "INTER", "TRAN", "RULE", "FRAME", "SCRIPT", "NOUN", "VERB", "ADJ", "OTHER", "RELATION",  "DOMAIN", "PARSE"]
    STR_TO_TAG = [None]*256

    STR_TO_TAG[ord("U")] = REL_UNKNOWN
    STR_TO_TAG[ord("S")] = REL_REC
    STR_TO_TAG[ord("W")] = REL_ORPHAN
    STR_TO_TAG[ord("Y")] = REL_SYN
    STR_TO_TAG[ord("G")] = REL_ORTH
    STR_TO_TAG[ord("I")] = REL_INTER
    STR_TO_TAG[ord("T")] = REL_TRAN
    STR_TO_TAG[ord("X")] = REL_RULE
    STR_TO_TAG[ord("F")] = REL_FRAME
    STR_TO_TAG[ord("Q")] = REL_SCRIPT

    STR_TO_TAG[ord("N")] = TAG_NOUN # anything that can act or be acted
                                      # upon
    STR_TO_TAG[ord("V")] = TAG_VERB # any action or connector between nouns. 
    STR_TO_TAG[ord("A")] = TAG_ADJ # anything that can be an
                                         # adjetive, adverb or descriptor of something else.
    STR_TO_TAG[ord("O")] = TAG_OTHER # all other words, including
                                     # prepositions and determiners
    STR_TO_TAG[ord("R")] = TAG_REL # relationships. forms of a-b are
                                   # automatically relationships. 
    STR_TO_TAG[ord("D")] = TAG_DOMAIN # all parents where the children are
                                     # mixed types
    STR_TO_TAG[ord("P")] = TAG_PARSE # a pattern used to parse a sequence of tokens

    KB_ID = 0
    KB_LEVEL = 1
    KB_REL_START = 2
    LEN_KB = 200

    # hints for potential tags based on suffixes
    en_guess_tags_from_suffix = \
        [('s', [[3, [('ing', TAG_VERB)]], [2, [('ed', TAG_VERB)]]]),\
             ('es', [[3, [('ing', TAG_VERB)]], [2, [('ed', TAG_VERB)]]]),\
             ('ed', [[3, [('ing', TAG_VERB)]], [2, [('es', TAG_VERB)]],  [1, [('e', TAG_VERB)]]]),\
             ('ing', [[2, [('es', TAG_VERB), ('ed', TAG_VERB)]], [1, [('e', TAG_VERB)]]])]

    en_guess_tags_from_suffix_fsm = trie_init(suffixList=en_guess_tags_from_suffix)

    # TODO - load the underlying ontoembedding
    def __init__(self, embed_or_filename, **kwargs):
        tokenizer = kwargs.get('tokenizer')
        model_name = kwargs.get('model_name')
        stop_words = kwargs.get('stop_words')
        token_table_name = kwargs.get('token_table_name', "Token")
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
           self.embed = OntoEmbedding(self.path+"/embedding.mmap", shape=(-1, embedding_dim))
        self.embed.model_name = model_name
        self.tlock = self.embed_weight_data().tlock or DummyLock()
        if os.path.exists(self.path):
           logger.info(self.path)
           tokenizer = AutoTokenizer.from_pretrained(self.path)
        if tokenizer is None:
           tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer
        if hasattr(tokenizer, 'bpe_ranks'):
           self.bpe_ranks = tokenizer.bpe_ranks
        else:
           self.bpe_ranks = {}
        self.encoder = tokenizer.added_tokens_encoder
        self.decoder = tokenizer.added_tokens_decoder
        if hasattr(tokenizer, 'encoder'):
            self.base_encoder = tokenizer.encoder
        elif hasattr(tokenizer, 'vocab'):
            self.base_encoder = tokenizer.vocab
        else:
            self.base_encoder = tokenizer.added_tokens_encoder
        if hasattr(tokenizer, 'decoder'):
            self.base_decoder = tokenizer.decoder
        elif hasattr(tokenizer, 'ids_to_token'):
            self.base_decoder = tokenizer.ids_to_token
        else:
            self.base_decoder = tokenizer.added_tokens_decoder
        self.unk_token_id = self.tokenizer.unk_token_id
        self.unk_token = self.tokenizer.unk_token
        self.pad_token = self.tokenizer.pad_token
        self.view_cnt = 0
        if stop_words is None:
            stop_words = self.en_stop_words
        self.stop_words = stop_words
        self.token_meta = OntoTokenMetada(token_table_name)
        conn = self.conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.text_factory = str
        conn.execute('PRAGMA journal_mode = WAL')         
        conn.commit()
        self.sql_execute(self.token_meta.CREATE_TABLE_STMT, commit=True)
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

    def create_indexer(self, items=None, rng_step = 10000,  level=0, do_hiearchical=True, num_threads=None, num_parents=None, cluster_size=None, lock=True, kb=None):
       with self.get_lock():
          self.embed.create_indexer(items=items, rng_step = rng_step,  level=level, do_hiearchical=do_hiearchical, num_threads=num_threads, num_parents=num_parents, cluster_size=cluster_size, lock=lock, kb=self)


    def search(self, items=None, vs=None, vs_norm=None, k=100, k_clusters=10, rng_step = 10000, search_file=None, indexer_nodes=None, level=None, match=None, update_search_file=False, cnt=None, num_threads=None, use_search_file=False, clear_search_file=False, init_search_file=False,  need_update=None, max_tokens=None, search_space=None, token_type=None, vs_fn=None, vs_callback_data=None, search_indexer_nodes_only=False):
       with self.get_lock():
          if self.embed.indexer_nodes == [] or self.embed.indexer_nodes is None:
             self.create_indexer()
       return self.embed.search(items=items, vs=vs, vs_norm=vs_norm, k=k, k_clusters=k_clusters, rng_step = rng_step, search_file=search_file, indexer_nodes=indexer_nodes, level=level, match=match, update_search_file=update_search_file, cnt=cnt, num_threads=num_threads, use_search_file=use_search_file, clear_search_file=clear_search_file, init_search_file=init_search_file,  need_update=need_update, max_tokens=max_tokens, search_space=search_space, token_type=token_type, vs_fn=vs_fn, vs_callback_data=vs_callback_data, kb=self, search_indexer_nodes_only=search_indexer_nodes_only)

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

    def has_token(self, token):
        if type(token) is BytesType:
            token = token.decode('utf-8')
        ret = token in self.encoder or token in self.base_encoder
        if not ret:
            ret = self.tokenizer.convert_tokens_to_ids(token)
            if ret == self.unk_token_id and token != self.unk_token:
                ret = None
        return ret

    def token_to_id(self, token, id=None): 
        if type(token) is BytesType:
            token = token.decode('utf-8')
        ret = self.encoder.get(token)
        if ret is None:
            ret = self.base_encoder.get(token)
            if ret is None:
                ret = self.tokenizer.convert_tokens_to_ids(token)
                if ret == self.unk_token_id and token != self.unk_token:
                    ret = None
                if ret is None:
                    ret = id
        return ret

    def id_to_token(self, id, token=None):
        ret = self.decoder.get(id)
        if ret is None:
            ret = self.base_decoder.get(id)
            if ret is None:
                ret = self.tokenizer.convert_ids_to_tokens(id)
                if ret == self.unk_token and id != self.unk_token_id:
                        ret = None
                if ret is None:
                    ret = token
        return ret
    
    def token_exists(self, id_or_rec):
        if self.is_token(id_or_rec):
            return id_or_rec[self.token_meta.CNT] > 0
        else:
            if id_or_rec < 0:
                id_or_rec += self.len_()
            return self.id_to_token(id_or_rec) is not None

    def embed_weight_data(self):
        return self.embed.get_view()


    def new_token(self, copy_rec=None):
        if copy_rec is not None:
            return copy.deepcopy(copy_rec)
        ret = [None]*self.token_meta.LEN
        ret[self.token_meta.TOKEN] = ''
        ret[self.token_meta.ID] = -1
        ret[self.token_meta.CNT] = -1
        for field in self.token_meta.ID_ARRAY_FIELDS:
            ret[field] = []
        return ret
                
    def is_token(self, a_rec):
        return (type(a_rec) in (ListType, TupleType) and len(a_rec) == self.token_meta.LEN) and type(a_rec[self.token_meta.TOKEN]) in (BytesType, StringType)

    def v(self, token):
        return self.embed_weight_data()[self.token_to_id(token)]

    def max_id(self):
       ret = self.sql_execute("SELECT MAX("+self.token_meta.ID_LABEL+") FROM "+self.token_meta.TABLE_NAME, fetch=True)
       if ret[0][0] is None:
          return -1
       return ret[0][0]

    def kb_item_exists(self, id):
       ret = self.sql_execute("SELECT EXISTS(SELECT 1 FROM "+self.token_meta.TABLE_NAME+" WHERE "+self.token_meta.ID_LABEL+"=? LIMIT 1)", [id], fethch=True)
       return ret[0][0]
                
    def get_token(self, id_or_token=None, field=None, field2=None, field3=None, token_type=None):
        """
        returns a field, tuple of fields or a token. id_or_token can be an int or a string token. token can be of the form, dog#animal
        """
        id = None
        a_rec = None
        if type(id_or_token) in (np.int32, np.int64, np.uint32, np.uint64, IntType):
            id = int(id_or_token)
            if field3 == None and field in (self.token_meta.TOKEN, self.token_meta.ID) and field2 in (self.token_meta.TOKEN, self.token_meta.ID, None):
                if field2 is not None:
                    if field == self.token_meta.TOKEN:
                        return (self.id_to_token(id), id)
                    else:
                        return (id, self.id_to_token(id))
                elif field == self.token_meta.TOKEN:
                    return self.id_to_token(id)
                else:
                    return id

        if type(id_or_token) is StringType:
            token = id_or_token
            id = self.token_to_id(token)
            if id is None:
                id = self.token_to_id(token.translate(trannum))

        #if not self.token_exists(id):
        #    return None

        if field3 is not None:
            stmt = "SELECT "+self.token_meta.FIELD_TO_LABEL[field] + "," + \
                self.token_meta.FIELD_TO_LABEL[field2] + "," + \
                self.token_meta.FIELD_TO_LABEL[field3] + \
                " FROM "+self.token_meta.TABLE_NAME+" WHERE "+self.token_meta.ID_LABEL+" = ? "
            val = [id]
        elif field2 is not None:
            stmt = "SELECT "+self.token_meta.FIELD_TO_LABEL[field] + "," + \
                self.token_meta.FIELD_TO_LABEL[field2] + \
                " FROM "+self.token_meta.TABLE_NAME+" WHERE "+self.token_meta.ID_LABEL+" = ? "
            val = [id]
        elif field is not None:
            stmt = "SELECT "+self.token_meta.FIELD_TO_LABEL[field] + \
                " FROM "+self.token_meta.TABLE_NAME+" WHERE "+self.token_meta.ID_LABEL+" = ? "
            val = [id]
        else:
            stmt = "SELECT * FROM "+self.token_meta.TABLE_NAME+" WHERE "+self.token_meta.ID_LABEL+" = ?"
            val = [id]

        if token_type is not None and type(token_type) is IntType:
            stmt += " AND "+self.token_meta.TYPE_LABEL+" = ? "
            val.append(token_type)
        elif token_type is not None and type(token_type) is ListType:
            stmt += " AND "+self.token_meta.TYPE_LABEL+" in (" + ("?,"*len(token_type)-1)+"?) "
            val.extend(token_type)

        if field is not None:
            dat =  self.sql_execute(stmt, val, fetch=True)[0]
            if dat is None:
                return None
            dat = list(dat)
            if field == self.token_meta.TOKEN:
                if type(dat[0]) is BytesType:
                    dat[0] = dat[0].decode('utf-8')
                dat[0] = str(dat[0])
            if field2 == self.token_meta.TOKEN:
                if type(dat[1]) is BytesType:
                    dat[1] = dat[1].decode('utf-8')
                dat[1] = str(dat[1])
            if field3 == self.token_meta.TOKEN:
                if type(dat[2]) is BytesType:
                    dat[2] = dat[2].decode('utf-8')
                dat[2] = str(dat[2])
            if field in self.token_meta.ID_ARRAY_FIELDS:
                if dat[0]: 
                    dat[0] = pickle.loads(bytes(dat[0]))
                else:
                    dat[0] = None
            if field2 in self.token_meta.ID_ARRAY_FIELDS:
                if dat[1]: 
                    dat[1] = pickle.loads(bytes(dat[1]))
                else:
                    dat[1] = None
            if field3 in self.token_meta.ID_ARRAY_FIELDS:
                if dat[2]: 
                    dat[2] = pickle.loads(bytes(dat[2]))
                else:
                    dat[2] = None
            return tuple(dat)
        a_rec = self.sql_execute(stmt, val, fetch=True)[0]
        a_rec = list(a_rec)
        if a_rec:
            if type(a_rec[self.token_meta.TOKEN]) is BytesType: 
                a_rec[self.token_meta.TOKEN] = a_rec[self.token_meta.TOKEN].decode('utf-8')
            a_rec[self.token_meta.TOKEN] = str(a_rec[self.token_meta.TOKEN])
            for f in self.token_meta.ID_ARRAY_FIELDS:
                if a_rec[f]: 
                    a_rec[f] = pickle.loads(bytes(a_rec[f]))
                else:
                    a_rec[f] = None                        
            return a_rec
        return None
            
    def get_token_iter(self, ids=None, field=None, field2=None, field3=None, token_type=None, reverse=False):
        """
        returns a field, tuple of fields or a rec. id_or_token can be an int or a string token. token can be of the form, dog#animal
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
        if field3 == None and field in (self.token_meta.TOKEN, self.token_meta.ID) and field2 in (self.token_meta.TOKEN, self.token_meta.ID, None):
            for id in ids:
                if field2 is not None:
                    if field == self.token_meta.TOKEN:
                        yield (self.id_to_token(id), id)
                    else:
                        yield (id, self.id_to_token(id))
                elif field == self.token_meta.ID:
                    yield self.id_to_token(id)
                else:
                    yield id
            return

        for rng in range(0, len_items, rng_step):
            max_rng = min(rng + rng_step, len_items)
            if field3 is not None:
                stmt = "SELECT "+self.token_meta.FIELD_TO_LABEL[field] + "," + \
                    self.token_meta.FIELD_TO_LABEL[field2] + "," + \
                    self.token_meta.FIELD_TO_LABEL[field3] + \
                    " FROM "+self.token_meta.TABLE_NAME+" WHERE "+self.token_meta.ID_LABEL+" in ("+("?,"*(max_rng-rng-1))+"?)"
                val = ids[rng:max_rng]
            elif field2 is not None:
                stmt = "SELECT "+self.token_meta.FIELD_TO_LABEL[field] + "," + \
                    self.token_meta.FIELD_TO_LABEL[field2] + \
                    " FROM "+self.token_meta.TABLE_NAME+" WHERE "+self.token_meta.ID_LABEL+" in ("+("?,"*(max_rng-rng-1))+"?)"
                val = ids[rng:max_rng]
            elif field is not None:
                stmt = "SELECT "+self.token_meta.FIELD_TO_LABEL[field] + \
                    " FROM "+self.token_meta.TABLE_NAME+" WHERE "+self.token_meta.ID_LABEL+" in ("+("?,"*(max_rng-rng-1))+"?)"
                val = ids[rng:max_rng]
            else:
                stmt = "SELECT * FROM "+self.token_meta.TABLE_NAME+" WHERE "+self.token_meta.ID_LABEL+" in ("+("?,"*(max_rng-rng-1))+"?)"
                val = ids[rng:max_rng]

            if token_type is not None and type(token_type) is IntType:
                stmt += " AND "+self.token_meta.TYPE_LABEL+" = ? "
                val.append(token_type)
            elif token_type is not None and type(token_type) is ListType:
                stmt += " AND "+self.token_meta.TYPE_LABEL+" in (" + ("?,"*len(token_type)-1)+"?) "
                val.extend(token_type)
            if field is not None:
                dat =  self.sql_execute(stmt, val, fetch=True)
                if reverse:
                   dat = reversed(dat)
                if not dat:
                    return
                for dat2 in dat:
                    dat2 = list(dat2)
                    if field == self.token_meta.TOKEN:
                        if type(dat2[0]) is BytesType:
                            dat2[0] = dat2[0].decode('utf-8')
                        dat2[0] = str(dat2[0])
                    if field2 == self.token_meta.TOKEN:
                        if type(dat2[1]) is BytesType:
                            dat2[1] = dat2[1].decode('utf-8')
                        dat2[1] = str(dat2[1])
                    if field3 == self.token_meta.TOKEN:
                        if type(dat2[2]) is BytesType:
                            dat2[2] = dat2[2].decode('utf-8')
                        dat2[2] = str(dat2[2])
                    if field in self.token_meta.ID_ARRAY_FIELDS:
                        if dat2[0]: 
                            dat2[0] = pickle.loads(bytes(dat2[0]))
                        else:
                            dat2[0] = None
                    if field2 in self.token_meta.ID_ARRAY_FIELDS:
                        if dat2[1]: 
                            dat2[1] = pickle.loads(bytes(dat2[1]))
                        else:
                            dat2[1] = None
                    if field3 in self.token_meta.ID_ARRAY_FIELDS:
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
                        if type(a_rec[self.token_meta.TOKEN]) is BytesType: 
                            a_rec[self.token_meta.TOKEN] = a_rec[self.token_meta.TOKEN].decode('utf-8')
                        a_rec[self.token_meta.TOKEN] = str(a_rec[self.token_meta.TOKEN])
                        for f in self.token_meta.ID_ARRAY_FIELDS:
                            if a_rec[f]: 
                                a_rec[f] = pickle.loads(bytes(a_rec[f]))
                            else:
                                a_rec[f] = None                                   
                        yield a_rec


    def set_token(self, id_or_tokens, a_rec=None, embeds=None, check_kb_item_exists=False, force_kb_insert=False):
        """ 
        Update the kb and array with the tokens.  id_or_tokens is an
        id or tokens list or a single rec. tokens can be an array of full rec or
        tuples. If a tuple, then the information will be updated
        against what is on disk. if of rec type, the information
        will be replace what is on disk.

        *Changing a token for a specific id will not give warnings.*

        Resetting a token to another id will remap the id in the
        whole kb. 

        If a token is already in the kb, the new information will
        replace the old information.
        """
        id = None
        if type(id_or_tokens) in (np.int32, np.int64, np.uint32, np.uint64, IntType):
            id = int(id_or_tokens)
            if not self.is_token(a_rec) and a_rec in ((), None):
                raise RuntimeError("trying to set an item to empty values. use del_token instead")
            else:
                if self.is_token(a_rec):
                    a_rec[self.token_meta.ID] = id
                else:
                    a_rec = list(a_rec)
                    len_a_rec = len(a_rec)
                    if len_a_rec <= self.token_meta.ID:
                        a_rec = a_rec+([0]*(self.token_meta.ID-len_a_rec+1))
                    a_rec[self.token_meta.ID] = id


        ret_single = False
        if type(id_or_tokens) is TupleType or self.is_token(id_or_tokens):
            ret_single=True
            tokens = [id_or_tokens]
        elif a_rec is not None:
            ret_single=True
            tokens = [a_rec]
        else:
           tokens = id_or_tokens
        if not tokens:
            return []

        with self.get_lock():
            tokens = [list(a_rec) for a_rec in tokens]
            for a_rec in tokens:
                len_a_rec = len(a_rec)
                if len_a_rec > self.token_meta.ID:
                    a_rec[self.token_meta.ID] = self.token_to_id(a_rec[self.token_meta.TOKEN], a_rec[self.token_meta.ID])
                else:
                   a_rec = a_rec+[None]*(self.token_meta.ID-len_a_rec+1)
                   a_rec[self.token_meta.ID] = -1
            max_id = [a_rec[self.token_meta.ID] for a_rec in tokens if a_rec[self.token_meta.ID] != -1] + [-1]
            max_id = max(max_id)
            if max_id >= self.len_():
               self.resize(max_id+1) 
            need_id = [a_rec for a_rec in tokens if a_rec[self.token_meta.ID] == -1]
            if need_id:
               id = self.len_()
               self.resize(id+len(need_id))
               max_id = id - 1
               for a_rec in need_id:
                  a_rec[self.token_meta.ID] = id
                  id +=1
            need_id = None
            tokens2 = {}
            if not force_kb_insert:
                tokens2_id = [a_rec[self.token_meta.ID] for a_rec in tokens if not self.is_token(a_rec) and a_rec[self.token_meta.ID] <= max_id]
                if tokens2_id:
                   tokens2 = dict([(a_rec[self.token_meta.ID], a_rec) for a_rec in self.get_token_iter(tokens2_id)])
            for i in range(len(tokens)):
                b_rec = tokens[i]
                if self.is_token(b_rec):
                    a_rec = b_rec
                else:
                    a_rec = None
                    if not force_kb_insert:
                        id = b_rec[self.token_meta.ID]
                        if id in  tokens2:
                            a_rec = tokens2[id]
                            del tokens2[id]
                    if a_rec is None:
                        a_rec = self.new_token()
                    for field, dat in enumerate(b_rec):
                        if field in self.token_meta.ID_ARRAY_FIELDS:
                            self.set_token_array_field(a_rec,field,dat)
                        elif field == self.token_meta.TOKEN:
                            a_rec[field] = str(dat)
                        else:
                            a_rec[field] = dat
                tokens[i] = a_rec
            if tokens2:
                raise RuntimeError("problem with getting tokens in update rec")
            if ret_single:
                self.set_token_by_id(tokens[0][self.token_meta.ID], tokens[0], check_kb_item_exists=check_kb_item_exists, force_kb_insert=force_kb_insert)
            else:
                self.set_token_by_id([a_rec[self.token_meta.ID] for a_rec in tokens], tokens, check_kb_item_exists=check_kb_item_exists, force_kb_insert=force_kb_insert)
            all_ids = dict([(int(a_rec[self.token_meta.ID]),1) for a_rec in tokens])
            remap_hash = {}
            del_ids = []
            for a_rec in tokens:
                id = a_rec[self.token_meta.ID]
                token = a_rec[self.token_meta.TOKEN]
                if self.has_token(token) and self.token_to_id(token) != id:
                    remapped_id = self.token_to_id(token)
                    remap_hash[remapped_id] = id
                    if remapped_id not in all_ids: del_ids.append(remapped_id)
            for  a_rec in tokens:
                token = a_rec[self.token_meta.TOKEN]
                id = int(a_rec[self.token_meta.ID])
                old_token = self.id_to_token(id)
                if old_token is not None and old_token != token:
                    if old_token in self.encoder: del self.encoder[old_token]
                    if old_token in self.base_encoder: self.base_encoder[old_token]
                if (self.has_token(token) and self.token_to_id(token) != id):
                    if self.has_token(token) and token not in self.encoder and token not in self.base_encoder:
                        raise RuntimeError("Trying to change id of a fixed token in tokenizer")
                    if token in self.encoder: self.encoder[token] = id
                    if token in self.base_encoder: self.base_encoder[token] = id
                    if id in self.decoder: self.decoder[id] = token
                    if id in self.base_decoder: self.base_decoder[id] = token
                elif (not self.has_token(token)):
                   self.encoder[token] = id
                   self.decoder[id] = token
            if del_ids:
                self.del_token(del_ids)
            if remap_hash:
                self.remap_tokens(remap_hash)

        if ret_single:
           if embeds is not None:
              self.embed_weight_data()[tokens[0][self.token_meta.ID]] = embeds#[0]
        else:
           if embeds is not None:
              self.embed_weight_data()[[a_rec[self.token_meta.ID] for a_rec in tokens]] = embeds
        self.flush()
        if ret_single:
           return tokens[0]
        return tokens

    def set_token_by_id(self, id, a_token_or_tokens, check_kb_item_exists=False, force_kb_insert=True):
        if type(id) == SliceType:
            start = (id.start or 0)
            stop = (id.stop or self.shape[0])
            id = range(start, stop)
        if type(id) in (ListType, RangeType):
            update_vals = []
            insert_vals = []
            for i, a_rec in zip(id, a_token_or_tokens):
                a_rec = copy.copy(a_rec)
                for field in self.token_meta.ID_ARRAY_FIELDS:
                    if a_rec[field]: 
                        a_rec[field] = sqlite3.Binary(pickle.dumps(a_rec[field], pickle.HIGHEST_PROTOCOL))
                    else:
                        a_rec[field] = None
                if not force_kb_insert and self.token_exists(i) and (not check_kb_item_exists or self.db_item_exists(i)):
                    update_vals.append(a_rec +[i])
                else:
                    insert_vals.append(a_rec)
            with self.get_lock():
                if insert_vals:
                   self.sql_execute(self.token_meta.INSERT_STMT, insert_vals, many=True, commit=True)
                if update_vals:
                   self.sql_execute(self.token_meta.UPDATE_STMT, update_vals, many=True, commit=True)
        else:
            i = id
            a_rec = copy.copy(a_token_or_tokens)
            for field in self.token_meta.ID_ARRAY_FIELDS:
                if a_rec[field]: 
                    a_rec[field] = sqlite3.Binary(pickle.dumps(a_rec[field], pickle.HIGHEST_PROTOCOL))
                else:
                    a_rec[field] = None
            with self.get_lock():
                logger.info((self.token_meta.UPDATE_STMT,))
                logger.info((a_rec,))
                if not force_kb_insert and self.token_exists(i) and (not check_kb_item_exists or self.db_item_exists(i)):
                    self.sql_execute(self.token_meta.UPDATE_STMT, a_rec+[i], many=False, commit=True)
                else:
                    self.sql_execute(self.token_meta.INSERT_STMT, a_rec, many=False, commit=True)

    def del_token_by_id(self, id):
        if type(id) == SliceType:
            start = (id.start or 0)
            stop = (id.stop or self.shape[0])
            id = range(start, stop)
        if type(id) == ListType:
           rng_step=999
           len_items = len(id)
           for rng in range(0, len_items, rng_step):
              max_rng = min(rng + rng_step, len_items)
              self.sql_execute("DELETE FROM "+self.token_meta.TABLE_NAME+" WHERE "+self.token_meta.ID_LABEL+" in ("+("?,"*(max_rng-rng-1))+"?)", id[rng:max_rng], commit=True)
        elif type(id) == RangeType:
             self.sql_execute("DELETE FROM "+self.token_meta.TABLE_NAME+" WHERE "+self.token_meta.ID_LABEL+" >= ? and "+self.token_meta.ID_LABEL+" < ?", (id[0], id[-1]), commit=True)
        else:
             self.sql_execute("DELETE FROM "+self.token_meta.TABLE_NAME+" WHERE  "+self.token_meta.ID_LABEL+" == ? ", id, commit=True)

    def mirror_kb_and_tokenizer(self, force_kb_insert=False):
        logger.info("mirroring kb, embed and tokenizer")
        with self.get_lock():
            #del_ids = []
            max_id = self.len_()
            dat = {}
            for id in range(len(self.tokenizer)):
                token = self.id_to_token(id)
                if not token or token.startswith("[unused"):
                    #del_ids.append(id)
                    continue
                cnt = (max_id - id + 1) * 100
                alpha = 1.
                dat[token] = ((token, id, cnt, alpha)) # ASSUMES THIS ORDERING
            for a_rec in self.get_token_iter():
                token = a_rec[self.token_meta.TOKEN]
                a_rec2 = dat.get(token)
                if a_rec2 and a_rec2[1] == a_rec[self.token_meta.ID]:
                    del dat[token]
            dat = list(dat.values())
            logger.info(("adding", len(dat), dat))
            if dat and len(dat) > 0: 
               for a_rec in self.set_token(dat, force_kb_insert=force_kb_insert):
                  pass # logger.info((self.token_to_id(a_rec[self.token_meta.TOKEN]), self.id_to_token(a_rec[self.token_meta.ID]), a_rec))
            dat = None
            #if del_ids: self.del_token(del_ids)
            #del_ids = None
            for a_rec in self.get_token_iter():
                id = a_rec[self.token_meta.ID]
                token = a_rec[self.token_meta.TOKEN]
                if (self.has_token(token) and self.token_to_id(token) != id) or (not self.has_token(token)):                        
                    if self.has_token(token):
                        if token not in self.encoder and token not in self.base_encoder:
                            raise RuntimeError("Trying to change id of a fixed token in tokenizer")
                        if token in self.encoder: self.encoder[token] = id
                        if token in self.base_encoder: self.base_encoder[token] = id
                        if id in self.decoder: self.decoder[id] = token
                        if id in self.base_decoder: self.base_decoder[id] = token
                    else:
                        self.encoder[token] = id
                        self.decoder[id] = token
            new_tokens = []
            for id in range(self.len_()):
               if self.id_to_token(id) is None and sum(self.embed_weight_data()[id]) != 0.0:
                  new_tokens.append(("[unused"+str(id)+"]", id))
            if new_tokens:
               self.set_token(new_tokens)
        self.save()

    def set_token_array_field(self, a_rec, field, dat):
        if not dat:
            dat = []
        a_len = len(dat)
        size=self.token_meta.ID_ARRAY_FIELDS_SIZE[field]
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
            do_defragment=[id for id in range(self.len_()) if not self.token_exists(id)]
            len_kb = self.len_()
            if not do_defragment:
                return False
            self.del_token(do_defragment)
            newTokens = []
            oldTokensId = []
            cnt = 1
            a_rec = None
            for id2, a_rec in zip(do_defragment, self.get_token_iter(reverse=True)):
                if a_rec[self.token_meta.ID] <= id2:
                    break
                do_defragment = do_defragment[1:]
                a_rec = copy.copy(a_rec)
                oldTokensId.append(a_rec[self.token_meta.ID])
                a_rec[self.token_meta.ID] = id2
                newTokens.append(a_rec)
                cnt +=1
                if cnt % rng_step == 0:
                    # copying in chunks is more efficient than multiple disk access
                    # copy the vector before adding because we are really moving the data
                    self.set_token(newTokens, embeds=self.embed_weight_data()[oldTokensId])
                    newTokens = []
                    oldTokensId = []
                if not do_defragment:
                    break
            resizeLen=a_rec[self.token_meta.ID]+1
            if newTokens:
                # copy the vector before adding because we are really moving the data
                self.set_token(newTokens, embeds=self.embed_weight_data()[oldTokensId])
                v = None
                newTokens = []
                oldTokensId = []
            self.resize(resizeLen, defrag=True)
            self.save()
        return True

    def remap_tokens(self, remap_hash, do_parent_child=True, ids=None):
        """ convience method for remapping {old_id:new_id} """
        with self.get_lock():
            if ids is None:
                id2 = range(self.len_())
            else:
                id2 = ids
            tokens = []
            for a_rec in self.get_token_iter(ids=ids):
                changed = False
                for field in self.token_meta.ID_ARRAY_FIELDS:
                    a_change = False
                    if a_rec[field]:
                        new_tokens = []
                        for key in a_rec[field]:
                            new_id = remap_hash.get(key, key)
                            if new_id != key:
                                a_change = True
                            if new_id != -1:
                                new_tokens.append(new_id)
                        if a_change:
                            self.set_token_array_field(a_rec,field,new_tokens)
                            changed = True
                if changed:
                   rec.append(a_rec)
            if tokens:
               self.set_token(tokens)


    def del_token(self, ids):
        """ delete tokens for all the ids """
        if not ids:
            return
        with self.get_lock():
            if type(ids) in (np.int32, np.int64, IntType):
                ids = [int(ids)]
            elif self.is_token(ids):
                ids = [int(ids[self.token_meta.ID])]
            for id in ids:
                token = self.id_to_token(id)
                if not token or token.startswith("[unused"):
                    continue
                if token in self.encoder: 
                    del self.encoder[token]
                if id in self.decoder:
                    del self.decoder[id]
                if token in self.base_encoder: 
                    del self.base_encoder[token]
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
                    self.del_token_by_id(del_items2)
                    self.flush() # if we don't save here, we will need to do a sanity check when we start up the daabase to check deleted items

    def cleanup_kb(self, min_cluster_size=-1, updated_items=[], recompute_means=True, rng_step=10000, token_type=None):
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
            ids = list(self.get_token_iter(field=self.token_meta.ID, field2=self.token_meta.LEVEL, ids=ids, token_type=token_type))
            if _pyVer==2:
                ids.sort(lambda a, b: cmp(b[1], a[1]))
            else:
                ids.sort(key=lambda a: a[1], reverse=True)
            while True:
                ids2 = []
                for a_rec in self.get_token_iter(ids=[i[0] for i in ids]):
                    cnt += 1
                    id = a_rec[self.token_meta.ID]
                    level = a_rec[self.token_meta.LEVEL]
                    (parents, children) = [i for i in a_rec[self.token_meta.PARENTS] if i >= 0], [i for i in a_rec[self.token_meta.CHILDREN] if i >= 0], 
                    for a in children or []:
                        if not self.token_exists(a):
                            children.remove(a)
                    if level > 0 and min_cluster_size > 0 and len(children) < min_cluster_size:
                        all_del_ids.append(id)
                        declustered_items.extend(children)
                        self.del_token(id)
                        ids2.extend([(i, level-1) for i in children])
                        ids2.extend([(i, level+1) for i in parents])
                    else:
                        if parents and not self.token_exists(parents[0]):
                            a_rec = self.get_token(id)
                            self.set_token_array_field(a_rec,self.token_meta.PARENTS,[])
                            self.set_token(id, a_rec)
                        elif parents:
                            parents = [parents[0]] + self.get_token(parents[0])[self.token_meta.PARENTS]
                            parents = [a for a in parents if self.token_exists(a) and a >= 0]
                            if tuple(parents) != tuple([i for i in a_rec[self.token_meta.PARENTS] if i >= 0]):
                                a_rec = self.get_token(id)
                                self.set_token_array_field(a_rec,self.token_meta.PARENTS,parents)
                                self.set_token(id, a_rec)
                        if tuple(children) != tuple([i for i in a_rec[self.token_meta.CHILDREN] if i >= 0]) and recompute_means:
                            a_rec = self.get_token(id)
                            # should this be all children or just the immediate children. this could get very expensive at the top.
                            if children != []:
                                updated_parent_embeds[a_rec[self.token_meta.ID]] = np.mean(self.embed_weight_data()[children], axis=0)
                            if cnt % rng_step == 0:
                                for id3 in list(updated_parent_embeds.keys()):
                                    if not self.token_exists(id3):
                                        del updated_parent_embeds[id3]
                                updated_parent_items = updated_parent_embeds.items()
                                self.embed_weight_data()[[i[0] for i in updated_parent_items]] = np.array([i[1] for i in updated_parent_items])
                                self.flush()
                                updated_parent_embeds={}
                                updated_parent_items = None
                            self.set_token_array_field(a_rec,self.token_meta.CHILDREN,children)
                            if children: 
                                a_rec[self.token_meta.CNT] = self.get_token(children[0], field=self.token_meta.CNT)
                            self.set_token(id, a_rec)
                if not ids2:
                    break
                ids = ids2
                ids = remove_duplicates([id for id in ids if self.token_exists(id[0])])
                if _pyVer==2:
                    ids.sort(lambda a, b: cmp(b[1], a[1]))
                else:
                    ids.sort(key=lambda a: a[1], reverse=True)
            for id3 in list(updated_parent_embeds.keys()):
                if not self.token_exists(id3):
                    del updated_parent_embeds[id3]
            updated_parent_items = updated_parent_embeds.items()
            declustered_items = remove_duplicates([id for id in declustered_items if self.token_exists(id)])
            self.embed_weight_data()[[i[0] for i in updated_parent_items]] = np.array([i[1] for i in updated_parent_items])
            updated_parent_embeds={}
            updated_parent_items = None
            for id in all_del_ids:
                a_rec = self.get_token(id)
                if a_rec:
                    logger.info(("deleting ", a_rec))
            self.del_token(all_del_ids)
            self.save()
            return (all_del_ids, [i for i in declustered_items if self.token_exists(i)])

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
        self.create_compound_token_count()
        self.save()
        return self

    def create_compound_token_count(self, ids=None):
        a_hash = {}
        if ids is None:
            ids = range(self.len_())
        token_hash = {}
        for id in ids:
            r = self.id_to_token(id)
            if not r: continue
            if "#" in r or "_" not in r:
                continue
            c = (r.split("_")[0], r.count("_")+1)
            if c[0] == '':
                continue
            token2 = c[0]
            if self.has_token(token2):
                a_hash[token2] = max(a_hash.get(token2,0), c[1])
                token_hash[id] = token2
            token2 = c[0].lower()
            if self.has_token(token2):
                a_hash[token2] = max(a_hash.get(token2,0), c[1])
                token_hash[id] = token2
            token2 = c[0].upper()
            if self.has_token(token2):
                a_hash[token2] = max(a_hash.get(token2,0), c[1])
                token_hash[id] = token2
            if len(c[0]) > 1:
                token2 = c[0][0].upper()+c[0][1:].lower()
                if self.has_token(token2):
                    a_hash[token2] = max(a_hash.get(token2,0), c[1])
                    token_hash[id] = token2
        tokens = list(a_hash.keys())
        ids = [self.token_to_id(token) for token  in tokens]
        tokens = []
        for a_rec in self.get_token_iter(ids=ids):
           a_rec[self.token_meta.COMPOUND_WORD] = a_hash[a_rec[self.token_meta.TOKEN]]
           tokens.append(a_rec)
        self.set_token(tokens)

    # orthographic level methods
    def split_stem(self, text, params=None, transStopWord=False, keepStopWord=False, stemStopWord=False, maxLen=7, minLen=4, autoTrim=True, transWord=False, retByRec=False, recno=1, retByDetails=False, do_split=True, keep_hash_tag_rec=True, split_char=None):
        """
        given a text string, split into words and stem the words using a trie data structure, stemTable
        """

        stopWordHash = self.stopWords
        stemTable = self.stemTable
        if isinstance(text, ListType):
            textArr = text
        else:
           if not do_split:
              textArr = [text.lower()] #.translate(trannum)
           elif split_char is None:
              textArr = text.lower().split() #.translate(trannum)
           else:
              textArr = text.lower().split(split_char) #.translate(trannum)
        textArr2 = []
        ret = []
        for word in textArr: 
            #word = word.strip(",.:'\"") # could speed this up a bit more by making some of these split chars
            if word[0] == '#' and keep_hash_tag_rec:
                ret.append(word)
                continue
            lenWord = len(word)
            if word[-1] == '#':
                word = word[:lenWord-1]
                lenWord -=1
            if lenWord == 1 and not keepStopWord:
                continue
            if lenWord == 0:
                continue
            if not stemStopWord:
                stop2tag = stopWordHash.get(word)
                if stop2tag != None: 
                    if transStopWord:
                        if retByRec:
                            ret.append([])
                        elif retByDetails:
                            ret.append(([], None, stop2tag))
                        else:
                            ret.append(stop2tag)
                    elif keepStopWord:
                        if retByRec:
                            ret.append(self.get_token(word, recno=recno))
                        elif retByDetails:
                            ret.append((self.get_token(word, recno=recno), None, stop2tag))
                        else:
                            ret.append(word)
                    continue
            retry=trie_find(stemTable, word, prefixSuffix=0)
            stemmed = False
            details = None
            while retry:
                #logger.info('==='+str(retry))
                lenEnding = retry[1]
                if lenWord < retry[0][1]:
                    break
                #logger.info(word + " " + str(retry))
                lenWord = lenWord-lenEnding
                if transWord:
                    foundTran = False
                    for targetEndingLen, targetEnding, wordEnding in retry[0][2]:
                        #logger.info(" +++ " + word[lenWord-1] + " " +  wordEnding)
                        if word[lenWord-1] == wordEnding:
                            details = (retry[0][0], retry[0][1], targetEndingLen, targetEnding, wordEnding)
                            foundTran=True
                            break
                    if not foundTran:
                        break
                    word = word[:lenWord]
                    if targetEndingLen > 0:
                        word += targetEnding
                        lenWord += targetEndingLen
                else:
                    word = word[:lenWord]
                if retByRec and self.token_to_id(word) is not None:
                    break
                retry=False
                stemmed=True
                if lenWord > minLen:
                    retry=trie_find(stemTable,  word, prefixSuffix=0)
            if retByRec or retByDetails:
                if self.token_to_id(word) is not None:
                    if retByDetails:
                        ret.append((self.get_token(word, recno=recno), details, word))
                    else:
                        ret.append(self.get_token(word, recno=recno))
                    continue
            if autoTrim and lenWord > maxLen and "_" not in word: # need to take care of compound words
                word = word[:maxLen]
            if retByRec or retByDetails:
                if self.token_to_id(word) is not None:
                    if retByDetails:
                        ret.append((self.get_token(word, recno=recno), details, word))
                    else:
                        ret.append(self.get_token(word, recno=recno))
                else:
                    if retByDetails:
                        ret.append(([], details, word))
                    else:
                        ret.append([])                        
            else:
                ret.append(word)
        return ret
   
    def parse_sentence(self, sent, parse_level=1):
        """
        basic parsing of a sentence. 

        parse_level 1 will collapse the longest possible compound words
        and split off puncutations at the end of words. 
        
        parse_level 2 will do 1 actions as well as
        replace a word with a rec if there is one

        parse_level 3 will do 1 and 2 actions as well as
        replace a word with a guessed rec if there is one

        TODO: parse_level 4 will do 1, 2 and 3 and will do basic word
        rec disambiguation in a sliding window.

        # TODO, internal sentence matching: he walked to the store and walked to the park
        # TODO: intra sentence matching. john walked to the store. he walked to the park.
        # TODO: basic non ordered sequence matching. john walked to the store to buy food. in the store, he buys chips. 
        # TODO, if pronouns are in either prev or currline, do:
        # -- basic anaphoric substituations. male->he, female->she, non-person->it, location->there. 
        
        """
        ret = []
        if isinstance(sent, StringType):
            sent = sent.replace("'", " '").replace('"',' " ').strip().split()
        lenSent = len(sent)
        i = 0
        while i < lenSent:
            extraWord = ""
            prevWord = ""
            wordNoPrevNoExtra=None
            word = sent[i]
            if word[0] == '#' or word[-1] == '#':
                ret.append(word)
                i+=1 
                continue
            word = cleanup_word(word)
            lenWord = len(word)
            idx = self.token_to_id(word)
            if idx is None:
                idx = self.token_to_id(word.lower())
            if idx is None:
                word3 = word
                lenWord3 = lenWord
                for j in range(3):
                    if  word3[-1] in "$,.:;?!`'\()[]" and lenWord3 > 1:
                        extraWord = word3[-1]+extraWord
                        word3 = word3[:lenWord3-1]
                        lenWord3-=1
                        if  word3[0] in "$,.:;?!`'\()[]" and lenWord3 > 2:
                            prevWord = prevWord+word3[0]
                            word3 = word3[1:]
                            lenWord3-=1
                        continue
                    elif  word3[0] in "$,.:;?!`'\()[]" and lenWord3 > 1:
                        prevWord = prevWord+word3[0]
                        word3 = word3[1:]
                        lenWord3-=1                        
                        continue
                    break
                if prevWord or extraWord:
                    wordNoPrevNoExtra = word3
                    idx = self.token_to_id(word3)
                    if idx is None:
                        idx = self.token_to_id(word3.lower())
                    if idx is None and parse_level > 2:
                        aRec2 = self.guess_token(word3)
                        if aRec2:
                            idx = aRec2[self.token_meta.IDX]
                    if idx is not None:
                        if prevWord: ret.append(prevWord)
                        if parse_level > 1:
                            ret.append(self.kb[idx][self.token_meta.TOKEN]+"#")
                        else:
                            ret.append(self.kb[idx][self.token_meta.TOKEN])                        
                        if extraWord: ret.append(extraWord)
                        i+=1 
                        continue
            if idx is None and parse_level > 2:
                aRec2 = self.guess_token(word)
                if aRec2:
                    idx = aRec2[self.token_meta.IDX]

            # unknown word
            if idx is None:
                if wordNoPrevNoExtra is not None:
                    if prevWord: ret.append(prevWord)
                    ret.append(wordNoPrevNoExtra)
                    if extraWord: ret.append(extraWord)
                else:
                    ret.append(word)
                i+=1 
                continue

            # this word is known in the vocab
            aRec = self.kb[idx]
            if aRec[self.token_meta.COMPOUND_CNT]>0:
                prevWord2 = ""
                extraWord2 = ""
                compoundSize = aRec[self.token_meta.COMPOUND_CNT]
                found = False
                for j in range(min(compoundSize, lenSent-i), 1, -1):
                    word2 = "_".join(sent[i:i+j])
                    lenWord2 = len(word2)
                    idx2 = self.token_to_id(word2)
                    if idx2 is None:
                        idx2 = self.token_to_id(word2.lower())
                    if idx2 is None and parse_level > 2:
                        aRec2 = self.guess_token(word2)
                        if aRec2:
                            idx2 = aRec2[self.token_meta.IDX]
                    if idx2 is None:
                        word3 = word2
                        lenWord3 = lenWord2
                        for j in range(3):
                            if  word3[-1] in "$,.:;?!`'\()[]" and lenWord3 > 1:
                                extraWord2 = word3[-1]+extraWord2
                                word3 = word3[:lenWord3-1]
                                lenWord3-=1
                                if  word3[0] in "$,.:;?!`'\()[]" and lenWord3 > 2:
                                    prevWord2 = prevWord2+word3[0]
                                    word3 = word3[1:]
                                    lenWord3-=1
                                continue
                            elif  word3[0] in "$,.:;?!`'\()[]" and lenWord3 > 1:
                                prevWord2 = prevWord2+word3[0]
                                word3 = word3[1:]
                                lenWord3-=1                        
                                continue
                            break
                        if prevWord2 or extraWord2:
                            idx2 = self.token_to_id(word3)
                            if idx2 is None:
                                idx2 = self.token_to_id(word3.lower())
                            if idx2 is None and parse_level > 2:
                                aRec2 = self.guess_token(word3)
                                if aRec2:
                                    idx2 = aRec2[self.token_meta.IDX]
                    if idx2 is not None:
                        if prevWord2: ret.append(prevWord2)
                        if parse_level > 1:
                            ret.append(self.kb[idx2][self.token_meta.TOKEN]+"#")
                        else:
                            ret.append(self.kb[idx2][self.token_meta.TOKEN])                                
                        if extraWord2: ret.append(extraWord2)
                        found=True
                        break
                if found:
                    i += j
                    continue
            if parse_level > 1:
                ret.append(self.kb[idx][self.token_meta.TOKEN]+"#")
            else:
                ret.append(self.kb[idx][self.token_meta.TOKEN])
            i+=1                 
        #logger.info(ret)
        return ret



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
