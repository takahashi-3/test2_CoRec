import time
import numpy as np
#import en_core_web_sm
from nltk.tree import Tree
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.translate.ribes_score import position_of_ngram

import os
from nltk.parse import stanford


os.environ['STANFORD_PARSER'] = 'stanford-parser-full-2015-12-09/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar'


def isLineEmpty(line):
    return len(line.strip()) < 1

# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


#given a parse tree (nltk tree object)
#return the subtree that contains 'CC' subtree and its siblings
def findConjunctions(root):
    #helper function to traverse the whole tree, find and store conjunctions
    def helper(root):
        if type(root) == str:
            return
        
        CC_label = False
        for i in range(len(root)):
            if type(root[i]) != str:
                if root[i].label() == 'CC':
                    CC_label = True
        #if the current subtree contains 'CC' subtree
        if CC_label == True:
            lst.append(root)
        
        #BFS: next layer
        for i in range(len(root)):
            helper(root[i])
        return            
    
    if root is None:
        return []
    
    lst = []
    helper(root)
    return lst


#given [tokenized sentence] and a set of subtrees containing conjunctions
#find the set of spans (start index, end index)
def findPositions(s, lst):
    spans = []
    for tree in lst:
        
        tree = list(tree) #berkeley test
        
        if tree is None:
            continue
        i = 0
        tree_spans = []
        cur_string = []
        start = 0
        end = 0

        while (i < len(tree)):
            if tree[i].label() == ',':
                if i != 0:
                    start = position_of_ngram(tuple(cur_string), s)
                    if start != None:
                        end = start + len(cur_string) - 1
                        tree_spans.append([start, end])
                    cur_string = []
            
            elif tree[i].label() == 'CC':
                if i > 0:
                    #if it has already been handled
                    if tree[i-1].label() == ',':
                        i += 1
                        continue
                    else:
                        start = position_of_ngram(tuple(cur_string), s)
                        if start != None:
                            end = start + len(cur_string) - 1
                            tree_spans.append([start, end])
                        cur_string = []
            
            else:
                #exclude '.'
                if tree[i].label() != '.':    
                    cur_string.extend(tree[i].leaves())
                if i == len(tree) - 1:
                    #print(tree[i].leaves())
                    start = position_of_ngram(tuple(cur_string), s)
                    if start != None:
                        end = start + len(cur_string) - 1
                        tree_spans.append([start, end])
                    cur_string = []    
            i += 1
        spans.append(tree_spans)
     
    return spans


#given sentence and the set of spans (start index, end index)
#return the set of labels              
def getLabels(s, spans):
    if spans is None:
        return None
    
    labels_set = []
    for span in spans:
        if (len(span) < 2):
            continue
        if spanChecker(span):
            l = len(span)
            conj_word_id = span[l-1][0] - 1
            start_id = span[0][0]
            end_id = span[l-1][1]
            labels = []

            for j in range(0,l):
                cur_str = "I-before"
                if j == 0:
                    k = 0
                    first = True
                    while (k < start_id):
                        labels.append('O')
                        k += 1
                    i = span[j][0]
                    while (i < span[j+1][0]):
                        if i == conj_word_id:
                            labels.append('C')
                        else:
                            if first == True:
                                labels.append("B-before")
                                first = False
                            else:
                                labels.append(cur_str)
                        i += 1    
                    
                elif (j == l-1):
                    i = span[j][0]
                    first = True
                    while (i <= end_id):
                        if first == True:
                            labels.append('B-after')
                            first = False
                        else:
                            labels.append('I-after')
                        i += 1    
                    k = end_id + 1
                    while (k < len(s)):
                        labels.append('O')
                        k += 1
                
                else:
                    i = span[j][0]
                    first = True
                    while (i < span[j+1][0]):
                        if i == conj_word_id:
                            labels.append('C')
                        else:
                            if first == True:
                                labels.append("B-before")
                                first = False
                            else:
                                labels.append(cur_str)
                        i += 1  
                
            labels_set.append(labels)      
    return labels_set            
            

def spanChecker(span):
    l = len(span)
    result = True
    for i in range(0,l):
        if i < l - 1:
            diff = span[i+1][0] - span[i][1]
            if diff != 2 and diff != 3:
                result = False
    return result    


# the following two functions are slightly different from those two in data_utils.py        
def get_chunk_type(tag_name):
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type
    
    
def get_chunks(seq):
    default = "O"
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # only end of a chunk
        if tok == default and chunk_type is not None:
            # add a chunk.
            chunk = (chunk_type, chunk_start, i)
            if chunk_type != "C":
                chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # end of a chunk + start of another chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i

            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                if chunk_type != "C":
                    chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


# if the text contain comma
def lsProc(ls):
    outls = []
    if len(ls) == 5:
        outls.append(ls[0])
        outls.append(",")
        outls.append(ls[3])
        outls.append(ls[4])
    
    return outls



def main():

    inF = open("data/wsj/wsj_test_simple.csv", "r")
    #inF = open("data/GENIA/genia_test_simple.csv", "r")
    #inF = open("data/ontoTest/ontoTest_simple.csv", "r")
    

    #---constituency parsing---
    predictor = stanford.StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    
    total_correct = 0
    correct_Allen_parser = 0
    total_Allen_parser = 0
    
    Lines = inF.readlines()
    inF.close()
    
    j = 0
    id_cnt = 0
    conj_id = -1
    sNums, token_s, labels = [], [], []
    cur_sNum = ""
    
    total_sentences = 0
    
    # start time
    print(time.localtime(time.time()))
    
    while (j < len(Lines) - 1):
        ls = Lines[j].split(',')

        if len(ls) != 4:
            ls = lsProc(ls)
        cur_sNum = ls[0]
        
        # a new sentence
        if len(sNums) != 0 and cur_sNum != sNums[0]:
            if (len(token_s) != 0) and (conj_id != -1):
                #s = TreebankWordDetokenizer().detokenize(token_s)
                labels_Allen_parser = []
                total_sentences += 1         
                
                try:                    
                    output = predictor.parse(token_s)
                    temp = list(output)
                    tree = temp[0]
                    #print(tree)
                    lst = findConjunctions(tree)
                    
                    result = findPositions(token_s, lst)
                    target_span = []
                    for span in result:
                        l = len(span)    
                        if l > 0 and conj_id == span[l-1][0] - 1:
                            target_span = span
                            break
                    
                    if len(target_span) > 0:
                        labels_set = getLabels(token_s, [target_span])
                        labels_Allen_parser = labels_set[0]                     
                
                except Exception:
                     pass  
                
            
                lab_chunks = set(get_chunks(labels))
                lab_Allen_parser_chunks = set(get_chunks(labels_Allen_parser))
                
                total_correct += len(lab_chunks)
                
                correct_Allen_parser += len(lab_chunks & lab_Allen_parser_chunks)
                total_Allen_parser += len(lab_Allen_parser_chunks)

                id_cnt = 0
                conj_id = -1
                sNums, token_s, labels = [], [], []
                test = []
                
        
        if ls[1] == '[C]':
            if conj_id == -1:
                conj_id = id_cnt
        else:
            sNums.append(ls[0])
            token_s.append(ls[1])
            labels.append(ls[3].rstrip('\n'))
            id_cnt += 1 
        j += 1    
    

    
    p = correct_Allen_parser / total_Allen_parser if correct_Allen_parser > 0 else 0
    r = correct_Allen_parser / total_correct if correct_Allen_parser > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_Allen_parser > 0 else 0

    
    print("Stanford parser precision:")
    print(p)
    print("Stanford parser recall:")
    print(r)
    print("Stanford parser f1:")
    print(f1)
    print()
    
    # end time
    print(time.localtime(time.time()))
    print()
    
    
    print("Total number of sentences:")
    print(total_sentences)
    
  
if __name__ == "__main__":
    main()
    
