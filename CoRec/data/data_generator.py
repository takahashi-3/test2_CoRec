import nltk.corpus
#import en_core_web_sm
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.translate.ribes_score import position_of_ngram


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
                if (root[i].label() == 'CC') and (root[i].leaves()[0] != '+/-'):
                    if i != 0:
                        CC_label = True
                elif root[i].label() == 'CONJP':
                    if i != 0:
                        CC_label = True
                        conj_lst = root[i].leaves()
                        if len(conj_lst) == 2:
                            if conj_lst[0] == 'but' and conj_lst[1] == 'also':
                                CC_label = False
                        for j in range(len(conj_lst)):
                            if conj_lst[j] == "not":
                                CC_label = False
                                break
                
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
            
            elif tree[i].label() == 'CC' or tree[i].label() == 'CONJP':
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
            conj_word_sid = span[l-2][1] + 1
            conj_word_eid = span[l-1][0] - 1
            start_id = span[0][0]
            end_id = span[l-1][1]
            labels = []

            for j in range(0,l):
                if j == 0:
                    k = 0
                    first = True
                    while (k < start_id):
                        labels.append('O')
                        k += 1
                    i = span[j][0]
                    while (i < span[j+1][0]):
                        if (i >= conj_word_sid) and (i <= conj_word_eid):
                            if s[i] == ',':
                                labels.append('I-before')
                            else:    
                                labels.append('C')
                        else:
                            if first == True:
                                labels.append("B-before")
                                first = False
                            else:
                                labels.append("I-before")
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
                        if (i >= conj_word_sid) and (i <= conj_word_eid):
                            if s[i] == ',':
                                labels.append('I-before')
                            else:    
                                labels.append('C')
                        else:
                            if first == True:
                                labels.append("B-before")
                                first = False
                            else:
                                labels.append("I-before")
                        i += 1  
                
            labels_set.append(labels)      
    return labels_set            
            

def spanChecker(span):
    l = len(span)
    result = True
    for i in range(0,l):
        if i < l - 1:
            diff = span[i+1][0] - span[i][1]
            
            if diff < 2 or diff > 5:
                result = False
    return result    


#remove the meaningless '*' words from token_s and return a clean sentence s 
def starRemover(token_s):
    token_s_new = []
    signal = "*"
    for t in token_s:
        if (signal in t) or t == '"':
            continue
        else:
            token_s_new.append(t)
    s = TreebankWordDetokenizer().detokenize(token_s_new)
    return s

"""
#Penn
inFolder = "./wsj/train"
outF = open("data/wsj_train_pre.txt", "a")
reader = nltk.corpus.BracketParseCorpusReader(inFolder, ".*mrg")
"""


#GENIA
inFolder = "./GENIA"
outF = open("pre_data/genia_test_pre.txt", "a")
reader = nltk.corpus.BracketParseCorpusReader(inFolder, ".*trees")


"""
#Ontonotes
#remember to add 2000 limit
inFolder = "./annotations"
outF = open("pre_data/ontoTest_pre.txt", "a")
reader = nltk.corpus.BracketParseCorpusReader(inFolder, ".*parse")
"""


trees = reader.parsed_sents()

count = 0
total_cnt = 0
for tree in trees:
    try:  
        if count >= 1000:
            break
        token_s = tree.leaves()
        lst = findConjunctions(tree)
        result = findPositions(token_s, lst)

        if len(result) == 0:
            continue
        else:
            count += 1
            #remove some strange words like '*-1', '*PRO*'...
            #s = starRemover(token_s)
            labels_set = getLabels(token_s, result)

            num_labs = len(labels_set)
            if num_labs > 0:
                outF.write("\n")
                outF.write("\n")
                len_s = len(labels_set[0])
                for i in range(0,len_s):
                    signal = "*"
                    if (signal in token_s[i]) or token_s[i] == '"':
                        continue
                    else:
                        outF.write(token_s[i])
                        
                        special = False
                        found = False
                        if token_s[i] == 'either' or token_s[i] == 'Either' or token_s[i] == 'neither' or token_s[i] == 'Neither' or token_s[i] == 'between' or token_s[i] == 'Between':
                            special = True
                        
                        for k in range(0, num_labs):    
                            outF.write("\t")
                            if special == True:
                                if i < (len_s - 1):
                                    #handle 'between'
                                    if token_s[i] == 'between' or token_s[i] == 'Between':
                                        for j1 in range(i+1,len_s):
                                            if token_s[j1] == 'and':
                                                if labels_set[k][j1] == 'C':
                                                    labels_set[k][i] = 'C'
                                                    for cj in range(i+1, j1):
                                                        if cj == i+1:
                                                            labels_set[k][cj] = 'B-before'
                                                        else:
                                                            labels_set[k][cj] = 'I-before'                                              
                                                found = True
                                                break
              
                                    #handle 'either', 'neither' 
                                    else:
                                        if labels_set[k][i+1] == 'B-before' and found == False:
                                            for j2 in range(i+1,len_s):
                                                if 'or' in token_s[j2] and labels_set[k][j2] == 'C':
                                                    labels_set[k][i] = 'C'
                                                    found = True
                                            
                                        if labels_set[k][i] == 'B-before' and labels_set[k][i+1] == 'I-before':
                                            labels_set[k][i] = 'C'
                                            labels_set[k][i+1] = 'B-before'
                                        
                
                            outF.write(labels_set[k][i]) 
                            
                    outF.write("\n")

    except Exception:
        pass



print("Number of conjunctive sentences in datasets:")
print(count)
print("Total number of sentences in datasets:")
print(len(trees))
print(total_cnt)


outF.close()