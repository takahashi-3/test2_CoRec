import csv
import torch
from pytorch_transformers import BertTokenizer
from pytorch_transformers import BertModel
from nltk.tokenize.treebank import TreebankWordDetokenizer


def isLineEmpty(line):
    return len(line.strip()) < 1


# This function converts several subword token embeddings into one word embedding 
# Like in BERT, for 'embedding', it will be tokenized as [em, ##bed, ##ding, ##s]
# We will take average of all the subword embeddings
def bert2word(tmp0, tokenized_text, all_embeddings):
    bert_word_embeddings = []
    words = []
    
    if len(all_embeddings) <= 2:
        return bert_word_embeddings, words
    
    else:
        cur_embed_sum = all_embeddings[0]
        cur_embed_num = 0
        cur_word = ''
        ori_cnt = 0
        # remember the first is [CLS] and the last is [SEP]
        for i in range(1, len(all_embeddings)):
            ori_token = tmp0[ori_cnt].lower() 
            token = tokenized_text[i]
            embed = all_embeddings[i]
            if token.startswith("##"):
                cur_embed_sum += embed
                cur_embed_num += 1
                tmp_token = token.lstrip("##")
                cur_word += tmp_token
            else:
                if cur_word != ori_token:
                    cur_embed_sum += embed
                    cur_embed_num += 1
                    cur_word += token
                    continue
                                   
                if cur_embed_num != 0:
                    bert_word_embeddings.append((cur_embed_sum/cur_embed_num).numpy())
                    words.append(cur_word)
                    ori_cnt += 1
                    cur_embed_num = 0
                    cur_word = ''
                cur_embed_sum = embed
                cur_embed_num += 1
                cur_word = token
        return bert_word_embeddings, words

          
def array2string(inarr):
    result = '['
    l = len(inarr)
    for i in range(0, l-1):
        result += str(inarr[i])
        result += ' '
    result += str(inarr[l-1]) 
    result += ']'
    
    return result
  
      
def tokens2string(tokens):
    result = '['
    l = len(tokens)
    if l == 0:
        return '[]'
    
    result += str(l)
    result += ': '
    for i in range(0, l-1):
        result += str(tokens[i])
        result += ' '
    result += str(tokens[l-1]) 
    result += ']'
    
    return result


#generate a group of target CC and labels from a specific column
def procColumn(lst, col):
    target_CC_lst_f = []
    CC_lst_f = []
    labels_lst_f = []
    
    len_s = len(lst)
    full_labels = []
    
    #serach CCs
    target_CC_ids = []
    start_id = -1
    end_id = -1
    for j in range(len_s):
        full_labels.append(lst[j][col])
        if lst[j][col] == 'C' or lst[j][col] == 'C\n':
            if start_id == -1:
                start_id = j
                end_id = j
            else:
                end_id += 1
        else:
            if start_id != -1:
                target_CC_ids.append([start_id, end_id])
                start_id = -1
                end_id = -1
    
    #generate taget_CC_lst
    prev_eid, next_sid = -1, 0 
    for k in range (len(target_CC_ids)):
        sid = target_CC_ids[k][0]
        eid = target_CC_ids[k][1]
        if k != len(target_CC_ids) - 1:
            next_sid = target_CC_ids[k+1][0]
        else:
            next_sid = len_s
        
        tmp_target_lst_f = []
        tmp_lst_f = []
        for l in range(len_s):
            if prev_eid < l and l < sid:
                tmp_target_lst_f.append('O')
                if '-before' in full_labels[l]:
                    tmp_lst_f.append(full_labels[l])
                else:
                    tmp_lst_f.append('O')
                          
            elif sid <= l and l <= eid:
                tmp_target_lst_f.append('C')
                tmp_lst_f.append('C')
            
            elif eid < l and l < next_sid:
                tmp_target_lst_f.append('O')
                if '-after' in full_labels[l]:
                    tmp_lst_f.append(full_labels[l])
                else:
                    tmp_lst_f.append('O')
                    
            else:
                tmp_target_lst_f.append('O')
                tmp_lst_f.append('O')
            
        target_CC_lst_f.append(tmp_target_lst_f)
        CC_lst_f.append(tmp_lst_f)
        prev_eid = eid
        
        
    #single
    if len(target_CC_ids) == 1:
        labels_lst_f.append(full_labels)
    #pair
    elif len(target_CC_ids) == 2:
        labels_lst_f.append(target_CC_lst_f[0])
        labels_lst_f.append(full_labels)
    #respectively (labeling schema changed)
    elif len(target_CC_ids) == 3:
        labels_lst_f.append(CC_lst_f[0])
        labels_lst_f.append(CC_lst_f[1])
        labels_lst_f.append(full_labels)
    else:
        target_CC_lst_f = []
    
    return target_CC_lst_f, labels_lst_f 
    
    

"""
inF = open("ori_anno_data/wsj_train.txt", "r")
outF = open("wsj_train_new.csv", "w")

#inF = open("ori_anno_data/wsj_dev.txt", "r")
#outF = open("wsj_dev_new.csv", "w")
"""

inF = open("ori_anno_data/wsj_test.txt", "r")
outF = open("wsj_test_all.csv", "w")


fieldnames = ['Sentence #', 'Text', 'c-Tag', 'Tag']
writer = csv.DictWriter(outF, fieldnames=fieldnames)
writer.writeheader()


Lines = inF.readlines()
t = []
i = 0
sentence_cnt = 1
target_CC_lst, labels_lst = [], []
tmp_lst = []
emp_mark = 2
#word, c-label
tmp0, tmp2 = [], []

while (i < len(Lines) - 1):
    ls = Lines[i].split('\t')

    if isLineEmpty(Lines[i]) and (len(tmp0) != 0):
              
        for col in range(1, emp_mark):
            lst1, lst2 = procColumn(tmp_lst, col)
            target_CC_lst.extend(lst1)
            labels_lst.extend(lst2)
        
        for n in range(len(target_CC_lst)):
            t_CC = target_CC_lst[n]
            labs = labels_lst[n]
            marked = False
            for j in range(0,len(tmp0)):
                t1 = "Sentence: " + str(sentence_cnt)
                t2 = tmp0[j]
                t3 = tmp2[j]
                if t_CC[j] == 'C':
                    t4 = 't-C'
                    if marked == False:
                        marked = True
                        writer.writerow({'Sentence #': t1, 'Text': '[C]', 'c-Tag': 'c-C', 'Tag': 'C'})
                else:
                    t4 = 't-O'
                    if marked == True:
                        marked = False
                        writer.writerow({'Sentence #': t1, 'Text': '[C]', 'c-Tag': 'c-C', 'Tag': 'C'})
                t5 = labs[j].rstrip('\n') 
                writer.writerow({'Sentence #': t1, 'Text': t2, 'c-Tag': t3, 'Tag': t5})   
            sentence_cnt += 1   
            #outF.write(Lines[i])
            #outF.write("\n")
            
            
        target_CC_lst, labels_lst = [], []
        tmp_lst = []
        emp_mark = 2
        tmp0, tmp2 = [], []
        i += 1
    
    else:
        if isLineEmpty(Lines[i]):
            target_CC_lst, labels_lst = [], []
            tmp_lst = []
            emp_mark = 2
            tmp0, tmp1, tmp2 = [], [], []
            i += 1
            continue
        
        tmp_lst.append(ls)
        tmp0.append(ls[0])
        token_c = 'c-O'
        for k in range(1, len(ls)):
            if ls[k] == '' or ls[k] == '\n':
                if emp_mark == 2:
                    emp_mark = k
                break
            if ls[k] == 'C' or ls[k] == 'C\n':
                token_c = 'c-C'
        #indicate all CCs
        tmp2.append(token_c)
        i += 1    



inF.close()
outF.close()     
