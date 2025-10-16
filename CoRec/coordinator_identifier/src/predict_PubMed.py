import joblib
import torch
import csv
import os
import config
import dataset
from train import process_data
from model import EntityModel
import nltk
import json

# 10/15 add
nltk.download('punkt_tab')

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


# eliminate the sub-words
def getCleanLabels(raw_tokens, raw_tags):
    tokens, tags = [], []
    tmp_str = ""
    tmp_tag = ""
    for i, t in enumerate(raw_tokens):
        if i == 0:
            # the first and last tokens are [CLS] and [SEP], should be skipped
            continue
        else:
            if raw_tokens[i].startswith("##"):
                tmp_str += raw_tokens[i].lstrip("##")
            elif raw_tokens[i] == 'c':
                if tmp_str == '[':
                    tmp_str += raw_tokens[i]
                else:
                    if tmp_str != "":
                        tokens.append(tmp_str)
                        tags.append(tmp_tag)
                    tmp_str = raw_tokens[i]
                    tmp_tag = raw_tags[i]
            elif raw_tokens[i] == ']':
                if tmp_str == '[c':
                    tmp_str += raw_tokens[i]
                else:
                    if tmp_str != "":
                        tokens.append(tmp_str)
                        tags.append(tmp_tag)
                    tmp_str = raw_tokens[i]
                    tmp_tag = raw_tags[i]
            else:
                if tmp_str != "":
                    tokens.append(tmp_str)
                    tags.append(tmp_tag)
                if raw_tokens[i] == "[SEP]":
                    break
                tmp_str = raw_tokens[i]
                tmp_tag = raw_tags[i]

    return tokens, tags



def split_target_c(sen_id, tks, cTags, offs):
    cur_sen_id = sen_id
    split_lst = []
    inC = False
    for i_c in range(len(cTags)):
        tmp_i = i_c
        if cTags[i_c] == 'C' and inC == False:
            inC = True
            cur_sen_id += 1
            t1 = "Sentence: " + str(cur_sen_id)
            for i_t in range(tmp_i):
                split_lst.append([t1, tks[i_t], 'c-' + cTags[i_t], 'O', offs[i_t]])
            split_lst.append([t1, '[C]', 'c-C', 'O', str([-1, -1])])
            split_lst.append([t1, tks[tmp_i], 'c-C', 'O', offs[tmp_i]])
            for i_t in range(tmp_i+1, len(cTags)):
                if cTags[i_t] == 'O' and inC == True:
                    inC = False
                    # step out the target coordinator
                    i_c = i_t
                    split_lst.append([t1, '[C]', 'c-C', 'O', str([-1, -1])])
                    split_lst.append([t1, tks[i_t], 'c-O', 'O', offs[i_t]])
                #elif cTags[i_t] == 'C' and inC == True:
                #    split_lst.append([t1, tks[i_t], 'c-C', 'O'])
                else:
                    split_lst.append([t1, tks[i_t], 'c-' + cTags[i_t], 'O', offs[i_t]])

    return cur_sen_id, split_lst



if __name__ == "__main__":


    test_model_path = "saved_models/model_c.bin"

    meta_data = joblib.load("saved_models/meta_c.bin")

    enc_train_tag = meta_data["enc_train_tag"]

    num_tag = len(list(enc_train_tag.classes_))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(test_model_path))
    model.to(device)

    inFolder = "../data/PubMed/"
    #PMIDs = ['29108061', '28341048', '24879756', '24634129', '20963633', '20373023', '20059931', '19763868', '19620661',
    #         '19181764', '22100596', '22436149', '22443815', '22445144', '23508028', '26440326']
    PMIDs = ['29108061']
    outFolder = "../predictions/"

    for pmid in PMIDs:

        path = outFolder + pmid
        if not os.path.exists(path):
            os.makedirs(path)

        print()
        print(pmid)

        inF = open("../../PubMed_abstracts/" + pmid + ".txt", "r")
        ori_Lines = inF.readlines()

        in_sentences = nltk.sent_tokenize(ori_Lines[0])
        num_sen = len(in_sentences)
        inF.close()
        #if pmid == '23508028':
        #    print(in_sentences)

        for f_cnt in range(1, num_sen+1):
            print(f_cnt)
            test_file = inFolder + pmid + "/" + str(f_cnt) + "_c.csv"

            test_sentences, test_tag, enc_test_tag = process_data(test_file)
            test_dataset = dataset.EntityDataset(
                texts=test_sentences, tags=test_tag
            )

            outF = open(path + "/" + str(f_cnt) + "_c_pred.csv", "w")
            fieldnames = ['Sentence #', 'Text', 'c-Tag', 'Tag', 'Offset']
            writer = csv.DictWriter(outF, fieldnames=fieldnames)
            writer.writeheader()

            correct_preds = 0
            total_preds = 0
            total_correct = 0
            sentence_cnt = 0

            offsets = []
            testF = open(test_file, "r")
            Lines = csv.reader(testF)
            is_fst = True
            for l in Lines:
                # skip the fieldnames
                if is_fst is True:
                    is_fst = False
                    continue
                offsets.append(l[3])
            #print(offsets)
            testF.close()

            with torch.no_grad():
                for j in range(len(test_dataset)):
                    try:
                        data = test_dataset[j]
                        tokenized_sentence_ids = data["ids"]
                        target_tags = data["target_tag"]
                        tokenized_sentence = config.TOKENIZER.convert_ids_to_tokens(tokenized_sentence_ids)

                        for k, v in data.items():
                            data[k] = v.to(device).unsqueeze(0)
                        tag, _ = model(**data)


                        raw_tags = enc_train_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[
                                   :len(tokenized_sentence_ids)]
                        #print(raw_tags)
                        # raw_tags = enc_train_tag.inverse_transform(tag[0])[:len(tokenized_sentence_ids)]
                        raw_target_tags = enc_train_tag.inverse_transform(target_tags)[
                               :len(tokenized_sentence_ids)]
                        #print(raw_target_tags)

                        tokens, labels = getCleanLabels(tokenized_sentence, raw_tags)
                        _, target_labels = getCleanLabels(tokenized_sentence, raw_target_tags)

                        #10/15 add
                        print(f"(predicted) \n tokens: {tokens}\n labels: {labels}")

                        # BERT tokenizer might be different from nltk tokenizer, need to align them
                        aligned_offsets = []
                        if len(tokens) != len(offsets):
                            #sep = False
                            sep_offset = []
                            sep_pace = 0
                            m = 0  # token pace
                            for n in range(len(offsets)):
                                #print(len(tokens[m]))
                                #print(offsets[n])
                                offsets_lst = json.loads(offsets[n])
                                #print(offsets_lst[0])
                                #print(offsets_lst[1]-offsets_lst[0]+1)
                                #print()
                                if len(tokens[m]) == (offsets_lst[1]-offsets_lst[0]+1):
                                    aligned_offsets.append(offsets[n])
                                    m += 1
                                else:
                                    sep_offset.append(offsets_lst[0])
                                    sep_offset.append(offsets_lst[1])
                                    aligned_offsets.append(str([sep_offset[0], sep_offset[0]+len(tokens[m])-1]))
                                    sep_pace = offsets_lst[0] + len(tokens[m])
                                    m += 1
                                    while(sep_pace <= sep_offset[1]):
                                        aligned_offsets.append(str([sep_pace, sep_pace+len(tokens[m])-1]))
                                        sep_pace += len(tokens[m])
                                        m += 1
                        else:
                            aligned_offsets = offsets

                        #print(len(tokens) == len(aligned_offsets))

                        cur_sen_id, split_lst = split_target_c(sentence_cnt, tokens, labels, aligned_offsets)
                        sentence_cnt = cur_sen_id
                        for line in split_lst:
                            writer.writerow({'Sentence #': line[0], 'Text': line[1], 'c-Tag': line[2], 'Tag': 'O', 'Offset': line[4]})

                        num_matched_labels = 0
                        for m in range(len(labels)):
                            if labels[m] == target_labels[m]:
                                num_matched_labels += 1

                        correct_preds += num_matched_labels
                        total_preds += len(labels)
                        total_correct += len(target_labels)

                    except Exception as e:
                        print(e)

                outF.close()

            p = correct_preds / total_preds if correct_preds > 0 else 0
            r = correct_preds / total_correct if correct_preds > 0 else 0
            f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0


            print("precision:")
            print(p)
            print("recall:")
            print(r)
            print("f1 score:")
            print(f1)
