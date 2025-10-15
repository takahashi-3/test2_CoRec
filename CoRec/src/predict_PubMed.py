import os
import nltk
import joblib
import torch
import csv
import config
import dataset
from train import process_data
from model import EntityModel


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


if __name__ == "__main__":

    test_model_path = "saved_models/model4_2.bin"

    meta_data = joblib.load("saved_models/meta_2.bin")
    enc_train_ctag = meta_data["enc_train_ctag"]
    enc_train_tag = meta_data["enc_train_tag"]

    num_ctag = len(list(enc_train_ctag.classes_))
    num_tag = len(list(enc_train_tag.classes_))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EntityModel(num_ctag=num_ctag, num_tag=num_tag)
    model.load_state_dict(torch.load(test_model_path))
    model.to(device)

    inFolder = "../../c_classifier_new/predictions/"
    PMIDs = ['29108061', '28341048', '24879756', '24634129', '20963633', '20373023', '20059931', '19763868', '19620661',
             '19181764', '22100596', '22436149', '22443815', '22445144', '23508028', '26440326']

    for pmid in PMIDs:

        path = "../predictions/" + pmid
        if not os.path.exists(path):
            os.makedirs(path)

        print()
        print(pmid)

        inF = open("../../PubMed_abstracts/" + pmid + ".txt", "r")
        ori_Lines = inF.readlines()

        in_sentences = nltk.sent_tokenize(ori_Lines[0])
        num_sen = len(in_sentences)
        inF.close()

        for f_cnt in range(1, num_sen + 1):
            test_file = inFolder + pmid + "/" + str(f_cnt) + "_c_pred.csv"
            outF = open(path + "/" + str(f_cnt) + "_pred_0.csv", "w")
            fieldnames = ['Sentence #', 'Text', 'c-Tag', 'Tag', 'Offset']
            writer = csv.DictWriter(outF, fieldnames=fieldnames)
            writer.writeheader()

            test_sentences, test_ctag, test_tag, enc_test_ctag, enc_test_tag = process_data(test_file)
            test_dataset = dataset.EntityDataset(
                texts=test_sentences, ctags=test_ctag, tags=test_tag
            )

            correct_preds = 0
            total_preds = 0
            total_correct = 0
            sentence_cnt = 0

            offsets_lst = []
            testF = open(test_file, "r")
            Lines = csv.reader(testF)
            is_fst = True
            sen_cnt = 1
            tmp_offsets = []
            for l in Lines:
                # skip the fieldnames
                if is_fst is True:
                    is_fst = False
                    continue
                if str(sen_cnt) == l[0].split(" ")[1]:
                    tmp_offsets.append(l[4])
                else:
                    sen_cnt += 1
                    offsets_lst.append(tmp_offsets)
                    tmp_offsets = []
                    tmp_offsets.append(l[4])
            offsets_lst.append(tmp_offsets)
            #print(offsets_lst)
            testF.close()


            with torch.no_grad():
                for j in range(len(test_dataset)):
                    try:
                        data = test_dataset[j]
                        tokenized_sentence_ids = data["ids"]
                        target_ctags = data["target_ctag"]
                        target_tags = data["target_tag"]
                        tokenized_sentence = config.TOKENIZER.convert_ids_to_tokens(tokenized_sentence_ids)

                        for k, v in data.items():
                            data[k] = v.to(device).unsqueeze(0)
                        tag, _ = model(**data)

                        #raw_tags = enc_train_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[:len(tokenized_sentence_ids)]
                        raw_tags = enc_train_tag.inverse_transform(tag[0])[:len(tokenized_sentence_ids)]
                        #print(raw_tags)
                        raw_target_tags = enc_train_tag.inverse_transform(target_tags)[
                               :len(tokenized_sentence_ids)]
                        raw_target_ctags = enc_train_ctag.inverse_transform(target_ctags)[
                                          :len(tokenized_sentence_ids)]


                        tokens, labels = getCleanLabels(tokenized_sentence, raw_tags)
                        _, target_labels = getCleanLabels(tokenized_sentence, raw_target_tags)
                        _, target_clabels = getCleanLabels(tokenized_sentence, raw_target_ctags)

                        sentence_cnt += 1
                        t1 = "Sentence: " + str(sentence_cnt)
                        for t in range(len(tokens)):
                            writer.writerow({'Sentence #': t1, 'Text': tokens[t], 'c-Tag': target_clabels[t], 'Tag': labels[t], 'Offset': offsets_lst[j][t]})

                        lab_chunks = set(get_chunks(labels))
                        target_lab_chunks = set(get_chunks(target_labels))

                        correct_preds += len(target_lab_chunks & lab_chunks)
                        total_preds += len(lab_chunks)
                        total_correct += len(target_lab_chunks)

                    except Exception:
                        pass

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
