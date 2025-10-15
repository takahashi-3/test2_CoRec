import numpy as np
import joblib
import torch

import config
import dataset
import engine
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

    #test_file = "../data/wsj/wsj_test_simple.csv"
    #test_file = "../data/wsj/wsj_test_complicated.csv"

    test_file = "../data/GENIA/genia_test_simple.csv"
    #test_file = "../data/GENIA/genia_test_complicated.csv"

    #test_file = "../data/ontoTest/ontoTest_simple.csv"
    #test_file = "../data/ontoTest/ontoTest_complicated.csv"

    test_model_path = "saved_models/model4_2.bin"
    #outF = open("../../predictions/test_crf_allaug4_1.txt", "a")

    meta_data = joblib.load("saved_models/meta_2.bin")
    enc_train_ctag = meta_data["enc_train_ctag"]
    enc_train_tag = meta_data["enc_train_tag"]

    num_ctag = len(list(enc_train_ctag.classes_))
    num_tag = len(list(enc_train_tag.classes_))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EntityModel(num_ctag=num_ctag, num_tag=num_tag)
    model.load_state_dict(torch.load(test_model_path))
    model.to(device)

    test_sentences, test_ctag, test_tag, enc_test_ctag, enc_test_tag = process_data(test_file)
    test_dataset = dataset.EntityDataset(
        texts=test_sentences, ctags=test_ctag, tags=test_tag
    )


    correct_preds = 0
    total_preds = 0
    total_correct = 0

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

                #raw_tags = enc_train_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[:len(tokenized_sentence_ids)]
                raw_tags = enc_train_tag.inverse_transform(tag[0])[:len(tokenized_sentence_ids)]
                #print(raw_tags)
                raw_target_tags = enc_train_tag.inverse_transform(target_tags)[
                       :len(tokenized_sentence_ids)]


                tokens, labels = getCleanLabels(tokenized_sentence, raw_tags)
                _, target_labels = getCleanLabels(tokenized_sentence, raw_target_tags)

                """
                for t in range(len(tokens)):
                    outF.write(tokens[t])
                    outF.write("\t")
                    outF.write(target_labels[t])
                    outF.write("\t")
                    outF.write(labels[t])
                    outF.write("\n")
                outF.write("\n")
                outF.write("\n")
                """

                lab_chunks = set(get_chunks(labels))
                target_lab_chunks = set(get_chunks(target_labels))

                correct_preds += len(target_lab_chunks & lab_chunks)
                total_preds += len(lab_chunks)
                total_correct += len(target_lab_chunks)

            except Exception:
                pass

        #outF.close()

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    print("precision:")
    print(p)
    print("recall:")
    print(r)
    print("f1 score:")
    print(f1)