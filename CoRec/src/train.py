import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel


def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    #df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

    enc_ctag = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

    df.loc[:, "c-Tag"] = enc_ctag.fit_transform(df["c-Tag"])
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence #")["Text"].apply(list).values
    ctag = df.groupby("Sentence #")["c-Tag"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values

    return sentences, ctag, tag, enc_ctag, enc_tag


if __name__ == "__main__":
    train_sentences, train_ctag, train_tag, enc_train_ctag, enc_train_tag = process_data(config.TRAINING_FILE)
    dev_sentences, dev_ctag, dev_tag, enc_dev_ctag, enc_dev_tag = process_data(config.DEVELOPMENT_FILE)

    meta_data = {
        "enc_train_ctag": enc_train_ctag,
        "enc_train_tag": enc_train_tag
    }

    joblib.dump(meta_data, "saved_models/meta_allaug.bin")

    num_ctag = len(list(enc_train_ctag.classes_))
    num_tag = len(list(enc_train_tag.classes_))

    """
    (
        train_sentences,
        dev_sentences,
        train_tag,
        dev_tag
    ) = model_selection.train_test_split(sentences, tag, random_state=42, test_size=0.1)
    """

    train_dataset = dataset.EntityDataset(
        texts=train_sentences, ctags=train_ctag, tags=train_tag
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.EntityDataset(
        texts=dev_sentences, ctags=dev_ctag, tags=dev_tag
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cuda")
    model = EntityModel(num_ctag=num_ctag, num_tag=num_tag)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    # use Adam optimizer
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        valid_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {valid_loss}")
        if epoch == 2:
            torch.save(model.state_dict(), "saved_models/model3_allaug.bin")
        if epoch == 3:
            torch.save(model.state_dict(), "saved_models/model4_allaug.bin")
        if valid_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = valid_loss
