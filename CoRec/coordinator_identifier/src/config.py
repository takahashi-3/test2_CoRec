import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 5
BASE_MODEL_PATH = "google-bert/bert-base-uncased"
MODEL_PATH = "saved_models/model_allaug.bin"
TRAINING_FILE = "../data/wsj/wsj_train_allaug.csv"
DEVELOPMENT_FILE = "../data/wsj/wsj_dev_allaug.csv"
TEST_FILE = "../data/wsj//wsj_test_all.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)
