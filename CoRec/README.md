# CoRec: An Easy Approach for Coordination Recognition

## Paper Link

https://aclanthology.org/2023.emnlp-main.934.pdf

## Resources

Data links can be found in Experiment section of the paper.

Download bert_base_uncased from https://huggingface.co/bert-base-uncased

## Environment SetUp

pip install -r requirements.txt

## Start Playing with a Demo

16 PubMed abstracts are newly included to help readers play with the CoRec model! 

Before playing, please download our models (trained on Penn TreeBank WSJ) from https://iowastate-my.sharepoint.com/:f:/g/personal/qingwang_iastate_edu/EglvnZ9wv6dPuX2WE5dINJgBn7fJmCa8c48in3Gc6fYqqg?e=6ocdK8, and save them into the corresponding "saved_models" empty folders.

Specifically, 
Coordinator Identifier (meta_c.bin & model_c.bin) --> coordinator_identifier/src/saved_models/
Conjunct Boundary Detector (meta_2.bin & model4_2.bin) --> src/saved_models/

Firstly, run predict_PubMed.py in coordinator_identifier/src/ to predict the coordinators.

Then you can run predict_PubMed.py in src/ to predict the conjuncts for each target coordinator.

If you want to generate the final splitted sentences, you can further run sentence_splitter_PubMed.py

