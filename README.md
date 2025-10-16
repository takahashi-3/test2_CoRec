# test2_CoRec
## 入力データ成形の流れ
1. Penn Treeの生データ?(.mrg)を "CoRec/data/data_generator.py" に入力
2. その後、"CoRec/data/add_features.py" を実行すると, 以降の "prediction_PubMed.py" 用の入力データが成形される 

## 注意点
### coordinator_identifier, conjunct_boundary_detector のmodelは入っていません
Before playing, please download our models (trained on Penn TreeBank WSJ) from https://iowastate-my.sharepoint.com/:f:/g/personal/qingwang_iastate_edu/EglvnZ9wv6dPuX2WE5dINJgBn7fJmCa8c48in3Gc6fYqqg?e=6ocdK8, and save them into the corresponding "saved_models" empty folders.

- Specifically, Coordinator Identifier (meta_c.bin & model_c.bin) --> coordinator_identifier/src/saved_models/ Conjunct
- Boundary Detector (meta_2.bin & model4_2.bin) --> src/saved_models/

### pip install に関する注意事項
1. "requirement.txt" を install する前に以下を実行
- pip install torchaudio==2.7.0+cu118 --index-url https://download.pytorch.org/whl/cu118
- pip install torch==2.8.0+cu126 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
2. pip install requirement.txt
