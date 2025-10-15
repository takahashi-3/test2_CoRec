# test2_CoRec
## coordinator_identifier, conjunct_boundary_detector のmodelは入っていません
Before playing, please download our models (trained on Penn TreeBank WSJ) from https://iowastate-my.sharepoint.com/:f:/g/personal/qingwang_iastate_edu/EglvnZ9wv6dPuX2WE5dINJgBn7fJmCa8c48in3Gc6fYqqg?e=6ocdK8, and save them into the corresponding "saved_models" empty folders.

- Specifically, Coordinator Identifier (meta_c.bin & model_c.bin) --> coordinator_identifier/src/saved_models/ Conjunct
- Boundary Detector (meta_2.bin & model4_2.bin) --> src/saved_models/

## 注意点
- 1. "requirement.txt" を install する前に以下を実行
  - pip install torch==2.8.0+cu126 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
  - pip install torchaudio==2.7.0+cu118 --index-url https://download.pytorch.org/whl/cu118
- 2. pip install requirement.txt
