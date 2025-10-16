# test2_CoRec
## 入力データ成形の流れ（デフォルトのDemoであるPubMedには正解データのアノテーションなし）
- __並列関係の出力のみ欲しい(precisionの自動評価などをしない)ならば、以下の作業は不要(入力を "CoRec/coordinator_identifier/data/PubMed/XX/YY_c.csv" と同様の形式にする)__
   - Penn Tree の生データ?(.mrg)を "CoRec/data/data_generator.py" に入力
      - precision などの算出における正解ラベルをここで付与する(形式は "CoRec/coordinator_identifier/data/PubMed/XX/YY_c.csv" と同様)
      - Tag: coordinationを表すラベル(coordination である単語には "C" が割り当てられる)

## coordinatorの特定(CoRec/coordinator_identifier/src/predict_PubMed.py)
- 入力csv について、各単語が coordinator(and, betweenなど)であるか推定
- coordinator である単語は、出力の c-Tag に "c-C" が付与され、入力の同単語に付与されている Tag と比較を行い、precision 等が計算される

## 並列関係の特定における正解ラベルの付与
- "CoRec/coordinator_identifier/prediction/XX/YY_c_pred.csv" にある上記の出力ファイル群は、並列関係の特定における正解ラベル "Tag" が付与されていない
- ~"CoRec/data/add_features.py" を実行すると, 上記の出力ファイルに正解ラベル "Tag" が付与される?(検証中)~
- 新規に作成する必要がありそう

## 並列関係の特定(CoRec/src/predict_PubMed.py)
- "coordinatorの特定" の出力にある c-Tag をもとにして、Tag を推定
- 入力の Tag と出力の Tag を比較して、precision 等が計算される

## 注意点
### coordinator_identifier, conjunct_boundary_detector のmodelは入っていません
Before playing, please download our models (trained on Penn TreeBank WSJ) from https://iowastate-my.sharepoint.com/:f:/g/personal/qingwang_iastate_edu/EglvnZ9wv6dPuX2WE5dINJgBn7fJmCa8c48in3Gc6fYqqg?e=6ocdK8, and save them into the corresponding "saved_models" empty folders.

- Specifically, Coordinator Identifier (meta_c.bin & model_c.bin) --> coordinator_identifier/src/saved_models/ Conjunct
- Boundary Detector (meta_2.bin & model4_2.bin) --> src/saved_models/

### pip install に関する注意事項
1. "requirement.txt" を install する前に以下を実行
    - pip install torchaudio==2.7.0+cu118 --index-url https://download.pytorch.org/whl/cu118
    - pip install torch==2.8.0+cu126 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
2. pip install -r requirement.txt
