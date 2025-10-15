
手順１："CoRec/coordinator_identifier/src/" に入り、 "./predict_PubMed.py" を実行
　入力対象となっているのが、"CoRec/coordinator_identifier/data/PubMed/[フォルダ名]/[ファイル名]"の各ファイル(プログラム内に記述されている)
  - [フォルダ名]: "CoRec/PubMed_abstracts/[フォルダ名].txt" に このディレクトリ内に存在する各ファイルの"Text"を平文にしたものを記入する必要がある（各センテンスは "." で区切る）。
                 *ただプログラムでは、このファイル内のセンテンスの数を参照しているだけなので、Google Drive にアップロードされているような形式にするのでも良いのかもしれない
  - [ファイル名]: 入力となるファイル。"Tag" はそのままでもいい(デモに限り) ("Sentence #" と "Offset" は役割が不明)

　出力は、入力の各単語が coordinator であるか判定したフラグ? (coordinatorである場合 "C", それ以外 "0")
　また、precisionなどは 入力の "Tag" と出力のフラグをもとに算出している(入力の"Tag"がアノテーションされていないので、デモの再現では気を使う必要がない)