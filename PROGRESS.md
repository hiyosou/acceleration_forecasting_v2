# PROGRESS.md — acceleration_forecasting_v2 実装ログ

SPEC.md の実装順序案(成果物5)に沿って、ステップごとの実装記録を追記していくログです。
新しい記録は末尾に追記します(上書きしません)。

---

## [2026-09-24] 初回実行前チェック

- 実装ファイル: なし(チェックのみ)
- テスト結果: 実行なし
- レビュー懸念点:
  - **Gitリポジトリが機能していない**。`my_project/.git` ディレクトリは存在するが中身が完全に空(`HEAD`・`config`・`objects`・`refs`が一切ない)。Git Bash・PowerShell両方の`git`コマンドで`git status`が`fatal: not a git repository`を返すことを確認した。会話冒頭のシステム情報では「Is a git repository: true」「ブランチ: main」「clean」と表示されていたが、実際のディスク上の状態と food一致しない。
  - タスク指示にある「該当ステップの変更のみを対象にコミットする」を実行できる状態ではない。
- 入出力の具体例: なし
- 判定: **STOP**
- 理由: 下記「STOPの詳細」を参照。

### STOPの詳細(平易な説明)

これからSPEC.mdに沿ってコードを1ステップずつ書き、書いたらすぐ「そのステップだけ」をコミットする、という進め方をする予定でした。ところが土台となるはずのGit管理(変更履歴を記録する仕組み)が、`my_project`フォルダの中で実質的に機能していません。`.git`という管理用のフォルダ自体はありますが、中身が空っぽで、本来入っているはずの管理情報が何も入っていない状態です。

このままコードを書き進めても、「コミットする」という指示の部分だけがずっと失敗し続けてしまいます。また、空っぽの`.git`フォルダを見つけた時点で自己判断で`git init`(新規に作り直す)を実行することもできます。ですが、なぜ空になっているのか原因が分からないまま(本来あったはずの履歴が何かの理由で消えてしまった可能性もゼロではない)、勝手に作り直してよいかは私には判断できないため、先に確認させてください。

### 前提情報のうち、テンプレートで空欄になっていた項目について(こちらで調査し埋めました)

- SPEC.mdのパス: `acceleration_forecasting_v2/SPEC.md`(作成済み)
- 参照可能な既存コード: `acceleration_forecasting_12m/`、`acceleration_retrieval/`(いずれも`my_project`直下)
- 新DBのスキーマ・接続情報: SPEC.md 成果物2.1の表、および元となる`acceleration_forecasting_12m/src/acceleration_forecasting_12m/retrieval/database.py`のSCHEMA定義
- ダミーデータ/サンプルデータの場所:
  - 生データ一式: `D:\railwaydata\方向別振動データ_位置補正` 以下(位置補正済み・0.2mリサンプリング済み・100m最大値集計済み、の各段階のデータが実際に存在)
  - 比較検証用の参照出力: `acceleration_retrieval\artifacts_dataset_split\`(旧パイプラインで実際に構築済みの`vector_database.sqlite`・`development_trends.csv`等が既に存在し、新旧比較テストにそのまま使える)

---

## [2026-09-24] Gitリポジトリのセットアップ

- 実装ファイル: なし(リポジトリ管理のみ)
- テスト結果: 実行なし
- レビュー懸念点: なし
- 入出力の具体例: なし
- 判定: GO
- 詳細:
  - `my_project`直下で`git init -b main`を実行し、空だった`.git`を正常化した(過去のコミット履歴は元々存在しなかったことを確認済み)。
  - コミット者情報(`user.name`/`user.email`)はユーザー本人に設定してもらった(hiyosou / csso23081@g.nihon-u.ac.jp)。
  - 初回コミットは`acceleration_forecasting_v2/`のみを対象とした(`my_project`直下の他のサブプロジェクトは一旦未追跡のまま。追跡範囲は別途相談)。
  - コミット: `535be1d acceleration_forecasting_v2: 初期スキャフォールド作成`(15ファイル、SPEC.md 実装順序案ステップ1に対応)。

---

## [2026-09-24] ステップ2: retrievalパイプラインのフォーク・移植

- 実装ファイル:
  - `src/acceleration_forecasting_v2/retrieval/csv_io.py`(生CSV読み込み、ルート共通モジュールから複製)
  - `src/acceleration_forecasting_v2/retrieval/constants.py`
  - `src/acceleration_forecasting_v2/retrieval/trends.py`(TrendCatalog、SPEC 1.2の打ち切りマスク仕様を実装)
  - `src/acceleration_forecasting_v2/retrieval/waveforms.py`(波形抽出・manifest/trend_catalog書き出し)
  - `src/acceleration_forecasting_v2/retrieval/splitting.py`(dataset_id単位2段階分割)
  - `src/acceleration_forecasting_v2/retrieval/model.py`(WaveformAutoencoder)
  - `src/acceleration_forecasting_v2/retrieval/database.py`(SQLiteスキーマ、SPEC 2.1に対応)
  - `src/acceleration_forecasting_v2/retrieval/autoencoder_training.py`
  - `src/acceleration_forecasting_v2/retrieval/search.py`(GuideIndex、旧forecasting_12mからほぼ無変更で移植)
  - `src/acceleration_forecasting_v2/retrieval/pipeline.py`(上記を束ねる`build_retrieval_database`)
  - `tests/test_trend_catalog.py`, `test_waveform_extraction.py`, `test_splitting.py`,
    `test_database_schema.py`, `test_guide_search.py`, `test_extraction_pipeline.py`
  - `pyproject.toml`(`--basetemp`追加、Windows環境の一時フォルダ権限問題を回避)
- テスト結果: **32件中32件パス**
- レビュー懸念点:
  - (修正済み)`test_waveform_record_requires_matching_trend_id`が`pytest.raises(Exception)`という
    広すぎる例外捕捉になっていたため、`sqlite3.IntegrityError`に限定した。
  - (修正済み)CSV抽出処理(`extract_manifest_and_trends`)のend-to-end統合テストが当初存在せず、
    個々の部品(波形抽出・TrendCatalog)は正しくても繋げたときの正しさが未検証だったため追加した。
  - (残存・許容範囲内)`pipeline.build_retrieval_database`(Autoencoder学習を含む一括実行関数)自体は
    torch学習を伴うため今回はテストしていない。SPEC.md 実装順序案の「end-to-end確認」フェーズで
    実データの一部を使って検証する計画。
  - (残存・許容範囲内)新旧の実出力を突き合わせる文字通りの比較テスト(SPEC.md 4の
    `test_new_trend_catalog_matches_old_for_sample_dates`)は実施していない。旧`acceleration_retrieval`が
    `my_project`直下からの実行を前提にした素のスクリプト群で、`uv`管理の独立パッケージである本リポジトリの
    テストから安全に読み込む方法がなかったため。代わりに、ロジックを移植する際に読み込んだ旧コードの
    仕様を手計算可能な小さな入力で単体テストとして固定する方針にした(結果として意図は同じ:
    「移植したロジックが期待通りに動くこと」を保証する)。
- 入出力の具体例: 本文の「なぜSPEC.mdの要件を満たせていると言えるか」節を参照(会話ログに記載、
  補修跨ぎ打ち切りの例とリーク防止の例)。
- 判定: GO

---

## [2026-09-24] ステップ3: マスク生成・履歴窓構築の実装

- 実装ファイル:
  - `src/acceleration_forecasting_v2/datasets/history.py`(`build_history_window`)
  - `tests/test_history_window.py`
- 補足: SPEC.md 成果物3の`build_future_target`相当は、ステップ2で`TrendCatalog.build_trend`
  として既に実装済み(SPEC 2.1が「future_values/future_maskは確定済みの状態でDBに保存される」
  という前提を明示していたため、retrieval層に統合することがSPEC通りと判断)。本ステップでは
  残る「6か月統一履歴」(`build_history_window`)のみを実装した。
- テスト結果: **40件中40件パス**(既存32件 + 新規8件)
- レビュー懸念点:
  - (修正済み)「今月の行が存在するが値がNaN」を「行が丸ごと存在しない」(raise対象)と
    区別するテストが当初なく、コードパスとして存在するのに未検証だったため追加した。
  - (修正済み)探索範囲内に複数候補があるときのタイブレーク(より近い方を優先)のテストを
    追加する際、自分の日付差の暗算を間違えて一度テストが失敗した(2023-03-28と対象月
    2023-04-01の差を27日と誤算 → 実際は4日)。テストの想定値を修正して再実行し、全て
    パスすることを確認した。
- 入出力の具体例:
  ```
  入力: 2023年2月〜7月まで毎月15日に実測データがある区間、anchor=2023年7月15日
  出力: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0](古い月→今月の順)、マスクは全て1

  入力: 上記のうち2023年3月分だけ欠測
  出力: [1.0, NaN, 3.0, 4.0, 5.0, 6.0]、マスクは [1,0,1,1,1,1]
  ```
- 判定: GO

---

## [2026-09-24] ステップ4: anchor選定ロジックの実装

- 実装ファイル:
  - `src/acceleration_forecasting_v2/datasets/anchors.py`(`AnchorCandidate`, `select_anchors_for_segment`, `compute_segment_weights`)
  - `tests/test_anchor_selection.py`
- テスト結果: **47件中47件パス**(既存40件 + 新規7件)
- レビュー懸念点:
  - (修正済み・自分の計算ミス)テスト作成時、「履歴条件を満たすのは4か月目から」と
    手計算していたが、実装を実行すると3か月目から満たすことが判明した。原因は
    「今月自体も履歴の有効月数1か月分としてカウントされる」ことを計算に入れ忘れていた
    ため(実装のバグではない)。実際の出力を確認した上でテストの期待値
    (10件/6件/18件、開始月)を全て修正した。
  - (修正済み)future_values/future_maskの長さが`forecast_months`と食い違う場合に
    明確に例外を送出する防御的チェックを実装していたが、それを確認するテストが
    当初無かったため追加した。
- 入出力の具体例:
  ```
  入力: 2022年1月〜2023年8月まで毎月データがある1区間、min_history_months=3、
        min_target_months=8(12か月中8か月以上)
  出力: 2022年3月〜2022年12月の10件のanchorが選ばれる
        (旧実装なら「一番条件の良い1件」だけが選ばれていた)
  ```
- 判定: GO

---

## [2026-09-24] ステップ5: セグメント重み計算の実装(ステップ4で完了済みを確認)

- 実装ファイル: なし(新規実装なし)
- テスト結果: 実行なし(ステップ4の47件がそのまま該当)
- レビュー懸念点: なし
- 詳細: SPEC.md 実装順序案ステップ5(セグメント重み計算)は、ステップ4で
  `datasets/anchors.py`に`compute_segment_weights`として既に実装・テスト済み
  だったため、本ステップでの追加実装は無い。重みの「算出」はここで完了しており、
  各anchorのレコードへの「付与・保存」はステップ6(データセット構築パイプライン)の
  責務として引き継ぐ。
- 判定: GO

---

## [2026-09-24] ステップ6: データセット構築パイプライン一式(1回目STOP → 決定 → 実装)

### STOP: target_modeがSPEC.mdで未決定だった

- 判定: STOP
- 理由: `datasets/build.py`のターゲット構築ロジックを書こうとしたところ、
  `target_mode`(residual差分予測 vs absolute実測値直接予測)がSPEC.mdで一度も
  明示的に決定されていないことに気付いた。1.3節でRMAの固定値を決めた際に根拠にした
  実測データ(`artifacts_prediction_parameterization`等)が、確認し直すと
  **全てtarget_mode="absolute"**で行われていたため、単純に「元々の設計思想は
  residual」と決め打ちすると、1.3節の根拠と矛盾する組み合わせになってしまう。
- ユーザーへの確認: 「ノイズ予測という拡散モデルの仕組み自体は変わるか」という質問を
  受け、prediction_type(何を予測するか: ノイズ/v/x0)とtarget_mode(そもそも
  "正解データ"をどう定義するか: 差分/実測値)が独立な軸であることを図解して回答。
- 決定: **absolute**を採用(SPEC.md 1.6節として追記)。

### 実装

- 実装ファイル:
  - `src/acceleration_forecasting_v2/datasets/baseline.py`(`softmax_guide_baseline`)
  - `src/acceleration_forecasting_v2/datasets/normalization.py`(`Normalization`)
  - `src/acceleration_forecasting_v2/datasets/build.py`(`build_raw_split`, `fit_condition_and_target_normalization`)
  - `tests/test_dataset_build.py`
  - `SPEC.md`(1.6節追加、guide_softmax_weightsの形状訂正、target_values/guide_baselinesの説明訂正)
- テスト結果: **52件中52件パス**(既存47件 + 新規5件)
- レビュー懸念点:
  - (修正済み・SPEC.mdの誤記)`guide_softmax_weights`の形状を当初`(N,3)`と記載していたが、
    実装元のコード(`softmax_guide_baseline`は月ごとに有効なguideでsoftmaxを取り直す)を
    確認すると正しくは`(N,3,12)`だった。SPEC.md成果物2.1を訂正した。
  - (修正済み)`test_build_raw_split_target_values_are_absolute_not_residual`が
    最初失敗した。原因はテストフィクスチャの不備: guideの「現在値」をターゲットと
    大きく異なる値(99.0)にしていたため、`max_current_difference`フィルタで
    guide自体が検索結果から除外されてしまい、「guideありの状態でabsoluteを検証する」
    という意図を満たせていなかった。guideの現在値と将来値を独立に指定できるよう
    テストヘルパーを修正し、現在値は近く・将来値だけ大きく異なる、という現実的な
    フィクスチャに直して再実行した。
  - (修正済み)最初の形状検証テストも、同じ理由でguideが実際には見つからない状態で
    「通ってしまっていた」ため、guideが確実に見つかる条件に修正し、
    `guide_count == 1`のassertionを追加した。
- 入出力の具体例:
  ```
  入力: ある区間・ある月(anchor)。似た過去の波形が1件見つかり、その将来推移は99で一定
  出力: target_values(モデルが学習する正解) = anchor自身の実測値(例: 5)
        guide_baselines(評価用の参考値、学習には使わない) = 99(guideの値)
  ```
- 判定: GO

---

## [2026-09-24] 専用uv環境の構築

- 実装ファイル: なし(環境構築のみ)
- 詳細: `uv sync --extra dev`を実行し、`acceleration_forecasting_v2/.venv`に
  torch==2.7.1+cu118を含む専用環境を構築した(`torch.cuda.is_available() == True`を確認)。
  ステップ1〜6は旧`acceleration_forecasting_12m/.venv-gpu`を一時的に借りてテストしていたが、
  以降は本プロジェクト専用の環境でテストを実行する。
- 判定: GO

---

## [2026-09-24] ステップ7: ForecastDatasetV2の実装

- 実装ファイル:
  - `src/acceleration_forecasting_v2/datasets/torch_dataset.py`(`ForecastDatasetV2`)
  - `tests/test_torch_dataset.py`
- テスト結果: **60件中60件パス**(既存52件 + 新規8件)。専用uv環境での初回実行。
- レビュー懸念点: 特筆すべき問題は見つからなかった。旧`ForecastDataset`
  ([acceleration_forecasting_12m/.../datasets/torch_dataset.py])のNaN→0変換パターン
  (`_finite_zero`)・正規化ロジックを踏襲しつつ、"current"/"history"の統合(SPEC 1.1)、
  "segment_weight"の追加(SPEC 1.1)、target_mode=absolute専用化(SPEC 1.6、
  residualモードのguide_baseline加算分岐を削除)を反映した。
- 入出力の具体例:
  ```
  入力: 正規化統計(mean=2.0, std=1.0)、保存された生の予測正解値=[3.0, ...]
  出力: モデルに渡される正規化後の値 = (3.0 - 2.0) / 1.0 = 1.0

  入力: モデルの出力(正規化空間で0.0)を物理単位に戻す
  出力: 0.0 × 1.0 + 2.0 = 2.0(guideの値を足したりはしない — absoluteモードの決定通り)
  ```
- 判定: GO

---

## [2026-09-24] ステップ8: ReferenceModulatedUNetV2(RMAモデル本体)の実装

- 実装ファイル:
  - `src/acceleration_forecasting_v2/models/reference_modulated_unet.py`
  - `tests/test_reference_modulated_unet.py`
- テスト結果: **73件中73件パス**(既存60件 + 新規13件)
- レビュー懸念点:
  - (修正済み)デフォルト設定(similarity無効・month_aligned・ratd_condition_reconstruction)
    のテストしかなく、クラスが技術的にサポートしている非デフォルトの分岐
    (similarity_bias有効・globalアテンション・residual_delta融合)が一度も
    実行されていなかった。forward+backwardが両方通ることを確認するテストを追加した。
- 入出力の具体例:
  ```
  入力: バッチサイズ4、ノイズ付き12か月系列(4,12)、履歴(4,6)、guide(4,3,12)
  出力: ノイズの推定値(4,12) ← 入力と同じ形
  パラメータ数: 約488万(旧リポジトリの同構成の実測値と同オーダーであることを確認)
  ```
- 判定: GO

---

## [2026-09-24] ステップ9: セグメント単位重み付き損失の実装(+ステップ11 DDIMサンプリングを前倒し実装)

- 実装ファイル:
  - `src/acceleration_forecasting_v2/diffusion/process.py`(`DiffusionProcess`: コサインスケジュール、
    `per_record_loss`、`model_output_to_x0_epsilon`、`ddim`)
  - `src/acceleration_forecasting_v2/training/loss.py`(`SegmentWeightedMaskedLoss`)
  - `tests/test_diffusion_process.py`, `tests/test_segment_weighted_loss.py`
- 補足: SPEC.md成果物3のインターフェース定義には`DiffusionProcess`を明記していなかったが、
  損失関数(`per_record_loss`の入力元)として必須の依存だったため本ステップで実装した。
  あわせてDDIMサンプリング(実装順序案ステップ11の中核)も`DiffusionProcess.ddim`として
  同時に実装済みとなった(旧`DiffusionProcess`が同じクラスに両方を持つ構造だったため、
  分離するより自然だった)。ステップ11では残る「predict()相当の呼び出し口」のみを行う。
  Min-SNR関連のコードは削除した(SPEC 1.5/1.6の決定により本プロジェクトでは使わないため、
  不要な複雑さを持ち込まない)。
- テスト結果: **89件中89件パス**(既存73件 + 新規16件)
- レビュー懸念点:
  - (修正済み)DDIMの`normalized_clip`(クリップ処理)を検証するテストが、実際には
    「NaNが出ないこと」しか確認しておらず、クリップの効果自体を検証できていなかった
    (弱いテスト)。`sampling_steps=1`・`eta=0`のときは最終出力が
    「クランプ後のpredicted_clean」に厳密に一致するという性質を使い、固定出力の
    ダミーモデルで具体的な数値(-1.0ちょうど)を検算するテストに書き直した。
    クリップなしの場合は範囲を大きく超えることも対比で確認した。
- 入出力の具体例:
  ```
  入力: セグメントAに10件(誤差0.20)、セグメントBに1件(誤差0.50)
  出力: 重み付き平均 = 0.35(SPEC.md 1.1の手計算例と一致)
  ```
- 判定: GO

---

## [2026-09-25] Git/GitHub整備(ステップ9とステップ10の間)

- 内容: `acceleration_forecasting_v2/`を`my_project`から切り離して独立したGitリポジトリ化
  (`git subtree split`で10コミットの履歴を保持)。`my_project`側は追跡から外し、
  `.gitignore`に追加。`gh`(GitHub CLI)をwingetでインストールし、
  https://github.com/hiyosou/acceleration_forecasting_v2 をPublicで作成してpush済み。
- 注意: 公開前の確認で、コミット作者情報に大学メールアドレスが含まれる旨を説明した上で、
  ユーザーが「そのままPublicでpush」を明示的に選択した。
- 以降のコミットは自動ではpushしない(pushは都合のよいタイミングでユーザー判断)。
- 判定: GO

---

## [2026-09-25] ステップ10: 学習ループの実装

- 実装ファイル:
  - `src/acceleration_forecasting_v2/training/train.py`(`train`, `validation_loss`, `dataset_build_id`)
  - `src/acceleration_forecasting_v2/training/ema.py`
  - `src/acceleration_forecasting_v2/datasets/build.py`に`write_split`/`write_dataset`を追加
  - `tests/test_training.py`
- 補足(ステップ6の抜けの補完): ステップ6は`build_raw_split`(配列の組み立て)までで、
  `ForecastDatasetV2`が読む`.npy`ディレクトリを**書き出す処理が存在しなかった**。
  学習ループの前提となるため`write_dataset`(正規化統計はmodel_trainのみからfit)を追加した。
- テスト結果: **102件中102件パス**(既存89件 + 新規13件)
- レビュー懸念点:
  - (対応済み)validation lossは、SPEC 1.1の決定どおり`segment_weight`を適用し、
    さらに「バッチごとの重み付き平均の平均」ではなく検証セット全体の Σ(w·loss)/Σw で
    計算する(バッチ構成に結果が依存しない)。`test_validation_loss_is_segment_weighted_over_whole_set_not_per_batch`
    でSPEC 1.1の手計算例(0.35 ≠ 単純平均0.227)を、バッチサイズ3に分割した状態で検証。
  - (対応済み)CPUテストだけではGPU固有経路(bf16 autocast・GradScaler)が未実行のため、
    `gpu`マーカー付きの実行テストを追加し、実GPUで通ることを確認した。
  - (確認済み)loss減少テストは合成データで train 0.57→0.14 と十分な余裕があり、偶然ではない。
  - (残存)勾配計算に`segment_weight`が効いていること自体を直接検証するテストは無い
    (重み付き平均の数式は`test_segment_weighted_loss.py`で検証済みで、trainはそれを呼ぶだけ)。
  - (残存)実データでの学習は未実施(ステップ14で実施)。
- 入出力の具体例:
  ```
  入力: 合成データ(履歴平均+月ごとの傾きが正解)64件、15エポック、prediction_type=epsilon
  出力: train_loss 0.95→0.14、validation_loss 1.33→0.87(いずれも単調に近く減少)
        best_model.pt / last_model.pt / training_history.csv / resolved_config.json を出力
  ```
- 判定: GO

---

## [2026-09-25] ステップ11: DDIM推論の呼び出し口(predict)の実装

- 実装ファイル: `src/acceleration_forecasting_v2/inference/predict.py`(`predict`, `load_process`)、`tests/test_predict.py`
- 補足: DDIMサンプリング本体はステップ9で`DiffusionProcess.ddim`として完成済み。本ステップは
  checkpoint復元・1レコードあたり`num_samples`本の生成・物理値への逆変換とクリップ・
  median/p10/p90/stdの算出と保存(`predictions.csv`, `samples.npz`, `prediction_run.json`)。
- テスト結果: **111件中111件パス**(既存102件 + 新規9件)
- レビュー懸念点:
  - (修正済み)`samples.npz`にtrend_idsをobject型で保存していたため`allow_pickle=False`で
    読めず、テストが1件失敗した。固定長のU型で保存するよう修正(pickleに依存しない形式)。
  - (対応済み)推論は正解(target_values/target_masks)を一切読まない(`include_targets=False`)。
    ファイルを削除しても動くことを`test_prediction_does_not_read_target_files`で確認(SPEC 2.3のリーク防止)。
  - (残存)実データ・実モデルでの推論は未実施(ステップ14)。sampling_steps/eta等の最適値の選定処理
    (旧`select_sampling`)は本リポジトリでは実装しない(設定の枝分かれ防止方針)。
- 入出力の具体例:
  ```
  入力: 推論splitの5レコード、num_samples=6、sampling_steps=4
  出力: predictions.csv(5×12=60行、p10<=median<=p90、全て0.1〜6.0の範囲)、
        samples.npz(samples shape=(5,6,12))
  ```
- 判定: GO

---

## [2026-09-25] ステップ12: 評価関数(record-level / segment-level)の実装

- 実装ファイル:
  - `src/acceleration_forecasting_v2/evaluation/metrics.py`(`target_metrics`、旧実装を無変更で移植)
  - `src/acceleration_forecasting_v2/evaluation/aggregate.py`(`evaluate_record_level`, `evaluate_segment_level`,
    `bootstrap_record_level`, `bootstrap_segment_level`, `confidence_interval`)
  - `src/acceleration_forecasting_v2/evaluation/evaluate.py`(`evaluate`, `per_target_frame`, `build_comparison_table`)
  - `tests/test_evaluation.py`
- 仕様どおりの点(SPEC 1.4): segment-levelは「セグメント内平均→セグメント間平均」の2段階平均。
  bootstrapもdataset_id単位の復元抽出で、segment-level側は各反復でセグメント内平均を単純平均する
  (旧実装は集計がanchor重み付きのままだった問題を解消)。record-level側のbootstrapは旧実装と同じ。
- 追加した点: 評価の参考としてguide加重平均ベースラインの誤差(`baseline_MAE`/`baseline_RMSE`)を
  両レベルで集計(SPEC 1.6でguide_baselinesを評価参考値として残すと決めたことに対応)。
  有効な正解月が`min_target_months`(既定8)未満のレコードは評価から除外(学習側のeligibilityと整合)。
- 実装しなかった点: 画像出力、複数実験ツリーを横断する比較処理(成果物乱立の防止)。
  モデル間比較は`build_comparison_table`にsummaryを渡す形に限定。
- テスト結果: **130件中130件パス**(既存111件 + 新規19件)
- レビュー懸念点:
  - (修正済み)テスト作成時、coverageの手計算(3/4=0.75)を一度0.5と誤記し、訂正コメントを
    残したままの状態だったため、実行前に整理して正しい期待値に直した(実装のバグではない)。
  - (確認済み)SPEC 1.4の手計算例(A:10件MAE0.20、B:1件MAE0.50)で record-level=0.227、
    segment-level=0.35 を集計関数単体とevaluate()全体の両方で検証。bootstrapは「取りうる値の集合」を
    理論値で検証(segment-levelは{0.20,0.35,0.50}のみ、record-levelは件数重みの値)。
  - (残存)指標のうち`correlation`は12点中の相関でNaNになりうるため、集計時はNaNを除外して平均している
    (旧実装と同じ挙動)。除外件数は出力していない。
- 入出力の具体例:
  ```
  入力: 推論5件(segA 4件は予測が+0.1ずれ、segB 1件は+0.5ずれ)
  出力: record-level MAE = 0.18、segment-level MAE = 0.30(セグメントに公平)
  ```
- 判定: GO

---

## [2026-09-25] ステップ13: リーク検証(+ステップ6の抜けだった一括構築処理の補完)

- 発見した抜け(ステップ6): 「retrievalの成果物から3splitのデータセットを一括で作る処理」が存在せず、
  ステップ14(実データ実行)に必須だった。あわせて、リーク検証のために各anchorのguideの出所
  (trend_id/dataset_id/model_split)がmetadataに残っている必要があった。
- 実装ファイル:
  - `src/acceleration_forecasting_v2/datasets/prepare.py`(`prepare_datasets`: anchor選定→guide検索→重み→正規化→書き出しを、
    SPEC 2.3の規則どおりに結合。guide許可split: train→{train}, validation→{train}, inference→{train,validation}。
    inferenceの埋め込みはDBに無いためAutoencoderでその場計算)
  - `src/acceleration_forecasting_v2/datasets/verify.py`(`verify_dataset_leakage`: SPEC 2.3の検査項目を
    実データにも使える関数化)
  - `src/acceleration_forecasting_v2/datasets/build.py`(metadataにguide_trend_ids/guide_dataset_ids/guide_model_splitsを追加)
  - `src/acceleration_forecasting_v2/retrieval/pipeline.py`(`write_vector_database`を関数として切り出し。挙動は不変)
  - `tests/synthetic_artifacts.py`(合成retrieval成果物)、`tests/test_split_leakage.py`、`tests/test_end_to_end.py`
- テスト結果: **154件中154件パス**(既存130件 + 新規24件)
- SPEC 4のリーク検証項目の充足状況:
  - dataset_idが1つのsplitにのみ存在 ✔ / trainのguideはtrainのみ ✔ / validationのguideはtrainのみ ✔ /
    inferenceのguideに他inferenceなし ✔ / 同一dataset_id除外 ✔ / 正規化統計はmodel_trainのみ ✔ /
    Autoencoder学習はinference波形に触れない(`MemmapWaveformDataset`に渡るindexを監視し、
    inference波形に+1000を足して統計への混入がないことも確認) ✔
- レビュー懸念点:
  - (修正済み)`test_inference_targets_are_kept_out_of_model_inputs`が「shapeが非ゼロ」しか見ておらず
    実質何も検証していない見せかけのテストだった。削除し、代わりに`verify_dataset_leakage`を実装した上で、
    **わざとリークを混ぜたデータ(禁止splitのguide・同一dataset_idのguide・dataset_idの重複・誤った重み/正規化)を
    検出できること**をテストした(検査自体の検出力を確認)。
  - (確認済み)推論が正解を読まないことはステップ11のテストで確認済み。
  - (残存)合成データでの検証。「実データ規模での実行」はステップ14で`verify_dataset_leakage`を実データに適用する。
  - (残存)`guide_available_date`(近傍guideの時間制約)は`prepare`経由の統合テストでは実質検証されていない
    (合成データは各セグメントの区間を1000m離しており近傍にならない)。単体では`test_guide_search.py`で検証済み。
- 入出力の具体例:
  ```
  入力: 12セグメント×22か月の合成成果物(train 8 / validation 1 / inference 3セグメント)
  出力: 3split全てが構築され、trainのguideは全てmodel_train由来、inferenceのguideはtrain/validation由来のみ。
        verify_dataset_leakage → 違反0件。学習→推論→評価まで通しで実行できる(epsilon/v_prediction両方)。
  ```
- 判定: GO

---

## [2026-09-25] ステップ14 着手前の実データ確認: 新旧retrieval差分検証(実データ) → STOP

### 新旧retrieval出力の実データ突き合わせ(SPEC 4「新旧retrieval出力差分の検証」)
- 方法: 実データ10ファイル(`D:\railwaydata\...0.2m\D\2023\`の先頭10本)に対し、本リポジトリの
  `extract_manifest_and_trends`の出力を、旧`acceleration_retrieval/artifacts`の同一ソースファイルの行と比較。
- 結果(全て一致):
  - record_id の集合が完全一致(新1,028件=旧1,028件、片側のみ0件)
  - waveform_sha256 / trend_id / bin_start_m / measurement_date / mean_velocity_kmh が1,028/1,028件で一致
  - trend_catalog: 401/401件がtrend_idで対応し、未来値(先頭12か月)・mask・current値・打ち切り日が100%一致
    (旧は18か月、新は12か月保持のため先頭12か月で比較)
- 意味: 移植したretrievalの抽出ロジックは旧実装とビット単位で同一の結果を返す。
  ステップ2で「単体テストで代替」としていた新旧差分検証の残存懸念が、実データで解消した。

### 実データの規模と所要時間の見積もり(STOPの理由)
- 実データ: 生CSV 5,954本・126GB。旧artifactsは420,686波形(waveforms.bin 841MB)・113,802トレンド。
- 実測: 抽出は3.4秒/ファイル → **全件の抽出し直しは約5.6時間**(CPU)。
- SPEC.mdには「実データをどう用意するか」の指定がなく、複数の方針がありうる:
  (1)全件再抽出、(2)旧artifactsの`waveforms.bin`/manifestを入力スナップショットとして取り込み、
  トレンドは12か月で再構築しsplit・AE・DBは新リポジトリで作る、(3)一部のみで縮小実行。
  加えて、実データでの学習(200エポック×2構成)もGPUで数時間規模になる見込みのため、実行方針の確認が必要。
- 判定: **STOP**(ユーザーの判断待ち)

## [2026-09-25] ステップ14 準備: 抽出結果の取り込みと最小CLI(ユーザー判断: 旧waveforms.binを入力スナップショットとして取り込む)

- STOPへの対応: ユーザーが「旧artifactsのwaveforms.binを入力スナップショットとして取り込む」を選択。
- 実装ファイル:
  - `src/acceleration_forecasting_v2/retrieval/snapshot.py`(`import_extraction_snapshot`)
  - `src/acceleration_forecasting_v2/retrieval/pipeline.py`(`build_retrieval_from_extraction`を切り出し。
    生CSV経路・取り込み経路の両方が共通で使う)
  - `src/acceleration_forecasting_v2/cli.py`(パイプライン各段階のみの最小CLI。実験ごとにサブコマンドを増やさない)
  - `tests/test_snapshot.py`, `tests/test_cli.py`
- 取り込みの仕様: 旧ファイルは読むだけでコード上は依存しない。トレンドは12か月で再構築、manifestにdataset_idを付与、
  旧の日付単位splitは捨てる。対応するトレンドが無い行は除外し波形を詰め直す(元と同一の波形を保つことをテスト)。
  waveforms.binのサイズ不正・必須列欠落・waveform_index重複・対応トレンド0件は明示的にエラー。
- テスト結果: **165件中165件パス**(既存154件 + 新規11件)。取り込み→DB構築→3split構築→リーク検査、
  およびCLIで全段階(epsilon/v_prediction両方の学習→推論→評価→比較表)を通しで実行するテストを含む。
- レビュー懸念点:
  - (残存)取り込みのトレンド再構築は実データで約50分かかる(1回のみ、バックグラウンド実行)。
  - (残存)`build_trend`は1件あたり約23msと遅い(旧実装から不変)。今回は1回限りのため最適化しない。
- 判定: GO(実データ実行は継続中)

## [2026-09-25] ステップ14 準備: 実データの実行規模と実行設定の確定(ユーザー判断)

### 実測した見込み(実データ)
- anchor数: train 36,657 / validation 3,301 / inference 19,505(セグメント数 763 / 85 / 213)。旧実装(1セグメント1件で数百件)の約85倍。
- GPU(RTX 2080)実測: RMA学習 0.33秒/ステップ(bs128) → 1エポック約1.2〜1.7分。DDIM(100サンプル×50ステップ)5.3秒/レコード。
  Autoencoderは1エポック数秒(データ読み込み込みで1分前後の見込み)。
- 見込みの結論: 学習200エポックは旧実験の約85倍の更新回数で過剰、推論は全件だと1構成約25時間で非現実的。
### ユーザーが選択した設定
- 学習: **60エポック、patience 10**(旧既定は200エポック)。
- 推論: **セグメントあたり最大10件に間引き、100サンプル×50ステップを維持**。
### 実装
- `datasets/prepare.py`: `inference_max_per_segment`(inferenceを評価用セットとして構築:
  正解が8か月以上ある(=評価に使える)anchorだけを、セグメントごとに日付均等で最大N件。決定的、乱数不使用)。
  `predict`が正解を読まない方針を保つため、選別は`prepare`側で行う。train/validationには影響しない。
- `inference/predict.py`: cudaでは既定でbf16 autocast(学習と同じ精度)。fp32との中央値の差は平均0.0016(合成データ)。
- `cli.py`: `prepare --inference-max-per-segment`
- テスト結果: **170件中170件パス**(既存165件 + 新規5件)
- レビュー懸念点:
  - (残存)間引きによりinferenceのrecord-levelの値は「間引き後の集合」での値になる(segment-levelは影響が小さい)。
  - (残存)bf16の影響は合成データでのみ確認。実モデルでの差は実データ実行時に(必要なら)fp32で再確認する。
- 判定: GO

## [2026-09-25] ステップ14: 実データでのepsilon / v_prediction比較の実行

### 実行内容(全て実データ、CLI経由。コマンドと所要時間)
| 段階 | 所要 | 結果 |
|---|---|---|
| import-snapshot(旧waveforms.bin取り込み、トレンド12か月再構築) | 約50分 | 420,686波形・1,061セグメント・トレンド113,802件、除外0件 |
| build-retrieval(Autoencoder+ベクトルDB) | 約7分 | 42エポックで早期終了、検証loss 約0.0004 |
| prepare(3split構築、inferenceは評価用に間引き) | 約70分 | train 36,657 / validation 3,301 / inference 1,119 anchors(432 / 51 / 116セグメント)、guide取得率100% |
| verify(リーク検査) | 14秒 | 違反0件 |
| epsilon: 学習 / 推論 / 評価 | 1.21h / 1.68h / 4秒 | 40エポックで早期終了(best=epoch30) |
| v_prediction: 学習 / 推論 / 評価 | 1.21h / 1.68h / 4秒 | 40エポックで早期終了(best=epoch31) |
- 設定(ユーザー選択): 60エポック・patience 10、推論は100サンプル×50ステップ・bf16、inferenceはセグメント最大10件に間引き。
- 実行中のwarning/error: なし(ログにTraceback無し。ログ検索で"error"が引っかかった3件は指標名`peak_*_error`)。

### 結果(inference 1,119件・116セグメント、bootstrap 1000回)
| 指標 | epsilon(seg-level) | v_prediction(seg-level) | 備考 |
|---|---|---|---|
| MAE | 0.2131 [0.194, 0.233] | **0.1945** [0.175, 0.214] | guide平均ベースライン 0.3228 |
| RMSE | 0.2594 | **0.2395** | |
| correlation | 0.3006 | 0.3017 | 両者とも低い |
| coverage_p10_p90 | **0.7636** | 0.7389 | 公称0.80 |
| mean_interval_width | 0.639 | 0.510 | vは区間が約20%狭い |
- record-levelとsegment-levelはほぼ一致(MAE 0.2114/0.2131、0.1931/0.1945)。
- セグメント対応ありの差(v−eps)のbootstrap 95%CI: MAE −0.0186 [−0.0268, −0.0101]、RMSE −0.0198 [−0.0289, −0.0115]
  (vが良いセグメントは68〜70%)。coverage −0.0247 [−0.0464, −0.0030]。→ vは点予測が有意に良く、epsilonは区間の較正がわずかに良い。
- どちらもguide平均ベースライン(MAE 0.323)を上回る(epsilon −34%、v −40%)。
- 旧実験で見られた極端な較正悪化(v_predictionのcoverage 0.62)は今回は起きていない(0.74)。
- 解釈上の注意: 各構成1シード。correlation約0.30・隣接月の増減方向一致率約53%(ほぼ偶然水準)と、月ごとの推移形状の再現は弱い。
  MAEの改善は主に「水準」を当てていることによる可能性が高い。x0_prediction・複数シードは未検証(スコープ外)。

## [2026-09-25] 最終レビュー

### 1. SPEC.md全項目チェックリスト
`src/acceleration_forecasting_v2/`配下の位置。PARTIALは仕様どおりだが留意点あり。
| 項目 | 判定 | 場所 |
|---|---|---|
| 1.1 セグメント重み(loss側・validationにも適用、Σ(w·l)/Σw) | PASS | training/loss.py:16、training/train.py:63(validation)・:88(train) |
| 1.2 補修跨ぎ打ち切り・マスク(保存NaN→入力直前で0) | PASS | retrieval/trends.py `build_trend`、datasets/torch_dataset.py:41 |
| 1.3 RMA固定値(month_aligned/ratd/similarity無効/64ch等) | PASS | models/reference_modulated_unet.py:189 |
| 1.4 record/segment-level(2段階平均)・bootstrap | PASS | evaluation/aggregate.py:29,40,46,60、evaluation/evaluate.py:66 |
| 1.5 epsilon/v_prediction並行比較(2回学習、同一メトリクス) | PASS | training/train.py、cli.py、evaluation/evaluate.py:96 |
| 1.6 target_mode=absolute | PASS | datasets/build.py:55(ターゲット=anchor.future_values)、torch_dataset.py |
| 入力形状6か月統一・フラットMLP | PASS | datasets/history.py:16、reference_modulated_unet.py(condition_encoder) |
| anchor: セグメント内全月・min_target 8 | PASS | datasets/anchors.py:51 |
| 出力12か月固定 | PASS | common/constants.py |
| 2段階dataset_id分割(80/20→9/1) | PASS | retrieval/splitting.py:30 |
| retrieval断ち切り・コピーして出発点 | PASS | retrieval/*(旧artifactsは一度きりの入力ファイルとしてsnapshot.py:32が読むのみ、旧コードはimportしない) |
| 2.1 データ契約(各配列のshape/dtype/正規化) | PASS | datasets/build.py:55,135,168(実データ: history(N,6)、guide(N,3,12)、target(N,12)、segment_weights(N,))。SPEC誤記(guide_softmax_weights)は訂正済み |
| 2.2 6か月履歴→12次元→MLP | PASS | reference_modulated_unet.py |
| 2.3 リーク項目1(dataset_idは1つのsplit) | PASS | datasets/verify.py:19、tests/test_split_leakage.py |
| 2.3 項目2(guide許可split) | PASS | datasets/prepare.py `GUIDE_SOURCE_SPLITS` |
| 2.3 項目3(同一dataset_id除外) | PASS | retrieval/search.py |
| 2.3 項目4(近傍guideの時間制約) | **PARTIAL** | retrieval/search.py:78。仕様どおり「近傍のみ」に実装。下記レビュー指摘1参照 |
| 2.3 項目5(正規化はmodel_trainのみ) | PASS | datasets/build.py:168、verify.py |
| 2.3 項目6(Autoencoderはinference波形に触れない) | PASS | retrieval/autoencoder_training.py、tests/test_split_leakage.py |
| 2.3 項目7(inferenceはproduction-ready判定) | PASS | datasets/anchors.py:51 |
| 2.3 項目8(重みはsplit内で完結) | PASS | datasets/prepare.py、verify.py |
| 2.3 項目9(推論の正解を読まない) | PASS | inference/predict.py(include_targets=False、テストで正解ファイル削除でも動作) |
| 3 インターフェース(AnchorCandidate、select_anchors…、SegmentWeightedMaskedLoss、RMA、評価、Dataset) | PASS | 上記各ファイル |
| 3 `build_future_target`(単体関数) | PARTIAL | 独立関数ではなく`TrendCatalog.build_trend`に統合(SPEC 2.1が「DBには確定済みのfuture_values/maskが入る」前提だったため。PROGRESSステップ3に記録済み) |
| 4 受け入れテスト観点 | PASS | 170件。SPEC記載名とは異なるが各観点を網羅。新旧retrieval差分は実データ10ファイル・1,028波形で全件一致を確認 |
| 5 実装順序 ステップ1〜14 | PASS | 本ログの各ステップ |

### 2. 実データでのend-to-end実行(少量ではなく全量で実施)
- 各段階のshape: history_values (N,6) / guide_values,guide_deltas (N,3,12) / guide_similarities (N,3) / target_values (N,12) / segment_weights (N,);
  モデル入出力 (B,12)→(B,12)、パラメータ数 4,876,537。
- 学習曲線: epsilon train 0.488→0.243、validation 0.868→0.241(best epoch30)。v_prediction train 0.723→0.382、validation 1.027→0.382(best epoch31)。
  いずれもvalidationが単調に近く低下し、10エポック改善なしで早期終了(過学習の兆候なし)。
- 評価指標: 上記の結果表のとおり。
- warning/error: なし。

### 3. 第三者(シニアMLエンジニア)視点のコードレビュー
致命的(正しさを損なうバグ): **なし**(テスト170件パス、リーク検査違反0、新旧retrieval差分一致)。

重大(結果の解釈・妥当性に影響):
1. **guide検索の時間制約が「空間的に近い区間」にしか掛かっておらず、未来のデータがguideに使われている**(retrieval/search.py:78。旧実装から継承、SPEC 2.3項目4も同じ仕様)。
   実データで定量化: inferenceのguideの**42.5%が予測起点日より後に計測**され、**75.6%は、起点日時点ではその12か月先の推移がまだ確定していない**。
   (train 46.3%/83.3%、validation 44.5%/80.9%)。dataset_id単位のリーク(train/inference間)ではないが、
   「実運用で予測する時点では手に入らない情報」を使っている。よって報告した精度は、運用時の精度より楽観的な可能性が高い。
   epsilonとv_predictionは同じ条件で比較しているため、両者の相対比較の公平性は保たれる。対処するなら
   「guideのavailable_date < 起点日」を全候補に課す(旧`acceleration_retrieval`のREADMEにある`strict-time`相当)。再実行に約6時間。
2. **各構成1シード、評価は間引き後の1,119件(116セグメント)**。差の有意性は対応ありbootstrapで確認したが、シード間ばらつきは未評価。
3. 月ごとの推移形状の再現は弱い(correlation 約0.30、増減方向一致率 約53%)。MAE改善の多くは水準の一致による可能性。

軽微:
4. 構築が遅い: `select_anchors_for_segment`が行ごとのiterrows、`build_trend`が1件約23ms(旧実装から不変)。prepareに約70分、import約50分。
5. `dataset_build_id`は正規化統計+件数のハッシュで、統計・件数が同じで配列の中身だけ違う場合はresume時に検出できない。
6. `verify_dataset_leakage`は時間順序(重大1)を検査しない。
7. 評価でcorrelationのNaN(12点中定数など)は除外して平均しているが、除外件数を出力していない。
8. LF/CRLFの改行変換警告(`.gitattributes`未設定)。動作には影響しない。
9. テストファイル間の相互import(`test_training`のヘルパーを他が利用)。pytestのrootdir依存で、単独実行時に壊れやすい。
10. bf16推論の影響は合成データ(差0.0016)でのみ確認。実モデルでfp32との差は未確認。

- 判定: GO(全14ステップ完了)。重大指摘1は、ユーザーの判断待ち。

## [2026-09-25] 最終レビュー重大指摘1に対するユーザー判断

- 判断: **guide検索の時系列的な整合性(起点日より後のデータをguideに使わない制約)は、今回は対応しない。**
- 理由(ユーザー): データセットが少ないため。時間制約を全候補に課すと、guide候補が大きく減る
  (実測: inferenceのguideの42.5%が起点日より後のデータ、trainでは46.3%)。
- 受け入れる限界: 報告した精度(MAE・被覆率など)は、運用時に手に入る情報だけで予測した場合より楽観的な可能性がある。
  epsilonとv_predictionは同条件のため、両者の相対比較には影響しない。
- 今後、この結果を運用時の精度の根拠として使う場合は、この限界を明記すること。
  対応する場合の方法: `retrieval/search.py:78`の時間制約を近傍に限らず全候補に適用(再実行に約6時間、guide候補の減少による精度低下が見込まれる)。
- 判定: GO(対応なしで完了)

## [2026-09-27] GitHubへpush / 自己検索の検証の定義を確定(未実装・未実行)

- push: 9コミットを`origin/main`へpush済み(ローカルとリモートが一致)。
- 自己検索の検証: 旧`acceleration_forecasting_12m`の`retrieval/self_check.py`(multiwave DBの自己一致チェック、314/314件合格)を調査し、
  v2向けの定義を`SELF_RETRIEVAL_CHECK.md`に確定した(**定義のみ。実装・本実行は未実施**)。
  - 構成: S1 保存済み自己一致(全件) / S2 再エンコード一致(標本) / S3 自己除外(標本)。合格基準・出力・テスト計画を記載。
  - 引き継がない点: multiwave(10日列)・4距離・3次元の比較(v2はcosine・256次元固定、単位は`record_id`)。
  - 実データで閾値を校正: 自己コサイン 0.99999923〜1.00000083、他レコードとの最大類似度 0.9886、margin最小 0.0114、
    同一埋め込み0件、再エンコードはCUDAでビット一致・CPUで要素差4e-7。
  - 発見: 5,000×335,986の類似度行列は6.3GiBでGPUに載らず(OOM)、チャンク処理が必須。全件S1は約130秒の見込み。
  - 未確定の6点(Q1〜Q6)を同書§10に列挙。ユーザー確認待ち。
- 判定: GO(定義まで)

## [2026-09-27] EMBEDDING_DIM 256→64 への変更・実データ再実行(ユーザー判断)

- 決定: `common/constants.py`の`EMBEDDING_DIM`を256から64に変更。retrieval全体(Autoencoder・
  ベクトルDB・guide検索)がこの値を参照するため、Autoencoder再学習→ベクトルDB再構築→3split再構築
  →リーク検査→epsilon/v_prediction再学習→推論→評価→比較まで全て実データで再実行する。
- 影響範囲の確認: 抽出結果(waveforms.bin/split_manifest.csv/trend_catalog.csv、`artifacts/retrieval`)は
  次元に依存しないため再取り込み不要。`assign_model_split`は同じseed・同じdataset_id集合であれば
  再実行しても同一の分割になるため、train/validation/inferenceの区分は変わらない。
  guide_values/guide_deltas/guide_similarities(検索結果に依存)は変わるため、データセットは再構築が必要。
- テスト: 変更後もテスト170件中170件パス(合成データはdim=8等を使うため次元変更の影響を受けない)。
- 判定: GO(実行を継続)

## [2026-09-28] 実データでの256次元 vs 64次元 比較(EMBEDDING_DIM変更の結果)

### 実行
build-retrieval(dim64) 13分 → prepare 57分 → verify 13秒 → epsilon学習/推論/評価 1.10h/1.68h → v_prediction学習/推論/評価 1.23h/1.66h → compare。
合計 約6時間52分。健全性チェック合格(anchor数は前回と同一: train 36,657/validation 3,301/inference 1,119、guide取得率100%)、
リーク検査違反0件、エラーなし。256次元の一式は`artifacts/*_dim256_backup`に保存済み。

### 結果比較(segment-level、inference 1,119件・116セグメント)
| 指標 | epsilon 256 | epsilon 64 | v_prediction 256 | v_prediction 64 |
|---|---|---|---|---|
| MAE | 0.2131 | 0.2143 | 0.1945 | 0.1946 |
| RMSE | 0.2594 | 0.2607 | 0.2395 | 0.2398 |
| correlation | 0.3006 | 0.3047 | 0.3017 | 0.3035 |
| coverage_p10_p90 | 0.7636 | 0.7667 | 0.7389 | 0.7417 |
| baseline_MAE(guide平均) | 0.3228 | 0.3152 | 0.3228 | 0.3152 |

- **結論: 256次元と64次元で、生成モデルの精度は実質的に差がない**(MAEの差は0.001〜0.0012、correlationの差は0.004以内)。
  256→64での改善は見られなかった。guide平均ベースライン自体がわずかに変わっている(0.323→0.315、選ばれるguideの
  近傍集合が変わるため)が、これも小さい差。
- epsilon/v_predictionの優劣関係(vが点予測で良い、epsilonが較正でわずかに良い)は、両次元で同じ傾向を再現した。
- 判定: GO。次元変更自体は「生成モジュールの精度向上」には寄与しなかったことを記録する。

## [2026-09-28] 自己検索の検証(SELF_RETRIEVAL_CHECK.md)Q1〜Q6を確定 + 自己ガイド検証(SELF_GUIDE_CHECK.md)を新規定義

- SELF_RETRIEVAL_CHECK.md §5を64次元DBで再校正(256次元と同じ傾向を確認: 自己コサイン0.99999958〜
  1.00000036、他最大類似度0.9933、margin最小0.0067、同一埋め込み0件)。§10のQ1〜Q6を確定に更新:
  S1・S2・S3の3層すべて/標本20,000件・S1全件/同点は許容し件数報告/距離はcosineのみ・次元は64固定/
  自己ガイド検証も対象に含める(別文書)/実行対象は現在の64次元DB+合成データテスト。
- 自己ガイド検証は範囲が異なる(検索の配管ではなくモデルの入力活用)ため、`SELF_GUIDE_CHECK.md`として
  別に定義した(定義のみ、未実装)。旧`diagnose-self-guide-conditioning`(既存の学習済みcheckpointに対し、
  正しいguide/シャッフルしたguide/無効化したguideで出力を比較、再学習不要)を引き継ぎ、
  旧`train-self-guide-sanity`一式(専用データセットで再学習、上限性能の確認、数時間規模)は
  範囲外とした。shuffled相手を「別dataset_id」に限定する点、attention可視化は既定オフにする点を
  旧実装から変更。§6に未確定のQ1〜Q6を記載、ユーザー確認待ち。
- 判定: GO(定義まで)

## [2026-09-28] SELF_GUIDE_CHECK.md のQ1〜Q6を確定

- Q1: 900のみ。Q2: model_validation全件(3,301件)。Q3: epsilon・v_predictionの両方。
  Q4: shuffledの相手は別dataset_idになるまでインデックスをずらす(決定的)。
  Q5: 可視化画像は生成せず、数値(guide_advantage_vs_shuffled/disabled、出力L1差、attention内部統計)で
  「どれだけ使われているか」を報告し、旧リポジトリの実測値と並べる。Q6: 固定閾値なし、数値と解釈を報告。
- 旧リポジトリの実測値を発見・追記(§2): **自己ガイド用モデル**(guide=自分自身の正解、29件)は
  advantage +0.05〜+0.07で明確にguideを使っていたが、**通常モデル**(guide=実際の検索結果、
  月次アラインメント+ratd融合・v_prediction、v2と同じattention設定、29件、3シード)は
  advantage +0.00004/−0.0010/−0.0014と、normal_MAE(約0.4)の1000分の1程度で実質ゼロ〜わずかにマイナス。
  「guideを使う能力はあるが、実際の検索結果ではほぼ使われていなかった」可能性を示す重要な事前情報。
  v2はsplit母数が3,301件(旧29件)と大きく異なるため、同じ傾向が出るかは本検証で確認する。
- 判定: GO(定義まで、実装・実行は未着手)

以降、SPEC.mdの実装順序案ステップ2から、1ステップずつ「実装→テスト→レビュー→説明→コミット→本ログ追記」のサイクルを回す。

## [2026-09-28] SELF_RETRIEVAL_CHECK.md 実装(ステップ1+2: `retrieval/self_check.py` + CLI)

- 実装ファイル: `src/acceleration_forecasting_v2/retrieval/self_check.py`(`self_retrieval_check`、
  内部の`_chunked_unfiltered_pass`/`_check_stored_self_match`/`_check_re_encoding`/`_check_self_exclusion`)、
  `cli.py`に`self-check`サブコマンド追加。
- テスト: `tests/test_self_check.py`(12件、新規)、`tests/test_cli.py`に2件追加。
  `.venv/Scripts/python.exe -m pytest tests/ -q` → **184 passed**(既存170 + 新規14、回帰なし)。
- 設計の要点:
  - S1(保存済み自己一致、全件)とS3の`filter_effect_rate`(除外規則を外した場合の上位3件)は、
    どちらも「制限なしの全件コサイン類似度ランキング」を必要とするため、`_chunked_unfiltered_pass`に
    まとめて1回の計算で共有する(二重計算しない)。
  - `record_pass`(個別レコードの合否)はS1の類似度・最大性・正規化・有限性のみで決め、DB整列・
    件数整合・自己除外の破損などは実行全体の`all_pass`にのみ反映する(定義書の設計どおり)。
  - 同一埋め込みの重複は「同点」として許容し記録するのみ(`tied_count`/`strict_top1_rate`で報告、
    `record_pass`は落とさない)。
- レビューで見つけて修正した2点(検出力テストで発覆):
  1. 非有限(NaN/Inf)な埋め込みが1件でもあると、コサイン類似度の行列積で**他の全レコードの
     max_otherまでNaN汚染**し、全件が`is_maximum`で不合格になっていた。類似度計算専用に
     非有限行をゼロベクトルへ隔離する(`finite_ok`/`norm_ok`自体は元の値で判定するため検出力は
     落ちない)よう修正。
  2. S3(自己除外)で、埋め込みが破損していて検索クエリとして使えない場合に`GuideIndex.search`が
     `ValueError`を送出し、**その1件で自己検索の検証全体が異常終了**していた。該当レコードだけを
     検索失敗(`record_pass=False`)として記録し、他のレコードの検証は継続するよう修正。
- 入出力の具体例: 合成DB(12セグメント×22か月、model_train/validation合計198件)で
  `self_retrieval_check`実行時、`all_pass=True`、`s1.pass_rate=1.0`、`s3.self_returned=0`。
  波形を1件改ざんした場合は`s2.cosine_min`が0.999999を下回り`all_pass=False`になることを確認。
- 判定: GO(実データでの実行はステップ5でまとめて行う)
