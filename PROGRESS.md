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

以降、SPEC.mdの実装順序案ステップ2から、1ステップずつ「実装→テスト→レビュー→説明→コミット→本ログ追記」のサイクルを回す。
