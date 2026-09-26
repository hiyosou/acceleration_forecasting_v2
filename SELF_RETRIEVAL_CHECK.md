# SELF_RETRIEVAL_CHECK.md — 自己検索の検証: 定義書

**状態: 定義のみ(未実装・未実行)。** 実装前にレビューし、末尾「§10 確認したい点」を確定させる。
旧リポジトリ(`acceleration_forecasting_12m`)で行っていた自己検索の検証を、v2のretrieval(単一波形の埋め込みDB)向けに
定義し直したもの。数値は全て実データ(`artifacts/retrieval/vector_database.sqlite`)での実測に基づく。

---

## 1. 目的と、検証しないこと

**目的**: 「DBに保存した各レコードを、そのレコード自身の埋め込みで検索すると、自分自身が最も似たものとして返る」ことを確認し、
検索の土台(保存・読み込み・正規化・行列の整列・類似度計算・除外規則)に取り違えや破損が無いことを保証する。

これで検出したい不具合の例:
- 埋め込み配列と`record_id`の並びのずれ(SQLのJOIN順やソート順の食い違い)
- 保存時のdtype/バイト数/次元の不一致、L2正規化の漏れ
- 検索の除外規則の不具合による「自分自身・同じセグメント・同じ日」のguide混入(リーク)
- 推論時にその場で埋め込む経路(`encode_waveforms`)が、DB構築時と違う結果を返す

**検証しないこと**: 埋め込みが波形の意味的な類似を捉えているか(=検索の品質)。自己検索は数学的にほぼ必ず通るため、
通っても品質の保証にはならない。あくまで「配管の健全性」の検査である。

## 2. 旧実装の要約(引き継ぐ点・引き継がない点)

旧`retrieval/self_check.py`(`self-check-multiwave-retrieval`): multiwave DB(1つの`data_id`=10日分の日別埋め込み列)の各`data_id`を、
自分自身の列でクエリし、全`data_id`とのスコア(日別コサイン類似度の平均)を計算。合格条件は
「自己の各日類似度が全て≥0.999999」「平均≥0.999999」「保存列とクエリ列がバイト一致」「自己のスコアが最大(同点許容1e-6)」。
DBのSHA256が前後で不変であることも確認。結果: 256/64/32次元すべて314件中314件が合格(自己類似度の最小値1.0)。

| 旧実装の要素 | v2での扱い |
|---|---|
| 検索単位=data_id(10日分の埋め込み列) | **引き継がない**。v2は1波形=1埋め込み(`waveform_records`)。単位は`record_id` |
| 4種の距離(cosine/euclidean/dot/manhattan) | **引き継がない**。v2はcosine固定(埋め込みはL2正規化済み) |
| 3種の次元(256/64/32)の比較 | **引き継がない**。v2は256次元固定 |
| 合格閾値 0.999999、同点許容 1e-6 | **引き継ぐ**(§5で実測により妥当性を確認) |
| DBのSHA256不変の確認、読み取り専用で開く | **引き継ぐ** |
| 自己が最大なら合格(同点は許容し、件数を別途報告) | **引き継ぐ** |

## 3. 検証の構成(3層)

| 層 | 名前 | 何を確かめるか | 対象件数 |
|---|---|---|---|
| S1 | 保存済み自己一致 | 各レコードを自身の保存埋め込みで検索→自分が最大 | **全件** |
| S2 | 再エンコード一致 | 波形から再計算した埋め込みが、DBの保存値と一致 | 標本(既定20,000件、seed固定) |
| S3 | 自己除外 | 実際のguide検索(`GuideIndex.search`)が自分自身・同一セグメント・同一日を返さない | 標本(既定20,000件、seed固定) |

S1は「肯定側」(自分は見つかるはず)、S3はその対となる「否定側」(実際のguide検索では自分は見つからないはず)。
S1で自分が確実に1位であることを確認したうえでS3を行うので、S3の除外が壊れていれば自分自身が返って検出できる(空振りしない)。
S2は、inference側が使う経路(その場で埋め込む)とDB構築時の経路が同一の結果を返すことの確認。

### S1 保存済み自己一致(全件)
1. `GuideIndex`(実際の検索が使うメモリ上のインデックス)を読み込み、`embeddings`(N×D)と`record_ids`を取得する。
2. **整列検査**: 各`record_id`について、`GuideIndex`の該当行の埋め込みが、SQLの`waveform_records`の同一`record_id`のBLOBとバイト一致する。
3. 各レコード`i`をクエリ`q_i = e_i/‖e_i‖`として、全候補との類似度`s_ij = q_i · (e_j/‖e_j‖)`を、**チャンク(既定250クエリ)ごとに**GPUで計算する
   (5,000×335,986の行列は6.3GiBでGPUに載らないため、チャンク処理が必須)。
4. レコードごとに記録: `self_similarity = s_ii`、`max_other = max_{j≠i} s_ij`、`margin = s_ii − max_other`、
   `tied_count = #{j : |s_ij − max_j s_ij| ≤ tie_tolerance}`、`embedding_norm`。
5. DBを読み取り専用(`mode=ro`)で開き、実行前後のSHA256が一致することを確認する。

### S2 再エンコード一致(標本)
1. 標本の`record_id`ごとに、manifestの`waveform_index`から`waveforms.bin`の波形を読み、`encode_waveforms`(`autoencoder.pt`)で埋め込みを再計算する。
2. DBの保存埋め込みとのコサイン類似度、要素ごとの最大絶対差、ビット一致か否かを記録する。
3. 実行機器の違いによる差を切り分けるため、既定は**DB構築と同じ機器(CUDA)**で行い、CPUでの差も参考値として報告する。

### S3 自己除外(標本)
1. 標本の各レコードについて、その自身の保存埋め込みをクエリ、`query_dataset_id`・`query_date`・`query_current`・`query_bin_start_m`をそのレコードの値にして、
   `GuideIndex.search(..., allowed_splits={model_train, model_validation}, config=既定)`を呼ぶ(実際のprepareが使う設定と同じ)。
2. 返ったguideに、自分自身の`record_id`/`trend_id`、同一`dataset_id`、同一`measurement_date`が**一つも含まれない**ことを確認する。
3. 除外が実際に働いていることの証拠として、`filter_effect`(=S1の無制限検索で上位3件に同一`dataset_id`のレコードが含まれていたレコードの割合)も報告する。

## 4. 合格基準

| # | 基準 | 閾値 | 層 |
|---|---|---|---|
| A | 各レコードの`self_similarity` | ≥ 0.999999 | S1 |
| B | 自己が最大(同点許容) | `s_ii ≥ max_j s_ij − 1e-6` | S1 |
| C | 埋め込みの健全性 | 有限、`|‖e‖−1| ≤ 1e-5`、次元=metadataの`embedding_dim`(256) | S1 |
| D | 整列検査 | 全`record_id`でバイト一致 | S1 |
| E | DB不変 | 実行前後のSHA256一致 | S1 |
| F | レコード数 | DB件数 = metadataの`record_count`、`record_id`は一意 | S1 |
| G | 再エンコード一致 | コサイン ≥ 0.999999(同機器ではビット一致を期待) | S2 |
| H | 自己除外 | 自分・同一dataset_id・同一日のguideが0件 | S3 |

- レコード単位の合格 = A ∧ B ∧ C(S1)。全体の合格 = 全レコードが合格 ∧ D ∧ E ∧ F ∧ G(全標本) ∧ H(全標本)。
- **厳密な1位**(同点なしで自己が1位)の割合`strict_top1_rate`は合格条件にせず報告のみ(旧実装と同じ扱い)。
  同点は許容するが`tied_count>1`の件数を必ず報告し、実データでは0件を期待する。
- 1件でも不合格なら`self_retrieval_failures.csv`に理由(`failure_reason`)つきで出力し、終了コードは1とする。

## 5. 閾値の根拠(実データでの校正、`artifacts/retrieval`のDB、335,986件)

| 実測項目 | 結果 | 閾値との関係 |
|---|---|---|
| 埋め込みのノルム(全件) | 0.9999998 〜 1.0000001 | C(±1e-5)に十分な余裕 |
| 自己コサイン(float32、5,000件標本) | 0.99999923 〜 1.00000083 | A(0.999999)を満たす。理論下限もノルム²≈0.9999996 > 0.999999 |
| 自己が`argmax`になった割合(5,000件標本) | 5,000/5,000 | B |
| 他レコードとの最大類似度(5,000件標本) | 中央値0.816、p99 0.974、最大0.9886 | 同点許容1e-6・閾値0.999999のいずれとも大きく離れている |
| 自己と次点の差 `margin`(5,000件標本) | 最小0.0114、下位1% 0.0259 | 取り違えを検出できる十分な差 |
| 埋め込みバイトが完全に一致する別レコード | 0件 | 同一波形による同点は発生しない |
| `waveform_sha256`の種類数 | 335,986(全件が異なる) | 同上 |
| 再エンコード vs 保存値、CUDA(2,000件) | ビット一致 2,000/2,000 | Gに余裕 |
| 再エンコード vs 保存値、CPU(2,000件) | コサイン最小0.99999976、要素差最大4.0e-7 | G(0.999999)を満たす |

## 6. 対象データと規模

- 実データのDB: `waveform_records` 335,986件(model_train 303,664 / model_validation 32,322)、`trends` 91,011件、256次元。
  同じ`trend_id`に複数のレコード(同じ日・区間の複数走行)を持つトレンドが82,575件ある。**S1の単位は`record_id`**(走行ごと)とする。
  guide検索の実運用(`build_raw_split`)は同一anchorの複数レコードを束ねて最大類似度を取るが、この束ね方は本検証の対象外(§9)。
- inference側のレコードはDBに存在しない(guide候補にならない)ため、S1・S3の対象外。inference経路は**S2の再エンコード一致**で担保する。
- 実行時間の見込み(RTX 2080): S1全件 約130秒(実測: 5,000クエリ1.9秒)。S2 20,000件 約20秒。S3 20,000件は`GuideIndex.search`が1件あたり約数十msのため数分〜十数分。
- 標本は`numpy.random.default_rng(seed=0)`で`record_id`から非復元抽出(件数・seedは引数で変更可)。

## 7. 出力

`--output-dir`直下に:
- `self_retrieval_results.csv` — S1のレコードごとの結果: `record_id, trend_id, dataset_id, model_split, self_similarity, max_other_similarity, margin, tied_count, embedding_norm, alignment_ok, record_pass, failure_reason`
- `self_retrieval_failures.csv` — 不合格レコードのみ
- `re_encode_results.csv` — S2標本の結果: `record_id, cosine, max_abs_diff, bit_identical, device`
- `self_exclusion_results.csv` — S3標本の結果: `record_id, returned_count, contains_self, contains_same_dataset, contains_same_date, unfiltered_top3_has_same_dataset`
- `self_retrieval_summary.json` — 全体サマリ:
  `database, database_sha256_before/after, database_unchanged, record_count, model_split_counts, embedding_dim,`
  `s1: {checked, pass_count, pass_rate, strict_top1_rate, tied_records, self_similarity_min/mean/max, margin_min/p1/median, alignment_failures},`
  `s2: {sampled, cosine_min, bit_identical_rate, max_abs_diff_max, device, cpu_reference_cosine_min},`
  `s3: {sampled, self_returned, same_dataset_returned, same_date_returned, filter_effect_rate},`
  `thresholds: {similarity, tie_tolerance, norm_tolerance, re_encode_cosine}, all_pass`

## 8. 実装方針(未実装)

- 場所: `src/acceleration_forecasting_v2/retrieval/self_check.py`
  - `self_retrieval_check(artifact_dir, output_dir, *, similarity_threshold=0.999999, tie_tolerance=1e-6, chunk_size=250, sample_size=20000, seed=0, device=None) -> dict`
  - 内部を`_check_stored_self_match`(S1)、`_check_re_encoding`(S2)、`_check_self_exclusion`(S3)に分ける。
- CLI: 既存の段階コマンドと同じ形で `self-check --artifact-dir ... --output-dir ...`(1サブコマンドのみ。距離・次元の切り替え引数は設けない)。
- 既存の`GuideIndex`・`encode_waveforms`をそのまま使い、検索の実コードを検査対象にする(検査用に別実装を書かない)。
- 新しい依存は追加しない。

### テスト計画(合成成果物`tests/synthetic_artifacts.py`を利用)
| テスト | 内容 |
|---|---|
| 正常系 | クリーンなDBで全層が合格、`all_pass=True`、DBのSHA256不変 |
| 異常注入: 埋め込み入れ替え | あるレコードの埋め込みを別レコードのものに書き換える → S1のA/Bまたは整列検査(D)が検出 |
| 異常注入: 正規化漏れ | ある埋め込みを2倍にする → Cが検出 |
| 異常注入: 同点 | 2レコードに同一の埋め込みを入れる → `tied_count`を報告(合格は維持、`strict_top1_rate<1`) |
| 異常注入: 波形改変 | `waveforms.bin`の1件を書き換える → S2が検出 |
| 異常注入: 除外の破損 | `GuideIndex`の同一dataset除外を外した状態にする(monkeypatch) → S3が検出 |
| 読み取り専用 | DBが変更されない(SHA256)/書き込み用に開かれていない |
| 件数不整合 | metadataの`record_count`と実件数が違うDBでFが検出 |
| チャンク境界 | `chunk_size`を件数と割り切れない値にしても結果が変わらない |

## 9. 範囲外
- multiwave(10日分の列)の自己検索、複数距離・複数次元の比較(§2)。
- 同一anchorの複数走行を束ねた検索(トレンド単位)の自己検索。`GuideIndex.search`の複数クエリ埋め込みの最大値取りは`test_guide_search.py`と`prepare`経由の検証で担保。
- 検索の品質評価(意味的な類似の妥当性)。
- **自己ガイド検証**(旧`self-guide-sanity`: 自分自身の正解をguideとして与えたとき、モデルがguideを使うか)は別の検証で、本書の対象外。

## 10. 確認したい点(未確定)

| # | 論点 | 提案 | 理由 |
|---|---|---|---|
| Q1 | 検証の範囲 | S1・S2・S3の3層すべて | S1だけだと検索の除外規則とinference経路が未検証のまま残る。S2・S3は低コスト |
| Q2 | S2・S3の対象件数 | 標本20,000件(seed固定)。S1は全件 | S1は全件でも約130秒。S2・S3は全件にする利点が小さい |
| Q3 | 同点の扱い | 許容して件数を報告(旧実装と同じ) | 実データでは同点が発生しない(最大他類似度0.9886) |
| Q4 | 旧実装の4距離・3次元の比較 | 引き継がない(cosine・256次元のみ) | v2の設計が固定のため。成果物の乱立を避ける |
| Q5 | 自己ガイド検証(§9) | 今回は対象外 | 別の目的の検証のため。必要なら別途定義する |
| Q6 | 実行対象 | 実データ`artifacts/retrieval`のDBで1回実行+合成データの単体テスト | 実データの校正値(§5)がある |
