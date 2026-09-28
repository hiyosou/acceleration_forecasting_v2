# SELF_GUIDE_CHECK.md — guide条件の利用診断: 定義書

**状態: 定義のみ(未実装・未実行)。** `SELF_RETRIEVAL_CHECK.md`とは対象が異なる別文書
(検索の配管ではなく、**学習済みモデルが実際にguide入力を使っているか**を診断する)。
末尾「§6 確認したい点」を確定させてからv1として実装する。

---

## 1. 目的

RMA(`ReferenceModulatedUNetV2`)は、今月+履歴に加えてguide(似た過去の波形の将来推移3件)をattentionで
条件として使う設計になっている。しかし設計上そうなっていることと、**学習の結果、実際にguideの情報を
予測に反映しているか**は別問題である。attentionの重みが実質ゼロに近く学習され、guideがほとんど
無視されている可能性は排除できない。

本検証は、既に学習済みのcheckpoint(`artifacts/runs/{epsilon,v_prediction}/model/best_model.pt`)に対して、
**再学習なしで**guideの入力を「正しいまま」「入れ替え」「無効化」の3通りに変えて出力を比較し、
guideが予測に効いているかを診断する。

**目的にしないこと**: guideをどう改善すれば精度が上がるか(それは`SELF_RETRIEVAL_CHECK.md`や
生成モジュール側の改善検討の役割)。本検証は「今の学習済みモデルがguideを使っているか」という
現状把握のみを行う。

## 2. 旧実装との違い(引き継ぐ点・引き継がない点)

旧リポジトリには目的の異なる2つの仕組みがあった。

| 旧実装 | 内容 | コスト | v2での扱い |
|---|---|---|---|
| `diagnose-self-guide-conditioning` | **既存の学習済みモデル**に対し、実際のguide/シャッフルしたguide/無効化したguideで出力を比較 | 再学習不要、数分 | **これを本書で定義・引き継ぐ** |
| `train-self-guide-sanity`一式(`prepare-self-guide-sanity`→`train-self-guide-sanity`→`evaluate-self-guide-sanity`) | guideを「自分自身の正解」に固定した専用データセットを作り、**新しいモデルを再学習**して、guideを最大限利用できる状況でモデルが実際に利用するかを見る(上限性能の確認) | 新規データセット構築+再学習(数時間) | **今回は定義しない**(§7 範囲外)。コストが大きく、「今の学習済みモデル」の診断という目的からも外れるため |

以降、本書は前者(`diagnose-self-guide-conditioning`相当)のみを対象とする。

## 3. 方法

対象split(既定`model_validation`、3,301件)の各レコードについて、**1つの固定タイムステップ**でモデルに
3通りの入力を与え、出力を比較する。

### なぜ1つの固定タイムステップで十分か、なぜ「ノイズの多い」タイムステップを選ぶか
拡散モデルは、タイムステップが小さい(ノイズがほぼ無い)ほど、入力の`noisy`自体がほぼ正解に等しくなるため、
条件(履歴・guide)を使わなくてもある程度当てられてしまう。逆にタイムステップが大きい(ノイズがほとんど)ほど、
`noisy`から得られる情報が乏しく、**条件(履歴・guide)への依存度が最も顕著に現れる**。
そのため旧実装同様、既定は**t=900(全1000ステップ中、ノイズが多い側)**に固定する。

### 手順(1レコードあたり)
1. データセット(`ForecastDatasetV2`、`include_targets=True`)から実際の1件を取り出す(実際のguide付き)。
2. レコードIDから決定的に導いたseedで固定ノイズ`noise`を生成し、`noisy = sqrt(alpha_t)*target + sqrt(1-alpha_t)*noise`を作る(t=900固定)。
3. 3パターンでモデルに通す:
   - **normal**: そのまま(実際のguide)
   - **shuffled**: `guide_values/guide_deltas/guide_mask/guide_similarities/retrieval_mask`を、**別のdataset_idを持つ**別レコードのものに入れ替える
     (旧実装は単純に「次のレコード」だったが、同一dataset_id内の隣接anchor(月違い)は互いに似た値になりやすく、
     入れ替えたつもりで実質ほぼ同じ情報を与えてしまう恐れがある。そのため入れ替え先は**dataset_idが異なる**
     レコードに限定する)
   - **disabled**: `guide_mask`と`retrieval_mask`をゼロにする(attentionで有効なguideトークンが無い状態)
4. 各出力を`model_output_to_x0_epsilon`で予測値(x0)に変換し、`physical_prediction`で物理値に戻し、
   有効な月のみでMAEを計算する(`normal_MAE`, `shuffled_MAE`, `disabled_MAE`)。
5. モデル出力そのものの差(`normal_vs_shuffled_output_L1`, `normal_vs_disabled_output_L1`)も記録する
   ―guideを変えても出力が全く変わらないなら、MAEの差を見るまでもなく「使っていない」ことが分かる。
6. `model.diagnostic_stats()`(`reference_context_norm`, `attention_rank_1/2/3`)も記録する
   ―attentionが実際に非ゼロの文脈を作れているかの直接証拠。

### 集計
- `guide_advantage_vs_shuffled = mean(shuffled_MAE - normal_MAE)`(正なら「正しいguide」の方が「入れ替えたguide」より良い)
- `guide_advantage_vs_disabled = mean(disabled_MAE - normal_MAE)`(正なら「正しいguide」の方が「guide無し」より良い)
- 上記2つを**レコード単位でも**保存し、平均だけでなく分布(中央値・下位/上位パーセンタイル)も見られるようにする。

## 4. 出力

- `condition_usage_per_target.csv` — レコードごとの`normal_MAE/shuffled_MAE/disabled_MAE`、出力L1差、
  `guide_advantage_vs_shuffled/disabled`、attention統計
- `condition_usage_summary.json` — 対象件数、上記指標の平均、使用checkpoint・model_config
- (既定オフ、`--plot`指定時のみ)代表3件(advantage最小・中央値・最大)のattention可視化画像

## 5. 実装方針(未実装)

- 場所: `src/acceleration_forecasting_v2/inference/self_guide_diagnostics.py`
  - `diagnose_guide_conditioning(dataset_dir, checkpoint, output_dir, *, split="model_validation", timesteps=(900,), device=None, seed=42, plot=False) -> dict`
  - `timesteps`をタプルにして複数指定できるようにする(既定は900のみ、旧実装と同じ)。指定した各tごとに
    別々の summary/CSV を出す(`condition_usage_per_target_t{timestep}.csv`等)。
- 既存の`DiffusionProcess.model_output_to_x0_epsilon`・`ForecastDatasetV2.physical_prediction`・
  `ReferenceModulatedUNetV2.diagnostic_stats`をそのまま使う(検証用に別実装を書かない)。
- CLIに`diagnose-guide-conditioning --dataset-dir ... --checkpoint ... --output-dir ...`を追加。

### テスト計画(合成データ)
| テスト | 内容 |
|---|---|
| 出力形状・件数 | 件数=対象split件数、CSVの列が仕様通り |
| shuffled相手がdataset_id違いになっている | 全レコードで確認 |
| disabled時にguide_mask/retrieval_maskが実際に0 | フォワード直前のbatchを検査 |
| 決定的seed | 同じseedなら同じ`noise`が生成される(2回実行して一致) |
| 極端ケース: guideを無視するダミーモデル | forwardでguide関連キーを一切参照しないモデルを渡すと、3パターンの出力が完全一致し、
  advantageが全て0になることを確認(検出力のテスト、偽陰性を出さないことの確認) |
| 極端ケース: guideのみで決まるダミーモデル | forwardがguide_valuesの平均だけを返すモデルを渡すと、disabled時に出力が大きく変わることを確認 |

## 6. 確認したい点(未確定)

| # | 論点 | 提案 | 理由 |
|---|---|---|---|
| Q1 | 対象タイムステップ | 900のみ(旧実装と同じ) | ノイズが多いほど条件への依存が顕著。複数タイムステップにするとコストは増えるが傾向の変化も見える(要判断) |
| Q2 | 対象件数 | model_validation全件(3,301件) | 再学習不要・DDIM不使用のため1件あたり数十ms程度で全件が現実的 |
| Q3 | 対象checkpoint | epsilon・v_predictionの両方 | 既に両方の学習済みcheckpointがある(SPEC 1.5の並行比較と一貫) |
| Q4 | shuffledの相手選び | 別dataset_idになるまでインデックスをずらす(決定的) | 旧実装の「次のレコード」は同一セグメント内の隣接anchor(似た値)を掴む恐れがある |
| Q5 | attention可視化画像 | 既定オフ、`--plot`指定時のみ生成 | 成果物の増殖を避ける方針と一貫。数値サマリで十分な場合が多い |
| Q6 | 合否判定 | 固定の合否閾値は設けず、数値と解釈を報告するのみ(`guide_advantage_vs_disabled`が0以下なら注意書きを添える) | 「使っているか」は程度問題であり、旧実装も閾値を設けていない |
