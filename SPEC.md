# SPEC.md — acceleration_forecasting_v2 実装仕様書

本書は、壁打ちで確定した要件定義(下表)を実装可能な粒度に落とし込んだものです。
**実装(モデルコード・学習スクリプト本体)はまだ着手していません。** これはレビュー用のドキュメントです。

参照した既存コード: `acceleration_forecasting_12m/`(旧リポジトリ、`my_project`直下)、
`acceleration_retrieval/`(旧retrieval、`my_project`直下)。本文中のファイル参照は特記なき限りこの2つを指します。

## 前提: 確定済み要件定義(転記)

| 領域 | 決定内容 |
|---|---|
| 入力形状 | 6か月履歴に統一(情報量は同じ)、フラットMLPでエンコード |
| anchor選定 | セグメント内、min_target_months=8を満たす月をすべて拾う |
| 学習時のセグメント間偏り | セグメント単位で重み付け(1/セグメント内anchor数) |
| 補修跨ぎ | 現行踏襲、打ち切り+可変長マスク |
| 出力ホライズン | 変更なし、12か月固定 |
| train/val/inference分割 | 現行の2段階dataset_id単位ランダム分割を踏襲(リーク対策済みを確認) |
| 評価の点推定値 | record-level・segment-level(2段階平均)を両方出力 |
| retrieval依存 | 断ち切り、独自パイプラインを構築(コードはコピーして出発点) |
| モデル実装 | RMA一本化、attention_mode/fusion_type/チャンネル構成は固定 |
| 損失関数 | マスク付きMSE + セグメント単位重み付け(新規実装)。prediction_typeはepsilon(Min-SNRなし)を主軸に、v_predictionも並行比較 |
| リポジトリ構成 | my_project内の新規フォルダ(本フォルダ)、旧12mへの追加は停止 |
| target_mode(1.6節、ステップ6実装時に追加決定) | absolute(実測値を直接予測)。guideは条件としてのみ使用、ターゲットからの引き算はしない |

---

# 成果物1: 曖昧な決定事項の仕様化

## 1.1 セグメント単位重み付け

### 選択肢

- **案A(推奨): loss計算時にサンプル単位の重みを掛ける**
  DataLoaderは通常どおり全anchorを均一シャッフルする。各anchorに`segment_weight`(下記式)を事前計算して持たせ、損失関数側で重み付き平均を取る。
- **案B: DataLoaderのsampler側で反映する(`WeightedRandomSampler`)**
  セグメント内anchor数が少ないほど高確率で再サンプリングされるようにする。

### 推奨: 案A

理由:
1. 既存の`DiffusionProcess.loss_details`([diffusion/process.py:31-66](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/diffusion/process.py#L31-L66))は既にMin-SNRの重み(`weights`、タイムステップ由来)を per-record loss に掛けてから`.mean()`を取る構造になっている。案Aはこの既存パターンに「もう1本の重み軸」を足すだけで済み、実装・レビューの差分が最小。
2. `WeightedRandomSampler`は**復元抽出**のため、1エポック内で同じanchorが複数回登場したり、逆に一度も登場しなかったりする確率的ゆらぎが生じる。案Aは決定的(全anchorが必ず1エポックに1回登場)で挙動の予測・デバッグが容易。
3. 案Bは勾配蓄積(`accumulation_steps=2`)・EMA更新タイミングとの相互作用を別途検証する必要があるが、案Aはバッチ内の集計式を変えるだけで既存の学習ループ構造([training/train.py:167-221](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/training/train.py#L167-L221))に影響しない。

### 数式

セグメント(`dataset_id`) `s` に属するanchorの総数を `N_s`(**split単位で計算**。`model_train`内での出現数、`model_validation`内での出現数、をそれぞれ独立に数える。`dataset_id`は必ず単一splitに属するため曖昧さはない)とする。

record `i` のセグメント重み:

```
segment_weight_i = 1 / N_{s(i)}
```

Min-SNR重み `snr_weight_i`(Min-SNR不使用時は`snr_weight_i = 1`、[diffusion/process.py:50-53](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/diffusion/process.py#L50-L53)参照)と合成し、バッチ損失は**重み付き平均**(単純合計ではなく、重みの合計で正規化)とする:

```
combined_weight_i = segment_weight_i * snr_weight_i
batch_loss = Σ_i (combined_weight_i * per_record_loss_i) / Σ_i combined_weight_i
```

`per_record_loss_i`は既存の`per_record`(マスク済み二乗誤差をレコード内で平均したもの、[diffusion/process.py:47-48](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/diffusion/process.py#L47-L48))をそのまま流用する。

### 実装上の注意(バリデーション損失にも適用が必要)

現状の`_validation_loss`([training/train.py:52-65](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/training/train.py#L52-L65))は、early stoppingとbest checkpoint選定に使う`validation_loss`を**単純平均**で計算している。これは学習lossと全く同じ偏りの問題を持つため、`segment_weight`は**学習lossだけでなくvalidation lossにも一貫して適用する**必要がある(適用しないと、モデル選択基準がanchor数の多いセグメントに偏ったままになり、案Aの目的が半分しか達成されない)。

---

## 1.2 補修跨ぎの打ち切り+可変長マスク

### 現行ロジックの整理

現行は2箇所に分散している: [datasets/history.py:35-53](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/datasets/history.py#L35-L53) `truncate_future`と、[retrieval/build.py:37-60](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/retrieval/build.py#L37-L60) `_truncate_future`(DB構築時)。新リポジトリでは後者(DB構築側)に一本化する(前者はforecasting側の後付け加工であり、新パイプラインではDB構築時に確定させた`future_values`/`future_mask`をそのまま信頼できるようにする)。

### 疑似コード

```
function build_future_target(
    anchor_date: date,             # このtrendのanchor日(月初に正規化して扱う)
    monthly_values: dict[int, float | None],  # 1..FORECAST_MONTHS -> 実測値 or 欠測
    next_maintenance_date: date | None,       # このセグメントの「次の」補修日(なければ None)
    forecast_months: int = 12,
) -> (values: float32[forecast_months], mask: float32[forecast_months]):

    values = [NaN] * forecast_months
    mask   = [0.0] * forecast_months

    # 1. 実測値の有無で仮マスクを立てる
    for month_index in 1..forecast_months:
        v = monthly_values.get(month_index)
        if v is not None and is_finite(v):
            values[month_index - 1] = float32(v)
            mask[month_index - 1] = 1.0

    # 2. 次の補修が来る月以降を強制的に打ち切る
    if next_maintenance_date is not None:
        first_target_month = first_day_of_month(anchor_date) + 1 month  # month_index=1が指す月
        cutoff = normalize_to_month_start(next_maintenance_date)
        for month_index in 1..forecast_months:
            target_month = first_target_month + (month_index - 1) months
            if target_month >= cutoff:
                values[month_index - 1] = NaN
                mask[month_index - 1] = 0.0

    # 3. 最終防御: マスク0の位置は必ずNaNに統一(浮動小数の中途半端な値を残さない)
    for month_index in 1..forecast_months:
        if mask[month_index - 1] == 0.0:
            values[month_index - 1] = NaN

    return values.astype(float32), mask.astype(float32)
```

同じロジックを**履歴側**(6か月統一履歴)にも適用する。履歴側は「補修跨ぎ」ではなく「単純な欠測(計測日が存在しない)」のみが対象になる点が異なる([datasets/history.py:7-32](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/datasets/history.py#L7-L32) `select_monthly_history`と同様のロジックを6点に拡張するだけで、補修跨ぎの打ち切りは履歴(過去)には適用しない — 過去は常に確定した実測値であり、未来のように「補修で無効化される」概念がないため)。

### dtype・パディング値の確定

| 項目 | 値 | 根拠 |
|---|---|---|
| `values`のdtype | `float32` | 既存`.astype(np.float32)`([retrieval/build.py](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/retrieval/build.py))を踏襲 |
| `mask`のdtype | `float32`(0.0/1.0、boolではない) | 既存コードがマスクを直接損失計算の乗数として使う([diffusion/process.py:47](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/diffusion/process.py#L47) `squared * mask`)ため、bool→float変換コストを省く目的で踏襲 |
| 無効位置のパディング値(**保存時**) | `NaN` | 「値が存在しない」ことをゼロと区別するため。0.0埋めだと将来別の閾値処理と衝突するリスクがある |
| 無効位置のパディング値(**モデル入力直前**) | `0.0`(`nan_to_num`) | **重要**: RMAはConv1dベースの全系列処理モデルであり、系列中にNaNが1点でも残ると畳み込みで周辺位置に伝播し、マスクされていない位置の予測まで破壊される。既存の`ForecastDataset._finite_zero`([datasets/torch_dataset.py:44-46](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/datasets/torch_dataset.py#L44-L46))が`Dataset.__getitem__`境界でこの変換を行っている。**新パイプラインでもこの境界(保存はNaN、テンソル化直前にnan_to_num)を厳守する** |

---

## 1.3 RMAのattention_mode/fusion_type/チャンネル構成の固定値

### 現行コードのデフォルト値の検証(根拠なき踏襲を避けるため)

`ReferenceModulatedUNet12.__init__`のPythonデフォルト引数([models/reference_modulated_unet.py:117-121](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/models/reference_modulated_unet.py#L117-L121))は`reference_attention_mode="global"`, `reference_fusion_type="residual_delta"`, `reference_similarity_enabled=True`だが、**これは単なるPython関数のデフォルト引数であり、実験で比較検証された値ではない**。

一方、旧リポジトリには実際に完走した比較実験が残っていた: `artifacts_prediction_parameterization/experiment_summary.json`(prediction_type A/B/C × diffusion_steps 2水準、計6構成、model_train/model_validation/inference全てで評価済み)。この実験で使われたアーキテクチャ設定を実際のcheckpointから確認した([artifacts_prediction_parameterization/normal_guides/A_epsilon_t1000/model/resolved_config.json](../acceleration_forecasting_12m/artifacts_prediction_parameterization/normal_guides/A_epsilon_t1000/model/resolved_config.json)):

```json
{
  "reference_similarity_feature": false,
  "reference_similarity_bias": false,
  "reference_attention_mode": "month_aligned",
  "reference_fusion_type": "ratd_condition_reconstruction",
  "base_channels": 64,
  "channel_multipliers": [1, 2, 4],
  "block_counts": [2, 2, 2, 2, 2],
  "condition_dim": 256,
  "attention_heads": 8,
  "attention_head_dim": 8
}
```

つまり**Pythonのデフォルト値(global/residual_delta/similarity有効)とは異なる設定**が、実際に評価まで完走した唯一の系統的比較実験で使われていた。

### 選択肢

- **案A(推奨): 実験で使われた設定を踏襲**
  `reference_similarity_enabled=False`, `reference_attention_mode="month_aligned"`, `reference_fusion_type="ratd_condition_reconstruction"`, `base_channels=64`, `channel_multipliers=(1,2,4)`, `block_counts=(2,2,2,2,2)`, `condition_dim=256`, `reference_dim=64`, `attention_heads=8`, `attention_head_dim=8`
- **案B: Pythonのデフォルト値(global/residual_delta/similarity有効)を踏襲**
- **案C: どちらも未検証扱いとし、新データ形状で両方比較してから決める**

### 推奨: 案A

理由: 案Bは「たまたまコードに書かれていた初期値」であり検証の跡がない。案Aは限定的とはいえ実際にmodel_train/model_validation/inferenceの3面で評価まで通した実績がある唯一の設定。案Cは今回のRMA一本化・設定固定という決定(実験乱立を避ける)と矛盾する。

**ただし一点、今回の決定事項に含まれていない軸として`diffusion_steps`がある。** 同じ実験内でepsilon/v_prediction/x0_predictionいずれも「diffusion_steps=200」が「diffusion_steps=1000」よりMAE/RMSEで一貫してわずかに優れていた(例: epsilon MAE 0.4055→0.3813、v_prediction MAE 0.3712→0.3610、x0_prediction MAE 0.3256→0.3189、いずれも`all_195`集計)。これは今回の壁打ちで明示的に決定した項目ではないため、**本書では現行デフォルトの`diffusion_steps=1000`を維持する案を仮採用しつつ、上記の実測差を付記するに留める**。200へ変更する場合は別途決定が必要。

---

## 1.4 record-level / segment-level評価の集計方法

「segment内でrecordを平均→segment間で平均」という2段階平均で合っているかを数式で確認する。

split内のセグメント集合を `S`、セグメント `s` に属するrecordの集合を `R_s`、record `i` の評価指標(例: RMSE)を `m_i` とする。

**record-level(現行と同じ、単純平均)**:

```
record_level_mean = ( Σ_{s∈S} Σ_{i∈R_s} m_i ) / ( Σ_{s∈S} |R_s| )
```

**segment-level(2段階平均)**:

```
segment_mean_s = ( Σ_{i∈R_s} m_i ) / |R_s|          # 第1段階: セグメント内平均
segment_level_mean = ( Σ_{s∈S} segment_mean_s ) / |S|  # 第2段階: セグメント間平均
```

この定義で合っている。`|R_s|`が全セグメントで等しい(=1、現行の「1セグメント1anchor」)場合、両者は一致する。今回`|R_s|`がセグメントごとに異なるようになるため、両者は乖離しうる — これが壁打ちで確認した偏りそのものである。

**bootstrap CIも同じ2段階構造にする**必要がある。現行の`evaluate()`([evaluation/evaluate.py:93-99](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/evaluation/evaluate.py#L93-L99))は「どのセグメントを再抽出するか」はセグメント単位だが、集計はrecord単位の重み付き平均のままだった(壁打ちで確認済みの問題点)。record-level出力用のbootstrapは現行のまま(record単位集計)、segment-level出力用のbootstrapは、各反復で`segment_mean_s`を計算してから`|S|`個の値の単純平均を取る、という2段階に修正する。

---

## 1.5 epsilon / v_predictionの並行比較の実行方式

**確定: 比較対象は epsilon(Min-SNRなし)と v_prediction の2構成のみとする。** x0_predictionは今回のスコープに含めない(1.3・本節末尾の参考情報で触れたx0_predictionの実測が良好だった点は将来の検討材料として記録するに留め、今回の実装・比較対象には加えない)。

### 選択肢

- **案A(推奨): 同一データセットに対し、config(`prediction_type`)だけを変えて2回学習を回す**
  `train(..., prediction_type="epsilon")`と`train(..., prediction_type="v_prediction")`を別の`output_dir`に対して独立実行。
- **案B: 1回の学習内で2つのheadを同時に持たせ、同時に両方の損失を見る**
  モデル出力を2ヘッド化(epsilon用・v用)し、1回の学習で両方を評価する。

### 推奨: 案A

理由: 案Bは`model_output_to_x0_epsilon`の変換式([diffusion/process.py:94-104](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/diffusion/process.py#L94-L104))がprediction_typeごとに異なるため、2ヘッドの損失を同一の最適化ループで合成すると、どちらのheadの学習信号がもう一方を歪めるか制御が難しい。またRMAは「設定を1パターンに固定」という決定と整合させるなら、**「1回の学習=1つのprediction_type」を単位とする**案Aの方が既存の`train()`関数の構造([training/train.py:68-77](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/training/train.py#L68-L77))ともそのまま整合する。

### 比較に使うメトリクス

`evaluate()`が既に出力している指標をそのまま使う([evaluation/evaluate.py:89-91](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/evaluation/evaluate.py#L89-L91)):
`MAE, MSE, RMSE, correlation, peak_value_error, peak_month_error, coverage_p10_p90, mean_interval_width, median_adjacent_abs_difference, adjacent_difference_MAE, direction_sign_agreement`

比較表は上記を **record-level と segment-level の両方** で並べる(1.4の決定と整合)。特に`coverage_p10_p90`(較正)と`MAE`/`RMSE`(点予測精度)は必ずセットで見る — 過去の実測(1.3参照、および壁打ち中に確認した`models_absolute_reference_modulated` vs `_v`の比較)で「点予測精度が良いほど区間較正が悪化する」というトレードオフが一貫して観測されているため、片方だけで優劣を決めない。

**参考情報(今回の比較スコープには含めないが記録しておく)**: 旧実験の`artifacts_prediction_parameterization`にはx0_predictionの結果も含まれており、点予測精度(MAE/RMSE/correlation)はepsilon/v_prediction/x0_predictionの3つの中で**x0_predictionが最良**、較正(coverage_p10_p90)は**epsilonが最良**という単調なトレードオフが観測されている。今回はユーザー指示によりepsilon/v_predictionの2つに比較対象を限定するが、将来x0_predictionも候補に入れる価値はある。

---

## 1.6 target_mode(residual差分予測 か absolute実測値直接予測か)【ステップ6実装時に発見・追記】

当初のSPEC.mdにはこの軸の決定が漏れていた(1.3節のRMA固定値は決めたが、target_modeは暗黙のままだった)。データセット構築(成果物5 ステップ6)の実装中に、1.3節で根拠にした実測(`artifacts_prediction_parameterization`)が**全てtarget_mode="absolute"**で行われていたことに気付き、確定させた。

### 決定: `target_mode = "absolute"`

- ターゲット(`target_values`)は`anchor.future_values`(実測値そのもの)。guideベースラインの引き算はしない。
- guideは引き続きRMAのreference-attentionへの条件として使われる(`guide_values`/`guide_deltas`/`guide_similarities`)。
- `guide_baselines`/`guide_softmax_weights`はターゲット構成には使わないが、「guideの単純な加重平均と比べてどれだけ拡散モデルが優れているか」を評価する参考値として引き続き算出・保存する(成果物2.1で訂正済み)。
- `prediction_type`(1.5節、epsilon/v_prediction)とは独立な軸であり、どちらのtarget_modeでも「ノイズを予測する」という拡散プロセスの仕組み自体は変わらない。target_modeは「ノイズを加える対象の"クリーンな信号"を何と定義するか」だけを決める。

### 実装への影響

- `datasets/build.py`の`target_norm`(旧SPEC案の`residual_norm`から命名変更)は、model_trainの`target_values`(物理値、absolute)から直接fitする。
- `RawSplit`/`ForecastDatasetV2`のtarget構築ロジックに、guide_baselineを減算するステップは存在しない。

---

# 成果物2: データ契約(I/O仕様)

## 2.1 新DBクエリ結果 → 学習用テンソルへの変換仕様

```
[trends table row]                          [waveform_records row]
  trend_id, dataset_id, measurement_date       record_id, trend_id, embedding(256,)
  current_acc_z_max: float                     measurement_date, direction, bin_start_m/end_m
  future_values/future_mask/selected_dates
  cutoff_maintenance_date
        │
        │ (1.2の疑似コードで future_values/future_mask を確定させた状態で保存されている前提)
        ▼
[anchor候補列挙] dataset_id ごとに measurement_date昇順で全件走査
        │
        │ select_monthly_history(6か月版) で history_values(6,), history_mask(6,) を計算
        │ evaluate_sequence_eligibility で is_offline_sequence_valid / is_production_inference_ready を判定
        │ min_target_months=8 を満たすものを "全件" 採用 (旧: dataset_id毎に1件のみ採用)
        ▼
[selected anchors] 1レコード = 1 (dataset_id, measurement_date, bin_start_m, bin_end_m)
        │
        │ GuideIndex.search() で allowed_splits に従い top_k=3 のguideを取得
        │ softmax_guide_baseline で guide_baselines(12,), guide_softmax_weights(3,12) を算出
        ▼
[.npy 保存形式] (split=model_train/model_validation/inference それぞれ別ディレクトリ)
```

| 配列名 | shape | dtype | 正規化 | 備考 |
|---|---|---|---|---|
| `history_values` | `(N, 6)` | float32 | あり(`condition_norm`、z-score) | **旧`current_values(N,1)`+`history_values(N,5)`を統合**。index 0が最も古い月、index 5が今月(降順/昇順は実装時に固定し、テストで検証する) |
| `history_masks` | `(N, 6)` | float32(0/1) | なし | |
| `guide_values` | `(N, 3, 12)` | float32 | あり(`condition_norm`) | 変更なし |
| `guide_masks` | `(N, 3, 12)` | float32(0/1) | なし | 変更なし |
| `guide_deltas` | `(N, 3, 12)` | float32 | あり(`condition_norm.std`で除算のみ、平均は引かない。旧実装踏襲) | 変更なし |
| `guide_similarities` | `(N, 3)` | float32 | なし | 変更なし |
| `retrieval_masks` | `(N, 3)` | float32(0/1) | なし | 変更なし |
| `guide_baselines` | `(N, 12)` | float32 | なし(物理値) | **訂正**: target_mode=absolute決定(下記参照)によりターゲット構成には使わない。guideの単純平均という「素朴な予測」との比較用の評価参考値として引き続き算出・保存する |
| `guide_softmax_weights` | `(N, 3, 12)` | float32 | なし | **訂正**: 当初`(N,3)`と誤記していたが、`softmax_guide_baseline`の実装(月ごとに有効なguideだけでsoftmaxを取り直すため、重みは月ごとに異なりうる)を確認した結果`(N,3,12)`が正しい。ステップ6実装時に発見・修正 |
| **`segment_weights`**(新規) | `(N,)` | float32 | なし | `1/N_{s(i)}`(1.1参照)。split構築時に確定するため事前計算して保存する |
| `target_values` | `(N, 12)` | float32 | あり(`target_norm`) | **訂正**: target_mode=absolute決定により、anchor.future_valuesそのもの(guide_baseline未使用)。正規化名も「residual_norm」から実態に合わせ「target_norm」に変更 |
| `target_masks` | `(N, 12)` | float32(0/1) | なし | 変更なし |

正規化統計(`condition_normalization.json`、`normalization.json`)は**`model_train`splitのみに`fit`し、model_validation/inferenceには`normalize`のみ適用する**(現行`Normalization.fit(training_values[train_mask>0], ...)`踏襲、[datasets/build.py:309-310](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/datasets/build.py#L309-L310))。

## 2.2 6か月履歴 → フラットMLPエンコードのshape変換

```
入力: history_values (batch, 6) float32, history_masks (batch, 6) float32
  │
  ▼ torch.cat([history_values, history_masks], dim=-1)
中間: (batch, 12) float32
  │
  ▼ nn.Linear(12, 128) → nn.SiLU() → nn.Linear(128, 128)
出力: history_condition (batch, 128) float32
  │
  │ time_embedding (batch, 128) と結合 (RMAの time_encoder は変更なし)
  ▼ torch.cat([history_condition, time_embedding], dim=-1)
最終条件ベクトル: condition (batch, 256) float32   # RMAのcondition_dim=256とそのまま整合
```

旧実装([models/reference_modulated_unet.py:147-150](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/models/reference_modulated_unet.py#L147-L150))との差分は**入力次元が11→12になるだけ**で、`condition_encoder`のLinear層の入力次元以外は無変更。

## 2.3 train/val/inferenceでの前処理差分とリーク発生箇所チェックリスト

| # | チェック項目 | 正しい挙動 | 対応する既存コード(踏襲元) |
|---|---|---|---|
| 1 | `dataset_id`(セグメント)は必ず単一splitにのみ属する | `model_train`/`model_validation`/`inference`が排他的 | [splitting.py:82-101](../acceleration_retrieval/splitting.py#L82-L101) |
| 2 | guide検索の許可split | train→{train}、validation→{train}、inference→{train,validation} | [datasets/build.py:295](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/datasets/build.py#L295) |
| 3 | guide検索の同一セグメント除外 | クエリと同じ`dataset_id`は常に除外 | [retrieval/search.py:63](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/retrieval/search.py#L63) |
| 4 | 空間近傍guideの時間順序制約 | `near`な候補は`guide_available_date < query_date`のみ許可。**【実データ検証で判明した限界】** 制約は空間的に近い候補にしか掛からず、遠い区間のguideは起点日より後のデータでも使われる(inferenceのguideの42.5%が起点日より後に計測、PROGRESS.md最終レビュー指摘1) | [retrieval/search.py:70-72](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/retrieval/search.py#L70-L72) |
| 5 | 正規化統計の`fit`範囲 | `model_train`のみ | [datasets/build.py:309-310](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/datasets/build.py#L309-L310) |
| 6 | Autoencoder(埋め込み)の学習データ範囲 | development(=train+validation)側の波形のみ。inference側波形は埋め込み計算(クエリ用)はするが学習には使わない | [acceleration_retrieval README](../acceleration_retrieval/README.md) 実行順セクション |
| 7 | inference anchorの適格性判定 | 未来値を参照しない`is_production_inference_ready`のみで判定(offlineの`is_offline_sequence_valid`は使わない) | [datasets/build.py:114-115](../acceleration_forecasting_12m/src/acceleration_forecasting_12m/datasets/build.py#L114-L115) |
| 8 | `segment_weights`の算出範囲 | split内で完結(他splitのanchor数を混ぜない) | 新規(1.1) |
| 9 | inferenceの正解(`future_values`)の隔離 | 検索処理・特徴量計算からは触れない別ファイル(`inference_targets.csv`相当)に分離 | [acceleration_retrieval README](../acceleration_retrieval/README.md) |

---

# 成果物3: 主要クラス・関数のインターフェース定義

中身は実装せず、シグネチャとdocstringのみ。

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# anchor選定
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorCandidate:
    """1つの(dataset_id, measurement_date, bin_start_m, bin_end_m)候補を表す。

    _select_anchors 相当の走査で dataset_id ごとに複数生成されうる点が
    旧実装(1 dataset_id につき1件のみ採用)との主な違い。
    """
    trend_id: str
    dataset_id: str
    measurement_date: "pd.Timestamp"
    bin_start_m: float
    bin_end_m: float
    direction: str
    current_acc_z_max: float
    history_values: np.ndarray  # shape (6,) float32
    history_mask: np.ndarray  # shape (6,) float32
    future_values: np.ndarray  # shape (12,) float32, 1.2の打ち切り適用済み
    future_mask: np.ndarray  # shape (12,) float32
    valid_history_months: int
    valid_future_months: int


def select_anchors_for_segment(
    segment_rows: "pd.DataFrame",
    *,
    split: Literal["model_train", "model_validation", "inference"],
    min_history_months: int,
    min_target_months: int,
    history_months: int = 6,
    forecast_months: int = 12,
) -> list[AnchorCandidate]:
    """1つのdataset_id(施工間セグメント)内で、適格性条件を満たす月を「全て」拾う。

    旧実装 `_select_anchors` ([datasets/build.py:96-139]) は
    「最も履歴が埋まっている1件」のみを選んでいたが、本関数はその制約を外し、
    `split == "inference"` なら `is_production_inference_ready`、
    それ以外なら `is_offline_sequence_valid` を満たす候補を全て返す。

    Returns:
        measurement_date昇順にソートされたAnchorCandidateのリスト。
        1件も適格な候補がなければ空リストを返す(呼び出し側で
        "segment_without_anchor" として診断カウントする)。
    """
    ...


def compute_segment_weights(
    anchors: Sequence[AnchorCandidate],
) -> dict[str, float]:
    """dataset_id ごとに 1/N_s (N_s = そのdataset_idのanchor数) を計算する。

    Args:
        anchors: 単一split内の全AnchorCandidate。他splitのanchorを混ぜてはいけない
            (1.1参照、split境界を越えた重み計算はリークではないが仕様違反)。

    Returns:
        {dataset_id: weight} の辞書。全dataset_idの重み付き件数
        (weight * N_s) の合計は常に |unique dataset_id| に等しくなる
        (各セグメントが「重み1相当」を均等に持つことの検算に使える)。
    """
    ...


# ---------------------------------------------------------------------------
# マスク生成
# ---------------------------------------------------------------------------

def build_future_target(
    anchor_date: "pd.Timestamp",
    monthly_values: dict[int, float | None],
    next_maintenance_date: "pd.Timestamp | None",
    *,
    forecast_months: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """1.2の疑似コードをそのまま実装する。

    Returns:
        (values, mask): 共に shape (forecast_months,) float32。
        mask==0 の位置の values は必ず NaN (保存形式としてのセンチネル)。
        呼び出し側(Dataset.__getitem__相当)で nan_to_num(0) を適用するまで、
        このNaNをモデルに直接渡してはならない。
    """
    ...


def build_history_window(
    section_history: "pd.DataFrame",
    anchor_date: "pd.Timestamp",
    *,
    months: int = 6,
    search_days: int = 15,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """今月を含む直近6か月分の実測値を月単位で選び出す。

    旧 `select_monthly_history` ([datasets/history.py:7-32]) の history_months
    引数を5から6に拡張し、offset起点を「今月を含む」よう1ずらしたもの。

    Returns:
        (values, mask, dates): shape (months,) の値配列・マスク配列と、
        各月に採用された実測日の文字列リスト(空文字は欠測)。
    """
    ...


# ---------------------------------------------------------------------------
# 損失関数
# ---------------------------------------------------------------------------

class SegmentWeightedMaskedLoss:
    """マスク付きMSE + セグメント単位重み付けを行う損失計算器。

    既存 DiffusionProcess.loss_details ([diffusion/process.py:31-66]) を
    置き換えるのではなく、その `per_record` 相当の値を受け取って
    セグメント重みを合成するアダプタとして実装する
    (diffusionのノイズスケジュール/prediction_type変換ロジックとは独立させる)。
    """

    def __init__(self, min_snr_gamma: float | None = None):
        """
        Args:
            min_snr_gamma: Noneならタイムステップ由来の重みは1.0固定
                (今回の主軸であるMin-SNRなしepsilonに対応)。
        """
        ...

    def batch_loss(
        self,
        per_record_loss: torch.Tensor,  # shape (batch,)
        segment_weights: torch.Tensor,  # shape (batch,)
        snr_weights: torch.Tensor | None = None,  # shape (batch,), Noneなら全て1.0
    ) -> torch.Tensor:
        """1.1の数式 (Σ w_i * loss_i) / (Σ w_i) を計算するスカラーを返す。

        segment_weights は DataLoader が __getitem__ で返す `segment_weight`
        フィールドをそのままバッチ化したものを想定する(モデルやデータセット側で
        再計算しない — 1.1で決めた「split構築時に確定・保存」を前提とする)。
        """
        ...


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------

class ReferenceModulatedUNetV2(nn.Module):
    """RMA (ReferenceModulatedUNet12) の一本化・設定固定版。

    旧 `ReferenceModulatedUNet12` ([models/reference_modulated_unet.py:114-176]) との違い:
    1. `condition_encoder` の入力次元が 11 (current 1 + history 5 + mask 5) から
       12 (history 6 + mask 6) に変わる (2.2参照)。current/historyの区別が
       なくなるため、コンストラクタから current 専用の引数・処理を削除する。
    2. attention_mode/fusion_type/チャンネル構成などのアーキテクチャ引数は
       コンストラクタのデフォルト値として固定し、CLI/学習関数からは
       上書きできないようにする (1.3の推奨値を既定値として埋め込む)。
    """

    def __init__(
        self,
        dropout: float = 0.1,
        *,
        history_months: int = 6,
        forecast_months: int = 12,
        # 以下、1.3で確定した固定値をデフォルトとして埋め込む。
        # CLI/学習コードからの上書きは想定しない (露出させない、という決定に基づく)。
        reference_similarity_enabled: bool = False,
        reference_attention_mode: Literal["global", "month_aligned"] = "month_aligned",
        reference_fusion_type: Literal[
            "residual_delta", "ratd_condition_reconstruction"
        ] = "ratd_condition_reconstruction",
        base_channels: int = 64,
        channel_multipliers: tuple[int, int, int] = (1, 2, 4),
        block_counts: tuple[int, int, int, int, int] = (2, 2, 2, 2, 2),
        condition_dim: int = 256,
        reference_dim: int = 64,
        attention_heads: int = 8,
        attention_head_dim: int = 8,
    ):
        ...

    def forward(
        self,
        noisy: torch.Tensor,  # shape (batch, forecast_months)
        timesteps: torch.Tensor,  # shape (batch,)
        batch: dict[str, torch.Tensor],  # history_values/history_masks/guide_* を含む
    ) -> torch.Tensor:  # shape (batch, forecast_months)
        """旧 forward ([models/reference_modulated_unet.py:207-228]) と同じ入出力契約。

        batch は最低限 "history_values", "history_masks", "guide_values",
        "guide_deltas", "guide_similarities", "guide_mask", "retrieval_mask" を含む。
        旧実装の "current"/"history" キーは廃止し、"history_values"(6点)に統合する。
        """
        ...


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------

def evaluate_record_level(
    per_target_metrics: "pd.DataFrame",  # 1行 = 1 target_id、列に MAE/RMSE/...
) -> dict[str, float]:
    """1.4の record_level_mean を各指標列について計算する。旧 evaluate() の
    `summary = {name: frame[name].mean(...) for name in metric_names}` と同じ挙動。
    """
    ...


def evaluate_segment_level(
    per_target_metrics: "pd.DataFrame",  # "dataset_id" 列を含む
) -> dict[str, float]:
    """1.4の segment_level_mean (2段階平均) を各指標列について計算する。

    Steps:
        1. dataset_id ごとに指標列を平均 (segment_mean_s)
        2. その平均値の単純平均 (segment_level_mean)
    """
    ...


def bootstrap_segment_level(
    per_target_metrics: "pd.DataFrame",
    *,
    iterations: int = 1000,
    seed: int = 42,
) -> "pd.DataFrame":
    """dataset_idを復元抽出→各反復内でevaluate_segment_levelを計算、を繰り返す。

    旧 evaluate() のbootstrapループ ([evaluation/evaluate.py:93-99]) は
    セグメント単位で再抽出はしていたが、集計はrecord単位の重み付き平均のままだった
    (1.4で確認した問題点)。本関数は集計も2段階平均に統一する。
    """
    ...


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ForecastDatasetV2(torch.utils.data.Dataset):
    """旧 ForecastDataset ([datasets/torch_dataset.py:20-71]) の6か月履歴版。

    __getitem__ が返す辞書のキーは "current"/"history" ではなく
    "history_values"(6,) に統合される。"segment_weight"(スカラー) が新たに追加される。
    NaN→0置換 (_finite_zero相当) は本クラスの __getitem__ 内で行い、
    保存された .npy ファイル自体は NaN を保持したままにする (1.2参照)。
    """

    def __init__(self, split_dir: Path, dataset_root: Path, *, include_targets: bool = True):
        ...

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ...
```

---

# 成果物4: 受け入れ基準(テスト観点)のリスト

## `test_mask_generation.py`

| テスト関数 | 検証内容 |
|---|---|
| `test_future_target_no_maintenance_returns_full_mask` | `next_maintenance_date=None`のとき、実測値がある月は全てmask=1になる |
| `test_future_target_truncates_from_maintenance_month` | 補修日が5か月目の月初にある場合、mask[0:4]=1、mask[4:12]=0になる(境界: 補修月**当月**は打ち切り対象に含まれる) |
| `test_future_target_truncates_immediately_next_month` | 補修が anchor の翌月に来る場合、mask が全て0になる(=このanchorはmin_target_months未満で不採用になるはずだが、本関数単体では0マスクを返すことだけを確認する) |
| `test_future_target_single_valid_month` | 有効な月が1つだけのケースでmask.sum()==1、他は0かつvalues=NaN |
| `test_future_target_padding_is_nan_not_zero` | mask=0の位置のvaluesが厳密にNaNであり0.0ではないことを`np.isnan`で確認 |
| `test_future_target_dtype_is_float32` | values.dtype/mask.dtypeが共にfloat32 |
| `test_history_window_six_months_all_valid` | 6か月連続で実測がある場合、mask全て1、valuesが期待順序(古い→新しい)で並ぶ |
| `test_history_window_partial_missing_months` | 一部欠測がある場合、該当月のみmask=0・values=NaN、他は正しい値 |
| `test_history_window_excludes_future_dates` | `anchor_date`以降の日付を履歴として絶対に採用しない(`available < anchor`の境界値テスト) |
| `test_nan_to_zero_applied_only_at_dataset_getitem` | `.npy`保存値はNaNのまま、`ForecastDatasetV2.__getitem__`が返すtorch.Tensorには`NaN`が一切含まれないことを確認 |

## `test_anchor_selection.py`

| テスト関数 | 検証内容 |
|---|---|
| `test_select_anchors_collects_all_eligible_months` | 8か月分の有効な未来を持つ月が複数ある人工セグメントで、旧実装なら1件だが新実装は複数件返すことを確認 |
| `test_select_anchors_respects_min_target_months` | `min_target_months=8`未満の月は候補から除外される |
| `test_select_anchors_empty_segment_returns_empty_list` | 適格なanchorが1つもないセグメントは空リスト(例外を投げない) |
| `test_select_anchors_inference_uses_production_ready_flag` | `split="inference"`のとき、未来値が欠けていても`is_production_inference_ready`のみで判定される(=offlineでは弾かれる条件でも通る) |
| `test_select_anchors_sorted_by_measurement_date` | 返り値が`measurement_date`昇順であることを確認(後続のsegment_weight計算やレポートの前提) |

## `test_segment_weighting.py`

| テスト関数 | 検証内容 |
|---|---|
| `test_compute_segment_weights_hand_example` | 手計算例: セグメントAに10件、セグメントBに1件 → weight_A=0.1が10件全部、weight_B=1.0が1件、という辞書が返ることを確認 |
| `test_segment_weighted_loss_matches_hand_computed_example` | 1.1の数式をSEG_A(loss=0.20×10件)・SEG_B(loss=0.50×1件)の手計算(結果0.35)と一致することを`SegmentWeightedMaskedLoss.batch_loss`で確認 |
| `test_segment_weighted_loss_equals_plain_mean_when_all_weights_equal` | 全レコードが異なるセグメントに1件ずつ属する(=全て重み1)なら、重み付き平均が単純平均と一致する |
| `test_segment_weighted_loss_combines_snr_weight_multiplicatively` | `segment_weight`と`snr_weight`を独立に設定し、combined_weightが積になっていることを確認 |

## `test_split_leakage.py`

| テスト関数 | 検証内容 |
|---|---|
| `test_dataset_id_appears_in_exactly_one_split` | 全`dataset_id`について、`model_train`/`model_validation`/`inference`のいずれか1つにのみ出現する |
| `test_guide_search_train_query_never_returns_non_train_guide` | `model_train`のクエリで検索した結果に含まれる`model_split`が全て`"model_train"` |
| `test_guide_search_validation_query_never_returns_validation_or_inference_guide` | `model_validation`のクエリ結果に`model_split=="model_validation"`または`"inference"`が混入しない |
| `test_guide_search_inference_query_never_returns_inference_guide` | `inference`のクエリ結果に他の`inference`レコードが混入しない(自split除外) |
| `test_guide_search_excludes_same_dataset_id` | クエリと同じ`dataset_id`の候補が検索結果に絶対に出てこない(旧anchorが複数化しても成立することを確認する回帰テスト) |
| `test_normalization_stats_fit_only_uses_model_train_rows` | `Normalization.fit`に渡される値の集合が、`model_validation`/`inference`の値を1件も含まない |
| `test_autoencoder_training_set_excludes_inference_waveforms` | retrieval側のAutoencoder学習データに`outer_split=="inference"`の波形が含まれない |

## `test_evaluation_aggregation.py`

| テスト関数 | 検証内容 |
|---|---|
| `test_record_level_mean_matches_plain_average` | `evaluate_record_level`が`DataFrame.mean()`と一致する |
| `test_segment_level_two_stage_mean_hand_example` | 1.4の数式の手計算例(セグメントA: 平均0.20、セグメントB: 平均0.50 → 0.35)と一致する |
| `test_segment_level_differs_from_record_level_when_unbalanced` | セグメントごとのrecord数が不均衡なデータで、2つの集計値が異なることを確認(一致してしまう場合はテストデータ設計ミスの検出にもなる) |
| `test_segment_level_equals_record_level_when_balanced` | 全セグメントのrecord数が等しい場合、2つの集計値が一致する |
| `test_bootstrap_segment_level_resamples_by_dataset_id` | bootstrap 1反復で選ばれる単位が`dataset_id`であり、record単位ではないことをモック乱数で確認 |

## `test_retrieval_migration.py`(新旧retrieval出力差分の検証)

| テスト関数 | 検証内容 |
|---|---|
| `test_new_trend_catalog_matches_old_for_sample_dates` | 同一の入力CSV群(小規模フィクスチャ)に対して、旧`acceleration_retrieval`のtrend生成ロジックと新パイプラインの生成結果を比較し、`current_acc_z_max`/`future_values`/`future_mask`が(浮動小数誤差許容で)一致する |
| `test_new_waveform_embedding_is_deterministic` | 同一seedで2回embeddingを計算し、bit-exactではなくとも許容誤差内で一致する(CUDA非決定性を考慮) |
| `test_new_vector_db_schema_matches_spec` | 新SQLiteスキーマが本書2.1の表と一致する(カラム名・型) |
| `test_new_pipeline_reproduces_old_split_ratio` | 同一seedでのdataset_id分割比率(80/20→9/1)が旧実装と一致する |

---

# 成果物5: 実装順序案

依存関係の下流から着手すると手戻りが大きいため、**基盤(データ)→モデル→学習→評価**の順で進める。各ステップは前段のテストが通ることを次段着手の前提条件とする。

| # | ステップ | 主な成果物 | 完了判定基準 |
|---|---|---|---|
| 1 | フォルダ雛形・環境構築 | `pyproject.toml`, パッケージ骨格, `uv.lock` | `uv run python -c "import acceleration_forecasting_v2"`が例外なく通る |
| 2 | retrievalパイプラインのフォーク・移植 | 波形抽出・SQLiteスキーマ・検索除外ロジック一式(2.1入口) | `test_retrieval_migration.py`が全てパス |
| 3 | マスク生成・履歴窓構築の実装 | `build_future_target`, `build_history_window` | `test_mask_generation.py`が全てパス |
| 4 | anchor選定ロジックの実装 | `select_anchors_for_segment` | `test_anchor_selection.py`が全てパス |
| 5 | セグメント重み計算の実装 | `compute_segment_weights` | `test_segment_weighting.py`の重み計算部分がパス |
| 6 | データセット構築パイプライン一式 | `.npy`一式(2.1の表を全て出力)、正規化統計 | 小規模フィクスチャで全配列のshape/dtypeが2.1の表と一致。`test_split_leakage.py`の正規化関連テストがパス |
| 7 | `ForecastDatasetV2`の実装 | Dataset/DataLoader層 | 1バッチ取り出してshape/dtypeが契約通り、かつNaNが一切含まれない(`test_nan_to_zero_applied_only_at_dataset_getitem`) |
| 8 | `ReferenceModulatedUNetV2`の実装 | モデル本体(1.3の固定値埋め込み) | forwardを1回通し、出力shapeが`(batch, 12)`。パラメータ数を旧RMA(約488万、[artifacts_prediction_parameterization/.../resolved_config.json](../acceleration_forecasting_12m/artifacts_prediction_parameterization/normal_guides/A_epsilon_t1000/model/resolved_config.json)参考値)と概ね近いオーダーであることを目視確認 |
| 9 | `SegmentWeightedMaskedLoss`の実装 | 損失関数クラス | `test_segment_weighting.py`(手計算例含む)が全てパス |
| 10 | 学習ループの実装(epsilon/v_prediction両対応) | `train()`相当関数 | 小規模データでのsmoke訓練でlossが単調減少傾向を示す(厳密な収束基準は設けない)。validation lossにもsegment_weightが適用されていることをログで確認 |
| 11 | DDIM推論の実装 | `predict()`相当関数 | 1バッチ予測を実行し、median/p10/p90が全て有限値で出力される |
| 12 | 評価関数の実装(record/segment-level) | `evaluate_record_level`, `evaluate_segment_level`, `bootstrap_segment_level` | `test_evaluation_aggregation.py`が全てパス |
| 13 | リーク検証テストの全実行 | `test_split_leakage.py`一式 | 全てパス(実データ規模での実行を含む) |
| 14 | epsilon/v_prediction比較の本実行 | 2本の学習・評価run、比較表 | 両configの`evaluation_summary.json`相当が揃い、1.5の比較表(record-level/segment-level両方)が作成できる |

この順序案・各ステップの完了基準についてもレビューをお願いします。フィードバック後、実装フェーズ(コーディング)に進みます。
