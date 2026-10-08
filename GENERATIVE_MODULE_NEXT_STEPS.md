# GENERATIVE_MODULE_NEXT_STEPS.md — 生成モジュール改善の優先順位付きアクションプラン

**状態: 2026-10-08時点の分析・提案。未実装・未実行(P1含む)。実行はユーザーの別途指示を待つ。**

本書の対象は「通常推論(自己参照ではない)時に、生成結果と検索guideおよび正解targetが
乖離している」という課題。過去の一連の診断結果とモデル構造(`ReferenceModulatedUNetV2`)を
突き合わせ、次に何を・どの順でやるべきかをまとめる。

---

## 1. 前提: これまでの診断が示す因果関係

| 実験 | 結果 | 含意 |
|---|---|---|
| self_target再学習(guide=正解で新規学習、PROGRESS.md 2026-09-30) | correlation 0.9999 | モデルは「guideをコピーする」能力自体は十分持てる(容量の問題ではない) |
| self_target診断(再学習なし、推論時だけ正解をguideとして投入、PROGRESS.md 2026-09-30) | MAE改善+0.0003程度(ほぼゼロ) | 学習済みモデルは完璧なguideを与えても使わない |
| → attention配分(block_breakdown) | self_target時に91.2%が1スロットに集中(両checkpointで同一値) | **attentionのルーティング自体は正常**(どのguideが有効かは正しく認識できている) |
| → block別context内訳 | enc12/dec12付近で-5〜7%、mid3(ボトルネック)で+4〜5% | 文脈情報はmid3までは増幅されるが、出口にかけて目減りする |
| → **FiLM(scale/shift)内訳**(決定的、PROGRESS.md 2026-09-30) | context変化(最大10%)に対しFiLM自体の変化はその1/10〜1/300(0.02〜1%) | ボトルネックは`ReferenceModulatedAttention.condition_fusion` + `ConditionalBlock.film` |
| `context_residual`追加(アーキテクチャ側、ゼロ初期化の直接残差) | self_target_improvementが**悪化**、FiLM応答は追加前とほぼ変わらず(下記§2参照) | 「経路を増やせば使われる」は不成立 → ボトルネックは容量でなく**学習インセンティブ** |
| `self_target_loss_weight`(損失関数側、補助損失) | self_target_improvementが445倍に(診断は大成功)、`guide_advantage_vs_shuffled`は符号反転したが事前閾値に僅差未達、**最終DDIM指標(MAE/correlation)は不変** | 診断レベルの改善が最終出力に転写されない |
| 検索方法の変更(guide品質+27%、本日) | guide_vs_target_MAE大幅改善、**最終DDIM指標はほぼ不変** | retrieval品質はボトルネックの主因ではない |

一貫しているのは、**guide側(検索品質)をどう変えても、guide経路側(アーキテクチャ容量)を
どう変えても最終出力は動かないが、学習の損失関数(インセンティブ)を変えると診断レベルでは
明確に動く**という構図。モデルは物理的にguideを使えるし、使うよう仕向けることもできるが、
現状の学習のかけ方では最終サンプリング結果にまで反映されていない。

## 2. 新たに確認した手がかり(2026-10-08、既存データの再分析)

`self_target_loss_weight`チェックポイント(`artifacts/runs_self_target_aux_loss/epsilon`)の
FiLM内訳(`self_guide_check/block_breakdown_t900.csv`)を他2チェックポイントと比較すると、
PROGRESS.mdに未記載だった点が見える。

**FiLM(scale_norm)のnormal→self_target相対変化(代表ブロック)**

| ブロック | 現行epsilon(補助損失なし) | context_residual | self_target_loss_weight |
|---|---|---|---|
| enc12-1 | 約+0.04% | -0.13% | **+2.9%** |
| mid3-2(ボトルネック) | 約+0.04% | -0.07% | **+11.7%**(最大) |
| dec12-1(出口直前) | 約-0.24% | -0.32% | **-13.0%**(符号反転・最大の逆方向) |
| dec12-2(出口直前) | 約-0.22% | -0.38% | **-8.6%** |

補助損失はFiLMの反応性を確かに引き出した(0.02〜1% → 2〜13%)。しかし**出口に一番近い
dec12-1/dec12-2だけ、他の全ブロックと逆方向に動く**(より良いguideを与えると、むしろ
出口直前の変調が小さくなる)。この符号反転は3チェックポイントとも位置が一致しており、
構造的なものと考えられる。`self.output`(最終GroupNorm+Conv1d)はdec12の直後にあるため、
「補助損失は診断指標を大きく動かしたのに最終DDIM出力には反映されなかった」ことの、
最も出力に近い場所での直接的な説明になりうる。

もう一つ: **これまでの`diagnose-guide-conditioning`診断は全チェックポイント・全実験を
通じて`t=900`の1点でしか実行されていない**(`artifacts/`内の`condition_usage_summary_t*.json`
/`block_breakdown_t*.csv`を全探索して確認済み、t=900以外は皆無)。DDIMは`t=999→0`を
50ステップで辿る(`diffusion/process.py::DiffusionProcess.ddim`)が、guideが実際にどの
ステップで効くべきか(高ノイズ=全体形状決定 vs 低ノイズ=数値レベルの微調整)は未検証。
CLIは`--timesteps`が複数値対応(`nargs="+"`)で既に実装されているのに、一度も使われていない。

## 3. 優先順位付きアクション

### P1(最優先・即実行可・再学習不要): 診断をt=900単点からDDIM全区間に拡張

`diagnose-guide-conditioning --timesteps 50 200 400 600 800 900 950`のように複数timestepで
現行best checkpoint(`runs_self_target_aux_loss/epsilon`)に対して実行し、
`guide_advantage_vs_shuffled`・FiLM内訳がtでどう変わるかを見る。低t(出力に近い、数値を
詰める段階)でguide利用が改善していれば、DDIMのどこかのステップ数/スケジュールの工夫で
拾える可能性がある。逆に全tで同じ傾向(出口ブロックで逆転)なら、P2・P3の必要性が
さらに裏付けられる。コストは数分、既存ツールの引数を変えるだけ。

### P2(安価): dec12の符号反転を狙い撃ちで深掘り

3チェックポイント共通で出口直前(dec12)だけFiLMが逆方向に動く原因を、
`condition_fusion`/`film`の重み統計(ノルム、勾配ノルム)を直接見る、または`dec12`
ブロックだけ`fusion_type`を`residual_delta`に変える等の局所的アブレーションで切り分ける。
P1の結果次第で優先度を調整。

### P3(本命・中コスト): 新retrieval方式のguideで学習データ自体を作り直して再学習

2026-10-08の発見(guideの質は改善できるが、今のモデルは応答しない)を踏まえると、
最も筋が通る次の一手は「model_train/model_validationのguideも新しい検索方式
(`GuideIndex.search_by_history`)で作り直し、`self_target_loss_weight`
(必要なら重みを1.0から引き上げて再挑戦)と組み合わせて再学習する」こと。理由:
モデルは**今の(品質の低い)guideの分布でしか学習していない**ため、検索方法だけ変えて
良いguideを推論時に与えても、モデルにとっては未知の分布で評価していることになり、
改善しなくて当然とも言える。学習時点からguide品質自体を底上げすれば、`condition_fusion`
がguideの変動に反応する動機(今は「弱い信号なので無視するのが合理的」)も変わるはず。

### P4(中コスト): self_target_loss_weightを「二値」から「段階的」に

現状の補助損失は「実guide」と「完全な正解」の2点だけを学習させている。これが
`self_target_improvement`だけ桁違いに伸びて`guide_advantage_vs_shuffled`が僅差止まりだった
非対称性(実guideのような中間品質の入力に対する校正が訓練中に一度も出てこない)の一因と
考えられる。guideと正解を連続的に混ぜる(ノイズレベルを変えた中間品質のguideを合成して
訓練に混ぜる)ことで、実際のguide品質のスペクトラムに対して校正された反応を学習させる。
P3(新しいguide分布での再学習)と同時に試すのが効率的。

### P5(アーキテクチャ側、優先度低め): condition_fusionの再設計 / reference_similarity_enabledを有効化

`context_residual`(並列の加算経路を追加)は失敗したので、単純な容量追加ではなく、
guideごとの信頼度(similarity)を明示的にFiLMへ伝える(`reference_similarity_enabled=True`、
SPEC.md 1.3では旧リポジトリの実験設定を踏襲して`False`に固定されたのみで、v2内で
明示的に比較検証されたわけではない)仕組みや、`condition_fusion`の共有MLP自体を信頼度で
ゲーティングする設計に変える。P3/P4(学習インセンティブ側の修正)を先に試し、それでも
最終指標が動かない場合の次の手として位置づける。

## 4. 参考: 本書が前提とする関連ファイル

- `PROGRESS.md`(2026-09-30の一連のエントリ、2026-10-08の検索方法変更エントリ)— 本書§1の根拠
- `src/acceleration_forecasting_v2/models/reference_modulated_unet.py` — `ReferenceModulatedAttention`
  (`condition_fusion`/`context_residual`)、`ConditionalBlock`(`film`)、`GuideEncoder12`
- `src/acceleration_forecasting_v2/inference/self_guide_diagnostics.py` — `diagnose_guide_conditioning`
  (本書§3 P1/P2で使う診断ツール)
- `src/acceleration_forecasting_v2/diffusion/process.py` — `DiffusionProcess.ddim`(本書§2後半の根拠)
- `SPEC.md` 1.3 — `reference_similarity_enabled`等の固定値の経緯(本書§3 P5の根拠)
