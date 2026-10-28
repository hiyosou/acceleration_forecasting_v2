"""パイプライン各段階の最小CLI。`python -m acceleration_forecasting_v2.cli <command> ...`

各コマンドは結果をJSONで標準出力に書く。モデルのアーキテクチャ等は固定(SPEC.md 1.3)のため引数にしない。
実験ごとにサブコマンドを増やさない方針(成果物の乱立を防ぐ)なので、ここにあるのは
パイプラインの段階そのものだけ:
  import-snapshot → build-retrieval → prepare → verify → train → predict → evaluate → compare

学習用の正解を忘却因子付き線形近似で「補修から2年」まで延長する場合(2026-10-22)は、prepareの前に
  extend-trends(延長版のtrend_catalog・検索DBを別ディレクトリに作る)
を実行し、prepareに --trend-catalog-path/--database-path/--extend-splits を渡す。比較用に
  predict-forgetting-factor(忘却因子付き線形近似 版D の予測を predict と同じ形式で書き出す)
も持つ。入力に足す特徴(版Dの直線・月内の全計測)の候補を安い予測器(ridge・MLP)でふるう
  screen-features(2026-10-27)
もある。ふるいで選んだ忘却因子Dの要約(D4)を拡散モデルの条件入力に足すには、データセットに
  add-cycle-features を実行し、train に --cycle-features を渡す。

上記のパイプライン段階に加えて、既存成果物の事後検証ツールを6つ持つ(いずれもSPEC.mdの
パイプライン段階ではなく、既にある成果物が正しいかを確認する診断コマンド):
  self-check(SELF_RETRIEVAL_CHECK.md) / diagnose-guide-conditioning(SELF_GUIDE_CHECK.md) /
  plot-prediction(1レコードの予測結果を暦日x軸でPNG化) /
  plot-prediction-batch(複数レコード×複数手法のplot-predictionを一括生成) /
  plot-prediction-pair(指定レコードの拡散モデルの図と忘却因子付き線形近似の図を別々に生成) /
  evaluate-guide-fidelity(生成データと検索guide自体との誤差を評価)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(prog="acceleration_forecasting_v2")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("import-snapshot", help="既存の抽出結果(waveforms.bin+manifest)を取り込みトレンドを12か月で再構築")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--trend-dir")

    p = commands.add_parser("build-retrieval", help="split確定→Autoencoder学習→ベクトルDB構築(抽出済み成果物から)")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device")

    p = commands.add_parser("extend-trends",
                            help="正解を忘却因子付き線形近似で補修から2年まで延長した成果物(catalog・DB)を別ディレクトリに作る")
    p.add_argument("--artifact-dir", required=True, help="元のretrieval成果物(変更しない)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--trend-dir", help="元のtrend CSVのディレクトリ(既定はretrieval.constants.DEFAULT_TREND_DIR)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon-months", type=int, default=24)
    p.add_argument("--min-span-months", type=int, default=6)

    p = commands.add_parser("prepare", help="3splitの学習用データセットを構築")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--trend-catalog-path", help="既定は--artifact-dir内。延長版(extend-trendsの出力)を使うときに指定")
    p.add_argument("--database-path", help="既定は--artifact-dir内。延長版(extend-trendsの出力)を使うときに指定")
    p.add_argument("--extend-splits", nargs="*", choices=("model_train", "model_validation", "inference"),
                   help="合成値で埋めた正解を使うsplit(指定しないsplitは元の欠測に戻す)")
    p.add_argument("--min-months-after-maintenance", type=int,
                   help="--maintenance-rule-splitsのsplitで、起点を補修からこの月数以上に限る")
    p.add_argument("--maintenance-rule-splits", nargs="*", default=(),
                   choices=("model_train", "model_validation", "inference"))
    p.add_argument("--max-datasets-per-split", type=int)
    p.add_argument("--inference-max-per-segment", type=int,
                   help="inferenceを評価用セットとして構築し、セグメントごとに日付均等で最大N件へ間引く")
    p.add_argument("--device")

    p = commands.add_parser("verify", help="構築済みデータセットのリーク検査(違反があれば終了コード1)")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--no-synthetic-split", action="append", default=[],
                   help="正解に合成値の月があってはいけないsplit。複数指定可")
    p.add_argument("--min-months-after-maintenance", type=int)
    p.add_argument("--maintenance-rule-split", action="append", default=[],
                   help="起点が補修から--min-months-after-maintenanceか月以上であるべきsplit。複数指定可")

    p = commands.add_parser("train", help="RMAを学習")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prediction-type", choices=("epsilon", "v_prediction"), default="epsilon")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--self-target-loss-weight", type=float, default=0.0,
                   help="0より大きいと、同じバッチでguideを自分自身の正解に差し替えた場合の"
                        "損失を補助項として加算する(既定0.0で既存動作と完全に同一)")
    p.add_argument("--guide-mix-probability-zero", type=float,
                   help="指定すると、学習時のguideを「λ·正解 + (1−λ)·実guide」に置き換える"
                        "(この確率でλ=0、それ以外はλ~U(0,1))。未指定なら混合しない")
    p.add_argument("--target-mode", choices=("absolute", "residual"), default="absolute",
                   help="residual: 「正解 − (current + guideの変化量の平均)」を生成する")
    p.add_argument("--cycle-features", action="store_true",
                   help="データセットの cycle_features.npy(忘却因子Dの要約8個、add-cycle-features で作る)を条件入力に足す")

    p = commands.add_parser("add-cycle-features",
                            help="データセットをコピーし、各splitに忘却因子Dの要約(D4)の cycle_features.npy を足す")
    p.add_argument("--source-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--trend-catalog-path", required=True, help="全計測を持つ(合成値を含まない)trend_catalog.csv")
    p.add_argument("--normalization-from", help="別のデータセットの cycle_feature_normalization.json を使う場合のそのディレクトリ")

    p = commands.add_parser("predict", help="DDIMで予測(median/p10/p90)")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="inference")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8,
                   help="1回にまとめて生成するレコード数(既定8)。値を変えると各レコードに割り当てる乱数が変わる"
                        "(サンプルの分布は同じ)")
    p.add_argument("--start-timestep", type=int,
                   help="DDIMの最初のtimestep(省略時は958)。epsilonではt=999でx0推定が悪条件になるため"
                        "既定で手前から始める。2026-10-14以前の結果を再現するときは999を指定する")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device")
    p.add_argument("--guide-mode", choices=("retrieved", "self_target"), default="retrieved",
                   help="self_targetは生成モジュールの上限性能(天井)を測る診断モード:"
                        "検索guideの代わりに各レコード自身の正解を全DDIMステップに渡す"
                        "(再学習なし、正解を読み込むためinference splitでも使用時のみ例外的に"
                        "targetファイルを読む)")

    p = commands.add_parser("evaluate", help="record-level / segment-levelで評価")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--prediction-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="inference")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--month-filter", choices=("all", "real"), default="all",
                   help="real: 合成値で埋めた正解の月を除き、実測の月だけで評価する")
    p.add_argument("--min-target-months", type=int, default=8,
                   help="評価に使う正解の月数の下限(--month-filterで残った月で数える)")

    p = commands.add_parser("predict-forgetting-factor",
                            help="忘却因子付き線形近似(版D、λごとに1本)の予測をpredictと同じ形式で書き出す")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--trend-catalog-path", required=True, help="計測値を読むtrend_catalog.csv")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="inference")
    p.add_argument("--lambdas", type=float, nargs="+", help="忘却係数(既定 1.00〜0.90 の0.01刻み)")

    p = commands.add_parser("screen-features",
                            help="入力に足す特徴(版Dの直線・月内の全計測)の候補を、ridge・MLPでふるいにかける")
    p.add_argument("--dataset-dir", required=True, help="延長なしのデータセット(model_train/model_validation/inference)")
    p.add_argument("--trend-catalog-path", required=True, help="全計測を持つtrend_catalog.csv")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--no-mlp", action="store_true", help="補助のMLPを省く")
    p.add_argument("--stage", type=int, choices=(1, 2), default=1,
                   help="1: D系・H系の種類のふるい、2: D4への追加特徴と月の値の置き換え")

    p = commands.add_parser("compare", help="複数runの評価summaryを比較表(CSV)にまとめる")
    p.add_argument("--run", action="append", required=True, metavar="NAME=EVALUATION_DIR")
    p.add_argument("--output", required=True)

    p = commands.add_parser("self-check", help="guide検索の自己一致/再エンコード一致/自己除外を検証(違反があれば終了コード1)")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample-size", type=int, default=20000)
    p.add_argument("--chunk-size", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device")

    p = commands.add_parser("diagnose-guide-conditioning",
                            help="学習済みcheckpointが実際にguide入力を使っているかを診断(再学習なし)")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="model_validation")
    p.add_argument("--timesteps", type=int, nargs="+", default=[900])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device")

    p = commands.add_parser("build-self-reference-dataset",
                            help="guideを自分自身の正解に差し替えたデータセットを構築(生成モジュールの上限性能測定用)")
    p.add_argument("--source-dataset-dir", required=True)
    p.add_argument("--output-dataset-dir", required=True)

    p = commands.add_parser("plot-prediction",
                            help="1レコードの予測サンプル・実測値・guide・施工マーカーを暦日x軸でPNG化")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", default="inference")
    p.add_argument("--prediction-dir", required=True)
    p.add_argument("--artifact-dir", required=True, help="trend_catalog.csvを含むretrieval成果物ディレクトリ")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--trend-id", help="指定時は自動選択を迂回してこのレコードを直接描画する")
    p.add_argument("--maintenance-relation", choices=("any", "immediately_after", "elapsed"), default="any")
    p.add_argument("--immediately-after-months", type=float, default=2.0)
    p.add_argument("--elapsed-months", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bin-width", type=float, default=0.1, help="生成頻度ヒストグラムの加速度刻み幅")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--min-target-range", type=float,
                   help="予測期間(12か月、有効な月のみ)の最大値-最小値がこの値以上のレコードだけ自動選択対象にする")

    p = commands.add_parser("plot-prediction-batch",
                            help="複数レコード×複数手法のplot-predictionを一括生成(レコードごとにフォルダ)")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", default="inference")
    p.add_argument("--artifact-dir", required=True, help="trend_catalog.csvを含むretrieval成果物ディレクトリ")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--method", action="append", required=True, metavar="NAME=PREDICTION_DIR",
                   help="手法名=prediction_dir。複数指定可(例: --method initial_retrieved=... --method improved_self_target=...)")
    p.add_argument("--count-per-category", type=int, default=3,
                   help="immediately_after/elapsedそれぞれから選ぶレコード数")
    p.add_argument("--maintenance-relation", choices=("any", "immediately_after", "elapsed"), default="any")
    p.add_argument("--immediately-after-months", type=float, default=2.0)
    p.add_argument("--elapsed-months", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bin-width", type=float, default=0.1)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--min-target-range", type=float,
                   help="予測期間(12か月、有効な月のみ)の最大値-最小値がこの値以上のレコードだけ選択対象プールに残す")

    p = commands.add_parser("plot-prediction-pair",
                            help="指定レコードごとに、拡散モデルの予測図と忘却因子付き線形近似の図を別々に生成")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", default="inference")
    p.add_argument("--prediction-dir", required=True, help="拡散モデルのsamples.npzを含むディレクトリ")
    p.add_argument("--artifact-dir", required=True, help="trend_catalog.csvを含むretrieval成果物ディレクトリ")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--trend-id", action="append", required=True, help="描画するレコード。複数指定可")
    p.add_argument("--model-label", default="拡散モデル", help="拡散モデルの図のタイトル末尾に付ける名前")
    p.add_argument("--lambdas", type=float, nargs="+", help="忘却係数(既定 1.00〜0.90 の0.01刻み)")
    p.add_argument("--bin-width", type=float, default=0.1)
    p.add_argument("--dpi", type=int, default=150)

    p = commands.add_parser("evaluate-guide-fidelity",
                            help="生成データ(100サンプル)と検索guide自体との誤差(guide忠実度)を評価")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--prediction-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="inference")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "import-snapshot":
        from acceleration_forecasting_v2.retrieval.constants import DEFAULT_TREND_DIR
        from acceleration_forecasting_v2.retrieval.snapshot import import_extraction_snapshot
        result = import_extraction_snapshot(args.source, args.output, trend_dir=args.trend_dir or DEFAULT_TREND_DIR)
    elif args.command == "build-retrieval":
        from acceleration_forecasting_v2.retrieval.pipeline import build_retrieval_from_extraction
        result = build_retrieval_from_extraction(args.artifact_dir, device=args.device, epochs=args.epochs,
                                                 batch_size=args.batch_size)
    elif args.command == "extend-trends":
        from acceleration_forecasting_v2.datasets.extension import build_extended_artifacts
        from acceleration_forecasting_v2.retrieval.constants import DEFAULT_TREND_DIR
        result = build_extended_artifacts(args.artifact_dir, args.output_dir, trend_dir=args.trend_dir or DEFAULT_TREND_DIR,
                                          seed=args.seed, horizon_months=args.horizon_months,
                                          min_span_months=args.min_span_months)
    elif args.command == "prepare":
        from acceleration_forecasting_v2.datasets.prepare import prepare_datasets
        result = prepare_datasets(args.artifact_dir, args.output_dir, device=args.device,
                                  max_datasets_per_split=args.max_datasets_per_split,
                                  inference_max_per_segment=args.inference_max_per_segment,
                                  trend_catalog_path=args.trend_catalog_path, database_path=args.database_path,
                                  extend_splits=args.extend_splits,
                                  min_months_after_maintenance=args.min_months_after_maintenance,
                                  maintenance_rule_splits=args.maintenance_rule_splits)
    elif args.command == "verify":
        import pandas as pd
        from acceleration_forecasting_v2.datasets.verify import verify_dataset_leakage
        manifest = pd.read_csv(Path(args.artifact_dir) / "split_manifest.csv", encoding="utf-8-sig")
        violations = verify_dataset_leakage(args.dataset_dir, manifest, no_synthetic_splits=args.no_synthetic_split,
                                            min_months_after_maintenance=args.min_months_after_maintenance,
                                            maintenance_rule_splits=args.maintenance_rule_split)
        result = {"violations": violations, "violation_count": len(violations)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if violations else 0
    elif args.command == "train":
        from acceleration_forecasting_v2.training.train import train
        result = train(args.dataset_dir, args.output_dir, device=args.device, epochs=args.epochs,
                       batch_size=args.batch_size, patience=args.patience, seed=args.seed,
                       resume=not args.no_resume, progress=not args.no_progress,
                       prediction_type=args.prediction_type,
                       self_target_loss_weight=args.self_target_loss_weight,
                       guide_mix_probability_zero=args.guide_mix_probability_zero,
                       target_mode=args.target_mode, cycle_features=args.cycle_features)
    elif args.command == "add-cycle-features":
        from acceleration_forecasting_v2.datasets.cycle_features import add_cycle_features
        result = add_cycle_features(args.source_dir, args.output_dir, args.trend_catalog_path,
                                    normalization_from=args.normalization_from)
    elif args.command == "predict":
        from acceleration_forecasting_v2.inference.predict import predict
        result = predict(args.dataset_dir, args.checkpoint, args.output_dir, split=args.split, device=args.device,
                         num_samples=args.num_samples, sampling_steps=args.sampling_steps, seed=args.seed,
                         batch_size=args.batch_size,
                         guide_mode=args.guide_mode, start_timestep=args.start_timestep)
    elif args.command == "evaluate":
        from acceleration_forecasting_v2.evaluation.evaluate import evaluate
        result = evaluate(args.dataset_dir, args.prediction_dir, args.output_dir, split=args.split,
                          bootstrap=args.bootstrap, month_filter=args.month_filter,
                          min_target_months=args.min_target_months)
    elif args.command == "predict-forgetting-factor":
        from acceleration_forecasting_v2.evaluation.forgetting_factor import DEFAULT_LAMBDAS, predict_forgetting_factor
        result = predict_forgetting_factor(args.dataset_dir, args.trend_catalog_path, args.output_dir, split=args.split,
                                           lambdas=tuple(args.lambdas) if args.lambdas else DEFAULT_LAMBDAS)
    elif args.command == "screen-features":
        from acceleration_forecasting_v2.evaluation.feature_screening import screen_features
        result = screen_features(args.dataset_dir, args.trend_catalog_path, args.output_dir, mlp=not args.no_mlp,
                                 stage=args.stage)["summary"]
    elif args.command == "compare":
        from acceleration_forecasting_v2.evaluation.evaluate import build_comparison_table
        summaries = {}
        for item in args.run:
            name, _, directory = item.partition("=")
            summaries[name] = json.loads((Path(directory) / "evaluation_summary.json").read_text(encoding="utf-8"))
        table = build_comparison_table(summaries)
        table.to_csv(args.output, encoding="utf-8-sig")
        result = {"output": str(args.output), "columns": list(table.columns), "metrics": list(table.index)}
    elif args.command == "self-check":
        from acceleration_forecasting_v2.retrieval.self_check import self_retrieval_check
        result = self_retrieval_check(args.artifact_dir, args.output_dir, sample_size=args.sample_size,
                                      chunk_size=args.chunk_size, seed=args.seed, device=args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0 if result["all_pass"] else 1
    elif args.command == "diagnose-guide-conditioning":
        from acceleration_forecasting_v2.inference.self_guide_diagnostics import diagnose_guide_conditioning
        result = diagnose_guide_conditioning(args.dataset_dir, args.checkpoint, args.output_dir, split=args.split,
                                             timesteps=tuple(args.timesteps), seed=args.seed,
                                             batch_size=args.batch_size, device=args.device)
    elif args.command == "plot-prediction":
        from acceleration_forecasting_v2.evaluation.visualize import plot_prediction
        result = plot_prediction(args.dataset_dir, args.prediction_dir, args.artifact_dir, args.output_dir,
                                 split=args.split, trend_id=args.trend_id, maintenance_relation=args.maintenance_relation,
                                 immediately_after_months=args.immediately_after_months,
                                 elapsed_months=args.elapsed_months, seed=args.seed, bin_width=args.bin_width,
                                 dpi=args.dpi, min_target_range=args.min_target_range)
    elif args.command == "plot-prediction-batch":
        from acceleration_forecasting_v2.evaluation.visualize import plot_prediction_batch
        methods = {}
        for item in args.method:
            name, _, directory = item.partition("=")
            methods[name] = directory
        result = plot_prediction_batch(args.dataset_dir, args.artifact_dir, args.output_dir, methods,
                                       split=args.split, count_per_category=args.count_per_category,
                                       maintenance_relation=args.maintenance_relation,
                                       immediately_after_months=args.immediately_after_months,
                                       elapsed_months=args.elapsed_months, seed=args.seed,
                                       bin_width=args.bin_width, dpi=args.dpi,
                                       min_target_range=args.min_target_range)
    elif args.command == "plot-prediction-pair":
        from acceleration_forecasting_v2.evaluation.visualize import plot_prediction_pair
        result = plot_prediction_pair(args.dataset_dir, args.prediction_dir, args.artifact_dir, args.output_dir,
                                      trend_ids=args.trend_id, split=args.split, model_label=args.model_label,
                                      lambdas=args.lambdas, bin_width=args.bin_width, dpi=args.dpi)
    elif args.command == "evaluate-guide-fidelity":
        from acceleration_forecasting_v2.evaluation.evaluate import evaluate_guide_fidelity
        result = evaluate_guide_fidelity(args.dataset_dir, args.prediction_dir, args.output_dir, split=args.split)
    else:
        from acceleration_forecasting_v2.datasets.self_reference import build_self_reference_dataset
        result = build_self_reference_dataset(args.source_dataset_dir, args.output_dataset_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
