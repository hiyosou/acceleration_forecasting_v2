"""パイプライン各段階の最小CLI。`python -m acceleration_forecasting_v2.cli <command> ...`

各コマンドは結果をJSONで標準出力に書く。モデルのアーキテクチャ等は固定(SPEC.md 1.3)のため引数にしない。
実験ごとにサブコマンドを増やさない方針(成果物の乱立を防ぐ)なので、ここにあるのは
パイプラインの段階そのものだけ:
  import-snapshot → build-retrieval → prepare → verify → train → predict → evaluate → compare

上記のパイプライン段階に加えて、既存成果物の事後検証ツールを2つ持つ(いずれもSPEC.mdの
パイプライン段階ではなく、既にある成果物が正しいかを確認する診断コマンド):
  self-check(SELF_RETRIEVAL_CHECK.md) / diagnose-guide-conditioning(SELF_GUIDE_CHECK.md)
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

    p = commands.add_parser("prepare", help="3splitの学習用データセットを構築")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-datasets-per-split", type=int)
    p.add_argument("--inference-max-per-segment", type=int,
                   help="inferenceを評価用セットとして構築し、セグメントごとに日付均等で最大N件へ間引く")
    p.add_argument("--device")

    p = commands.add_parser("verify", help="構築済みデータセットのリーク検査(違反があれば終了コード1)")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--artifact-dir", required=True)

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

    p = commands.add_parser("predict", help="DDIMで予測(median/p10/p90)")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="inference")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device")

    p = commands.add_parser("evaluate", help="record-level / segment-levelで評価")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--prediction-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="inference")
    p.add_argument("--bootstrap", type=int, default=1000)

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
    elif args.command == "prepare":
        from acceleration_forecasting_v2.datasets.prepare import prepare_datasets
        result = prepare_datasets(args.artifact_dir, args.output_dir, device=args.device,
                                  max_datasets_per_split=args.max_datasets_per_split,
                                  inference_max_per_segment=args.inference_max_per_segment)
    elif args.command == "verify":
        import pandas as pd
        from acceleration_forecasting_v2.datasets.verify import verify_dataset_leakage
        manifest = pd.read_csv(Path(args.artifact_dir) / "split_manifest.csv", encoding="utf-8-sig")
        violations = verify_dataset_leakage(args.dataset_dir, manifest)
        result = {"violations": violations, "violation_count": len(violations)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if violations else 0
    elif args.command == "train":
        from acceleration_forecasting_v2.training.train import train
        result = train(args.dataset_dir, args.output_dir, device=args.device, epochs=args.epochs,
                       batch_size=args.batch_size, patience=args.patience, seed=args.seed,
                       resume=not args.no_resume, progress=not args.no_progress,
                       prediction_type=args.prediction_type,
                       self_target_loss_weight=args.self_target_loss_weight)
    elif args.command == "predict":
        from acceleration_forecasting_v2.inference.predict import predict
        result = predict(args.dataset_dir, args.checkpoint, args.output_dir, split=args.split, device=args.device,
                         num_samples=args.num_samples, sampling_steps=args.sampling_steps, seed=args.seed)
    elif args.command == "evaluate":
        from acceleration_forecasting_v2.evaluation.evaluate import evaluate
        result = evaluate(args.dataset_dir, args.prediction_dir, args.output_dir, split=args.split,
                          bootstrap=args.bootstrap)
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
    else:
        from acceleration_forecasting_v2.datasets.self_reference import build_self_reference_dataset
        result = build_self_reference_dataset(args.source_dataset_dir, args.output_dataset_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
