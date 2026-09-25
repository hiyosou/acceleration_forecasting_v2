"""合成成果物から 学習 → 推論 → 評価 までを通す統合テスト(小規模・CPU)。

個々のステップ間の入出力(shape/dtype/ファイル形式)が食い違っていないことの確認が目的で、
予測精度そのものは検証しない(それは実データでのステップ14の役割)。
"""

import numpy as np
import pandas as pd

from synthetic_artifacts import build_artifacts

from acceleration_forecasting_v2.datasets.prepare import prepare_datasets
from acceleration_forecasting_v2.evaluation.evaluate import build_comparison_table, evaluate
from acceleration_forecasting_v2.inference.predict import predict
from acceleration_forecasting_v2.training.train import train


def test_prepare_train_predict_evaluate_pipeline_runs_end_to_end(tmp_path):
    artifact_dir, dataset_dir = tmp_path / "artifacts", tmp_path / "dataset"
    build_artifacts(artifact_dir, n_datasets=12, months=22, seed=1)
    prepare_datasets(artifact_dir, dataset_dir, device="cpu")

    summaries = {}
    for prediction_type in ("epsilon", "v_prediction"):
        run_dir = tmp_path / prediction_type
        train(dataset_dir, run_dir / "model", device="cpu", epochs=2, batch_size=16, accumulation_steps=1,
              learning_rate=1e-3, progress=False, prediction_type=prediction_type)
        predict(dataset_dir, run_dir / "model" / "best_model.pt", run_dir / "pred", device="cpu",
                num_samples=8, sampling_steps=4, batch_size=4)
        summaries[prediction_type] = evaluate(dataset_dir, run_dir / "pred", run_dir / "eval", bootstrap=20)

    for summary in summaries.values():
        assert np.isfinite(summary["record_level"]["MAE"])
        assert np.isfinite(summary["segment_level"]["MAE"])
        assert summary["counts"]["evaluated_records"] > 0
    table = build_comparison_table(summaries)
    assert {"epsilon|record_level", "v_prediction|segment_level"} <= set(table.columns)
    assert table.loc["MAE"].notna().all()


def test_predictions_cover_every_prepared_inference_record(tmp_path):
    artifact_dir, dataset_dir = tmp_path / "artifacts", tmp_path / "dataset"
    build_artifacts(artifact_dir, n_datasets=12, months=22, seed=2)
    prepare_datasets(artifact_dir, dataset_dir, device="cpu")
    train(dataset_dir, tmp_path / "model", device="cpu", epochs=1, batch_size=16, accumulation_steps=1,
          progress=False)
    predict(dataset_dir, tmp_path / "model" / "best_model.pt", tmp_path / "pred", device="cpu",
            num_samples=4, sampling_steps=3)
    metadata = pd.read_csv(dataset_dir / "inference" / "metadata.csv", encoding="utf-8-sig")
    predictions = pd.read_csv(tmp_path / "pred" / "predictions.csv")
    assert set(predictions["trend_id"]) == set(metadata["trend_id"])
    assert len(predictions) == len(metadata) * 12
