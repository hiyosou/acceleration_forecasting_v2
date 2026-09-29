"""cli.main が各段階を順に呼べることの確認(合成データ、CPU、小規模)。"""

import json
import sqlite3

import pandas as pd

from synthetic_artifacts import build_artifacts
from test_snapshot import _make_source

from acceleration_forecasting_v2.cli import main
from acceleration_forecasting_v2.datasets.build import write_dataset


def _run(capsys, *argv):
    code = main([str(a) for a in argv])
    out = capsys.readouterr().out
    return code, json.loads(out)


def test_cli_runs_every_stage_in_order(tmp_path, capsys):
    source, trend_dir, _ = _make_source(tmp_path, n_datasets=12, months=22)
    artifacts, dataset = tmp_path / "artifacts", tmp_path / "dataset"

    code, result = _run(capsys, "import-snapshot", "--source", source, "--output", artifacts, "--trend-dir", trend_dir)
    assert code == 0 and result["eligible_records"] == 12 * 22

    code, result = _run(capsys, "build-retrieval", "--artifact-dir", artifacts, "--epochs", 1, "--device", "cpu")
    assert code == 0 and (artifacts / "vector_database.sqlite").is_file()

    code, result = _run(capsys, "prepare", "--artifact-dir", artifacts, "--output-dir", dataset, "--device", "cpu")
    assert code == 0 and set(result["splits"]) == {"model_train", "model_validation", "inference"}

    code, result = _run(capsys, "verify", "--dataset-dir", dataset, "--artifact-dir", artifacts)
    assert code == 0 and result["violation_count"] == 0

    evaluations = {}
    for prediction_type in ("epsilon", "v_prediction"):
        run = tmp_path / prediction_type
        code, result = _run(capsys, "train", "--dataset-dir", dataset, "--output-dir", run / "model", "--epochs", 1,
                            "--batch-size", 32, "--device", "cpu", "--no-progress", "--prediction-type", prediction_type)
        assert code == 0 and result["model_config"]["prediction_type"] == prediction_type
        code, result = _run(capsys, "predict", "--dataset-dir", dataset, "--checkpoint", run / "model" / "best_model.pt",
                            "--output-dir", run / "pred", "--num-samples", 4, "--sampling-steps", 3, "--device", "cpu")
        assert code == 0 and result["num_samples"] == 4
        code, result = _run(capsys, "evaluate", "--dataset-dir", dataset, "--prediction-dir", run / "pred",
                            "--output-dir", run / "eval", "--bootstrap", 10)
        assert code == 0 and "segment_level" in result
        evaluations[prediction_type] = run / "eval"

    code, result = _run(capsys, "compare", "--run", f"epsilon={evaluations['epsilon']}",
                        "--run", f"v_prediction={evaluations['v_prediction']}", "--output", tmp_path / "compare.csv")
    table = pd.read_csv(tmp_path / "compare.csv", index_col=0, encoding="utf-8-sig")
    assert code == 0 and "MAE" in table.index
    assert list(table.columns) == result["columns"]


def test_cli_verify_returns_nonzero_when_leakage_is_present(tmp_path, capsys):
    source, trend_dir, _ = _make_source(tmp_path, n_datasets=12, months=22)
    artifacts, dataset = tmp_path / "artifacts", tmp_path / "dataset"
    main(["import-snapshot", "--source", str(source), "--output", str(artifacts), "--trend-dir", str(trend_dir)])
    main(["build-retrieval", "--artifact-dir", str(artifacts), "--epochs", "1", "--device", "cpu"])
    main(["prepare", "--artifact-dir", str(artifacts), "--output-dir", str(dataset), "--device", "cpu"])
    capsys.readouterr()
    path = dataset / "model_train" / "metadata.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    row = frame.index[frame["guide_count"] > 0][0]
    values = json.loads(frame.at[row, "guide_model_splits"])
    values[0] = "inference"
    frame.at[row, "guide_model_splits"] = json.dumps(values)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    code, result = _run(capsys, "verify", "--dataset-dir", dataset, "--artifact-dir", artifacts)
    assert code == 1 and result["violation_count"] >= 1


def test_cli_self_check_passes_on_a_clean_database(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    build_artifacts(artifacts, n_datasets=12, months=22, seed=0)

    code, result = _run(capsys, "self-check", "--artifact-dir", artifacts, "--output-dir", tmp_path / "self_check",
                        "--device", "cpu")
    assert code == 0 and result["all_pass"] is True
    assert (tmp_path / "self_check" / "self_retrieval_summary.json").is_file()


def test_cli_self_check_returns_nonzero_when_the_database_is_broken(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    build_artifacts(artifacts, n_datasets=12, months=22, seed=0)
    connection = sqlite3.connect(str(artifacts / "vector_database.sqlite"))
    connection.execute("UPDATE metadata SET value=? WHERE key='record_count'", ("999999",))
    connection.commit()
    connection.close()

    code, result = _run(capsys, "self-check", "--artifact-dir", artifacts, "--output-dir", tmp_path / "self_check",
                        "--device", "cpu")
    assert code == 1 and result["all_pass"] is False


def test_cli_diagnose_guide_conditioning_runs_against_a_trained_checkpoint(tmp_path, capsys):
    from dataclasses import replace

    from test_self_guide_diagnostics import _raw_split_with_datasets

    dataset = tmp_path / "dataset"
    write_dataset({
        "model_train": _raw_split_with_datasets(16, seed=1, n_datasets=4),
        "model_validation": _raw_split_with_datasets(8, seed=2, n_datasets=4),
    }, dataset)
    code, _ = _run(capsys, "train", "--dataset-dir", dataset, "--output-dir", tmp_path / "run", "--epochs", 1,
                   "--batch-size", 8, "--device", "cpu", "--no-progress")
    assert code == 0

    code, result = _run(capsys, "diagnose-guide-conditioning", "--dataset-dir", dataset, "--checkpoint",
                        tmp_path / "run" / "best_model.pt", "--output-dir", tmp_path / "diagnose",
                        "--split", "model_validation", "--timesteps", 900, "--batch-size", 4, "--device", "cpu")
    assert code == 0 and result["record_count"] == 8
    frame = pd.read_csv(tmp_path / "diagnose" / "condition_usage_per_target_t900.csv", encoding="utf-8-sig")
    assert len(frame) == 8


def test_cli_build_self_reference_dataset_runs(tmp_path, capsys):
    from synthetic_artifacts import build_artifacts

    from acceleration_forecasting_v2.datasets.prepare import prepare_datasets

    artifacts, dataset = tmp_path / "artifacts", tmp_path / "dataset"
    build_artifacts(artifacts, n_datasets=12, months=22, seed=0)
    prepare_datasets(artifacts, dataset, device="cpu")

    code, result = _run(capsys, "build-self-reference-dataset", "--source-dataset-dir", dataset,
                        "--output-dataset-dir", tmp_path / "dataset_self_reference")
    assert code == 0
    assert set(result["counts"]) == {"model_train", "model_validation", "inference"}
    assert (tmp_path / "dataset_self_reference" / "model_validation" / "guide_values.npy").is_file()
