"""retrieval.self_check の検証(SELF_RETRIEVAL_CHECK.md ステップ1: S1/S2/S3)。

`build_artifacts`で作った合成DBに対し、まず正常系がall_pass=Trueになることを確認し、
その後は「コピーをわざと壊して検出できるか」という検出力のテストを中心に置く
(`test_split_leakage.py`と同じ流儀)。
"""

import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest

from synthetic_artifacts import build_artifacts

from acceleration_forecasting_v2.retrieval import search as search_module
from acceleration_forecasting_v2.retrieval.constants import SAMPLES_PER_BIN
from acceleration_forecasting_v2.retrieval.search import GuideIndex
from acceleration_forecasting_v2.retrieval.self_check import (
    _check_stored_self_match,
    _chunked_unfiltered_pass,
    self_retrieval_check,
)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("self_check") / "artifacts"
    build_artifacts(artifact_dir, n_datasets=12, months=22, seed=0)
    return artifact_dir


def _copy_artifacts(artifacts, tmp_path):
    copy = tmp_path / "artifacts"
    shutil.copytree(artifacts, copy)
    return copy


# --- 正常系 ---------------------------------------------------------------

def test_clean_database_passes_everything(artifacts, tmp_path):
    output_dir = tmp_path / "self_check"
    summary = self_retrieval_check(artifacts, output_dir, device="cpu", seed=1)

    assert summary["all_pass"] is True
    assert summary["s1"]["pass_rate"] == 1.0
    assert summary["s1"]["alignment_failures"] == 0
    assert summary["s2"]["cosine_min"] >= 0.999999
    assert summary["s3"]["self_returned"] == 0
    assert summary["s3"]["same_dataset_returned"] == 0
    assert summary["s3"]["same_date_returned"] == 0
    assert summary["database_unchanged"] is True
    assert summary["record_count_matches_metadata"] is True
    assert summary["record_ids_unique"] is True
    assert 0.0 <= summary["s3"]["filter_effect_rate"] <= 1.0

    for name in ("self_retrieval_results.csv", "self_retrieval_failures.csv",
                 "re_encode_results.csv", "self_exclusion_results.csv", "self_retrieval_summary.json"):
        assert (output_dir / name).is_file()
    failures = pd.read_csv(output_dir / "self_retrieval_failures.csv", encoding="utf-8-sig")
    assert failures.empty


def test_database_file_is_untouched_by_the_check(artifacts, tmp_path):
    db_path = artifacts / "vector_database.sqlite"
    before = db_path.read_bytes()
    self_retrieval_check(artifacts, tmp_path / "out", device="cpu")
    after = db_path.read_bytes()
    assert before == after
    assert not db_path.with_name(db_path.name + "-wal").exists()
    assert not db_path.with_name(db_path.name + "-journal").exists()


def test_sample_size_larger_than_database_uses_every_record(artifacts, tmp_path):
    summary = self_retrieval_check(artifacts, tmp_path / "out", device="cpu", sample_size=10_000_000)
    assert summary["s2"]["sampled"] == summary["record_count"]
    assert summary["s3"]["sampled"] == summary["record_count"]


def test_chunk_size_does_not_change_the_result(artifacts, tmp_path):
    a = self_retrieval_check(artifacts, tmp_path / "a", device="cpu", chunk_size=7, seed=3)
    b = self_retrieval_check(artifacts, tmp_path / "b", device="cpu", chunk_size=250, seed=3)
    # fp32行列積はchunk分割の順序でわずかに丸め誤差が変わるため、絶対誤差で比較する。
    assert a["s1"]["self_similarity_min"] == pytest.approx(b["s1"]["self_similarity_min"], abs=1e-5)
    assert a["s1"]["margin_min"] == pytest.approx(b["s1"]["margin_min"], abs=1e-5)
    assert a["s1"]["tied_records"] == b["s1"]["tied_records"]
    assert a["s3"]["filter_effect_rate"] == pytest.approx(b["s3"]["filter_effect_rate"])


def test_chunked_pass_is_invariant_to_a_chunk_size_that_does_not_divide_n():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(37, 6)).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    rows = {0, 10, 36}
    small = _chunked_unfiltered_pass(embeddings, chunk_size=5, device="cpu", tie_tolerance=1e-6, topk_query_rows=rows)
    whole = _chunked_unfiltered_pass(embeddings, chunk_size=37, device="cpu", tie_tolerance=1e-6, topk_query_rows=rows)
    np.testing.assert_allclose(small[0], whole[0], atol=1e-6)  # self_similarity
    np.testing.assert_allclose(small[1], whole[1], atol=1e-6)  # max_other
    np.testing.assert_array_equal(small[3], whole[3])  # tied_count
    for row in rows:
        np.testing.assert_allclose(small[4][row][0], whole[4][row][0], atol=1e-6)
        np.testing.assert_array_equal(small[4][row][1], whole[4][row][1])


# --- 異常注入(検出力) ------------------------------------------------------

def test_unnormalized_embedding_is_flagged(artifacts, tmp_path):
    copy = _copy_artifacts(artifacts, tmp_path)
    connection = sqlite3.connect(str(copy / "vector_database.sqlite"))
    record_id, blob = connection.execute("SELECT record_id, embedding FROM waveform_records LIMIT 1").fetchone()
    embedding = np.frombuffer(blob, dtype=np.float32).copy() * 2.0
    connection.execute("UPDATE waveform_records SET embedding=? WHERE record_id=?", (embedding.tobytes(), record_id))
    connection.commit()
    connection.close()

    summary = self_retrieval_check(copy, tmp_path / "out", device="cpu")
    assert summary["all_pass"] is False
    failures = pd.read_csv(tmp_path / "out" / "self_retrieval_failures.csv", encoding="utf-8-sig")
    row = failures.loc[failures["record_id"].astype(str) == record_id].iloc[0]
    assert "embedding_not_normalized" in row["failure_reason"]


def test_non_finite_embedding_is_flagged_without_corrupting_other_records(artifacts, tmp_path):
    copy = _copy_artifacts(artifacts, tmp_path)
    connection = sqlite3.connect(str(copy / "vector_database.sqlite"))
    record_id, blob = connection.execute("SELECT record_id, embedding FROM waveform_records LIMIT 1").fetchone()
    dim = len(np.frombuffer(blob, dtype=np.float32))
    embedding = np.full(dim, np.nan, dtype=np.float32)
    connection.execute("UPDATE waveform_records SET embedding=? WHERE record_id=?", (embedding.tobytes(), record_id))
    connection.commit()
    connection.close()

    summary = self_retrieval_check(copy, tmp_path / "out", device="cpu")
    assert summary["all_pass"] is False
    results = pd.read_csv(tmp_path / "out" / "self_retrieval_results.csv", encoding="utf-8-sig")
    tampered_row = results.loc[results["record_id"].astype(str) == record_id].iloc[0]
    assert not tampered_row["record_pass"]
    # 汚染を防ぐ実装なら、他のレコードの大半は影響を受けない。
    others = results.loc[results["record_id"].astype(str) != record_id]
    assert others["record_pass"].mean() > 0.9


def test_duplicate_embedding_is_a_tie_at_s1_but_still_fails_re_encoding(artifacts, tmp_path):
    # record_bの保存済み埋め込みをrecord_aのものにすり替える。これはS1(自己一致)だけを見ると
    # 「同点」として許容されるべきだが、record_bの実際の波形を再エンコードすればrecord_a由来の
    # 値とは一致しないはずなので、正当な偶然の一致ではなく破損としてS2で検出されるべき。
    copy = _copy_artifacts(artifacts, tmp_path)
    connection = sqlite3.connect(str(copy / "vector_database.sqlite"))
    (record_a, blob_a), (record_b, _) = connection.execute(
        "SELECT record_id, embedding FROM waveform_records LIMIT 2"
    ).fetchall()
    connection.execute("UPDATE waveform_records SET embedding=? WHERE record_id=?", (blob_a, record_b))
    connection.commit()
    connection.close()

    summary = self_retrieval_check(copy, tmp_path / "out", device="cpu", sample_size=10_000_000)
    results = pd.read_csv(tmp_path / "out" / "self_retrieval_results.csv", encoding="utf-8-sig")
    tied_rows = results.loc[results["record_id"].astype(str).isin([record_a, record_b])]
    assert tied_rows["record_pass"].all()
    assert (tied_rows["tied_count"] > 1).all()
    assert summary["s1"]["tied_records"] >= 2
    assert summary["s1"]["strict_top1_rate"] < 1.0
    assert summary["all_pass"] is False  # S2が食い違いを検出するため


def test_tampered_waveform_is_flagged_by_re_encoding(artifacts, tmp_path):
    copy = _copy_artifacts(artifacts, tmp_path)
    manifest = pd.read_csv(copy / "split_manifest.csv", encoding="utf-8-sig")
    developed = manifest.loc[manifest["model_split"].isin(["model_train", "model_validation"])].iloc[0]

    data = np.memmap(copy / "waveforms.bin", dtype=np.float32, mode="r+", shape=(len(manifest), SAMPLES_PER_BIN))
    data[int(developed["waveform_index"])] += 50.0
    data.flush()
    del data

    summary = self_retrieval_check(copy, tmp_path / "out", device="cpu", sample_size=10_000_000)
    assert summary["all_pass"] is False
    assert summary["s2"]["cosine_min"] < 0.999999
    results = pd.read_csv(tmp_path / "out" / "re_encode_results.csv", encoding="utf-8-sig")
    failing = results.loc[~results["record_pass"]]
    assert str(developed["record_id"]) in failing["record_id"].astype(str).tolist()


def test_record_count_mismatch_with_metadata_is_flagged(artifacts, tmp_path):
    copy = _copy_artifacts(artifacts, tmp_path)
    connection = sqlite3.connect(str(copy / "vector_database.sqlite"))
    connection.execute("UPDATE metadata SET value=? WHERE key='record_count'", (str(999999),))
    connection.commit()
    connection.close()

    summary = self_retrieval_check(copy, tmp_path / "out", device="cpu")
    assert summary["record_count_matches_metadata"] is False
    assert summary["all_pass"] is False


def test_broken_self_exclusion_is_detected(artifacts, tmp_path, monkeypatch):
    original_search = search_module.GuideIndex.search

    def leaking_search(self, query_embeddings, *, query_date, query_dataset_id, query_bin_start_m, **kwargs):
        results = original_search(self, query_embeddings, query_date=query_date, query_dataset_id=query_dataset_id,
                                  query_bin_start_m=query_bin_start_m, **kwargs)
        match = np.flatnonzero(
            (self.dates == str(query_date)) & (self.datasets == str(query_dataset_id)) &
            (np.abs(self.bin_starts.astype(np.float64) - float(query_bin_start_m)) < 1e-6)
        )
        if len(match):
            index = int(match[0])
            results = [{
                "record_id": self.record_ids[index], "dataset_id": self.datasets[index],
                "measurement_date": str(self.dates[index]), "model_split": self.splits[index],
            }] + results
        return results

    monkeypatch.setattr(search_module.GuideIndex, "search", leaking_search)
    summary = self_retrieval_check(artifacts, tmp_path / "out", device="cpu")
    assert summary["all_pass"] is False
    assert summary["s3"]["self_returned"] > 0


def test_alignment_check_flags_a_guide_index_stale_relative_to_the_connection(artifacts, tmp_path):
    copy = _copy_artifacts(artifacts, tmp_path)
    connection = sqlite3.connect(str(copy / "vector_database.sqlite"))
    guide_index = GuideIndex(connection)
    record_id = str(guide_index.record_ids[0])
    dim = guide_index.embeddings.shape[1]

    # GuideIndexが読み込んだ後にDBが書き換わる状況(並行書き込み等)を模擬する。
    connection.execute(
        "UPDATE waveform_records SET embedding=? WHERE record_id=?",
        (np.zeros(dim, dtype=np.float32).tobytes(), record_id),
    )
    connection.commit()

    frame, _ = _check_stored_self_match(
        connection, guide_index, chunk_size=8, tie_tolerance=1e-6, norm_tolerance=1e-5,
        similarity_threshold=0.999999, sample_positions=set(), device="cpu",
    )
    connection.close()
    row = frame.loc[frame["record_id"].astype(str) == record_id].iloc[0]
    assert row["alignment_ok"] == False  # noqa: E712 (numpy boolのため`is False`は使わない)
