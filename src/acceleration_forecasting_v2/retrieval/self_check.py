"""自己検索の検証(SELF_RETRIEVAL_CHECK.md)。

guide検索の「配管」(埋め込みの保存・整列・除外規則)が壊れていないかを確認する。
3層構成:
  S1 保存済み自己一致(全件) — 各レコードを自身の保存埋め込みで検索し、自分が最大になるか。
  S2 再エンコード一致(標本) — 波形から再計算した埋め込みが保存値と一致するか。
  S3 自己除外(標本) — 実際のguide検索が自分・同一セグメント・同一日を返さないか。

検索の品質(意味的な類似の妥当性)は対象外。埋め込みが壊れずに保存・検索されているかだけを見る。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from acceleration_forecasting_v2.datasets.prepare import GUIDE_SOURCE_SPLITS
from .database import connect_read_only
from .pipeline import encode_waveforms
from .search import GuideIndex, SearchConfig


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(value):
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


def _chunked_unfiltered_pass(embeddings, *, chunk_size, device, tie_tolerance, topk_query_rows, topk=3):
    """L2正規化した`embeddings`に対し、行(クエリ)を`chunk_size`ごとに区切って全件との
    コサイン類似度を計算する(制限なし、split/日付/dataset_idの除外は一切かけない)。

    S1(全件)とS3の`filter_effect_rate`(除外規則を外した場合の上位3件)は、どちらも
    この「制限なしの全件ランキング」を必要とするため、1回の計算で両方に使う。

    Returns:
        self_similarity, max_other, margin, tied_count(自己を含む最大値との同点数),
        topk_by_row(`topk_query_rows`に含まれる行だけ、(similarities, indices)の辞書)。
    """
    device = _device(device)
    # 非有限(NaN/Inf)な埋め込みが1件でもあると、行列積で他の全レコードのmax_otherまで
    # NaN汚染してしまう。そのレコード自身の自己一致は別途finite_ok/norm_okで確実に
    # 検出できるので、ここでは類似度計算専用にゼロベクトルへ置き換えて汚染を防ぐ。
    finite_row = np.isfinite(embeddings).all(axis=1)
    safe_embeddings = np.where(finite_row[:, None], embeddings, 0.0)
    norms = np.linalg.norm(safe_embeddings, axis=1, keepdims=True)
    normalized = safe_embeddings / np.clip(norms, 1e-12, None)
    matrix = torch.from_numpy(normalized.astype(np.float32)).to(device)
    n = matrix.shape[0]
    self_similarity = np.zeros(n, dtype=np.float64)
    max_other = np.zeros(n, dtype=np.float64)
    tied_count = np.zeros(n, dtype=np.int64)
    topk_by_row = {}
    topk_query_rows = set(topk_query_rows or ())
    with torch.inference_mode():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            rows = torch.arange(start, end, device=device)
            scores = matrix[rows] @ matrix.T
            self_scores = scores[torch.arange(end - start), rows]
            self_similarity[start:end] = self_scores.cpu().numpy()
            row_max = scores.max(dim=1).values
            tied = (scores - row_max[:, None]).abs() <= float(tie_tolerance)
            tied_count[start:end] = tied.sum(dim=1).cpu().numpy()
            scores_excl_self = scores.clone()
            scores_excl_self[torch.arange(end - start), rows] = -2.0
            max_other[start:end] = scores_excl_self.max(dim=1).values.cpu().numpy()
            for offset, absolute_row in enumerate(range(start, end)):
                if absolute_row in topk_query_rows:
                    row_scores = scores_excl_self[offset]
                    top = torch.topk(row_scores, k=min(topk, n - 1))
                    topk_by_row[absolute_row] = (top.values.cpu().numpy(), top.indices.cpu().numpy())
    margin = self_similarity - max_other
    return self_similarity, max_other, margin, tied_count, topk_by_row


def _check_stored_self_match(connection, guide_index, *, chunk_size, tie_tolerance, norm_tolerance,
                              similarity_threshold, sample_positions, device):
    embeddings = guide_index.embeddings
    norms = np.linalg.norm(embeddings, axis=1)

    raw = dict(connection.execute("SELECT record_id, embedding FROM waveform_records").fetchall())
    alignment_ok = np.array([
        raw.get(record_id) == embeddings[index].astype(np.float32).tobytes(order="C")
        for index, record_id in enumerate(guide_index.record_ids)
    ])

    self_similarity, max_other, margin, tied_count, topk_by_row = _chunked_unfiltered_pass(
        embeddings, chunk_size=chunk_size, device=device, tie_tolerance=tie_tolerance,
        topk_query_rows=sample_positions,
    )
    norm_ok = np.abs(norms - 1.0) <= float(norm_tolerance)
    finite_ok = np.isfinite(embeddings).all(axis=1)
    similarity_ok = self_similarity >= float(similarity_threshold)
    is_maximum = self_similarity >= max_other - float(tie_tolerance)
    record_pass = similarity_ok & is_maximum & norm_ok & finite_ok
    strict_top1 = record_pass & (tied_count <= 1)

    frame = pd.DataFrame({
        "record_id": guide_index.record_ids, "trend_id": guide_index.trend_ids,
        "dataset_id": guide_index.datasets, "model_split": guide_index.splits,
        "self_similarity": self_similarity, "max_other_similarity": max_other, "margin": margin,
        "tied_count": tied_count, "embedding_norm": norms, "alignment_ok": alignment_ok,
        "record_pass": record_pass, "strict_top1": strict_top1,
    })
    frame["failure_reason"] = ""
    reasons = np.where(~similarity_ok, "self_similarity_below_threshold;", "")
    reasons = np.char.add(reasons, np.where(~is_maximum, "self_not_maximum;", ""))
    reasons = np.char.add(reasons, np.where(~norm_ok, "embedding_not_normalized;", ""))
    reasons = np.char.add(reasons, np.where(~finite_ok, "non_finite_embedding;", ""))
    reasons = np.char.add(reasons, np.where(~alignment_ok, "alignment_mismatch;", ""))
    frame["failure_reason"] = reasons
    return frame, topk_by_row


def _check_re_encoding(artifact_dir, guide_index, sample_record_ids, *, device, cosine_threshold):
    artifact_dir = Path(artifact_dir)
    full_manifest = pd.read_csv(artifact_dir / "split_manifest.csv", encoding="utf-8-sig",
                                usecols=["record_id", "waveform_index"])
    total_count = len(full_manifest)
    manifest = full_manifest.set_index("record_id")
    record_id_to_row = {record_id: index for index, record_id in enumerate(guide_index.record_ids)}
    sample_record_ids = [record_id for record_id in sample_record_ids if record_id in manifest.index]
    waveform_indices = manifest.loc[sample_record_ids, "waveform_index"].to_numpy()

    device = _device(device)
    embeddings = encode_waveforms(artifact_dir / "waveforms.bin", total_count, waveform_indices,
                                  artifact_dir / "autoencoder.pt", device=device, batch_size=512)
    stored = np.stack([guide_index.embeddings[record_id_to_row[record_id]] for record_id in sample_record_ids])
    cosine = (embeddings * stored).sum(axis=1)
    max_abs_diff = np.abs(embeddings - stored).max(axis=1)
    frame = pd.DataFrame({
        "record_id": sample_record_ids, "cosine": cosine, "max_abs_diff": max_abs_diff,
        "bit_identical": max_abs_diff == 0.0, "device": str(device),
    })
    frame["record_pass"] = frame["cosine"] >= float(cosine_threshold)

    cpu_reference_cosine_min = float(cosine.min())
    if device.type != "cpu":
        cpu_embeddings = encode_waveforms(artifact_dir / "waveforms.bin", total_count, waveform_indices,
                                          artifact_dir / "autoencoder.pt", device="cpu", batch_size=512)
        cpu_reference_cosine_min = float((cpu_embeddings * stored).sum(axis=1).min())
    return frame, cpu_reference_cosine_min


def _check_self_exclusion(guide_index, sample_positions, topk_by_row):
    rows = []
    for position in sample_positions:
        record_id = guide_index.record_ids[position]
        trend_id = guide_index.trend_ids[position]
        dataset_id = guide_index.datasets[position]
        model_split = guide_index.splits[position]
        allowed = GUIDE_SOURCE_SPLITS.get(str(model_split), {"model_train"})
        # 埋め込みが非有限などで検索クエリとして使えない場合、GuideIndex.searchはValueErrorを
        # 送出する。ここではその1件だけを検索失敗として記録し(record_pass=False)、
        # 全体の実行は継続する(1件の破損で自己除外の検証全体が止まらないようにする)。
        try:
            results = guide_index.search(
                guide_index.embeddings[position], query_date=guide_index.dates[position],
                query_dataset_id=dataset_id, query_current=float(guide_index.current[position]),
                query_bin_start_m=float(guide_index.bin_starts[position]), allowed_splits=allowed,
                config=SearchConfig(min_valid_months=0, max_current_difference=None),
            )
            search_error = ""
        except ValueError as exc:
            results = []
            search_error = str(exc)

        contains_self = any(item["record_id"] == record_id for item in results)
        contains_same_dataset = any(item["dataset_id"] == dataset_id for item in results)
        contains_same_date = any(item["measurement_date"] == guide_index.dates[position] for item in results)

        unfiltered_indices = topk_by_row[position][1] if position in topk_by_row else np.asarray([], dtype=int)
        unfiltered_top3_has_same_dataset = bool(np.isin(guide_index.datasets[unfiltered_indices], [dataset_id]).any())

        rows.append({
            "record_id": record_id, "trend_id": trend_id, "dataset_id": dataset_id,
            "returned_count": len(results), "contains_self": contains_self,
            "contains_same_dataset": contains_same_dataset, "contains_same_date": contains_same_date,
            "unfiltered_top3_has_same_dataset": unfiltered_top3_has_same_dataset,
            "search_error": search_error,
            "record_pass": not search_error and not (contains_self or contains_same_dataset or contains_same_date),
        })
    return pd.DataFrame(rows)


def self_retrieval_check(artifact_dir, output_dir, *, similarity_threshold=0.999999,
                          tie_tolerance=1e-6, norm_tolerance=1e-5,
                          re_encode_cosine_threshold=0.999999, chunk_size=250,
                          sample_size=20000, seed=0, device=None) -> dict:
    """guide検索の自己一致・再エンコード一致・自己除外を検証する(SELF_RETRIEVAL_CHECK.md)。

    Returns:
        summary dict。`all_pass`が全体の合否(record_pass全件 ∧ D ∧ E ∧ F ∧ G ∧ H)。
        `output_dir`直下にCSV/JSONを書き出す。
    """
    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = artifact_dir / "vector_database.sqlite"
    sha_before = _sha256_file(db_path)
    wal_before = (db_path.with_name(db_path.name + "-wal")).exists() or (db_path.with_name(db_path.name + "-journal")).exists()

    read_connection = connect_read_only(db_path)
    metadata = dict(read_connection.execute("SELECT key, value FROM metadata").fetchall())
    guide_index = GuideIndex(read_connection)

    rng = np.random.default_rng(seed)
    record_count = len(guide_index.record_ids)
    sample_size = min(int(sample_size), record_count)
    sample_positions = rng.choice(record_count, size=sample_size, replace=False)
    sample_record_ids = [guide_index.record_ids[position] for position in sample_positions]

    s1_frame, topk_by_row = _check_stored_self_match(
        read_connection, guide_index, chunk_size=chunk_size, tie_tolerance=tie_tolerance,
        norm_tolerance=norm_tolerance, similarity_threshold=similarity_threshold,
        sample_positions=set(sample_positions.tolist()), device=device,
    )
    read_connection.close()
    sha_after = _sha256_file(db_path)
    wal_after = (db_path.with_name(db_path.name + "-wal")).exists() or (db_path.with_name(db_path.name + "-journal")).exists()

    s2_frame, cpu_reference_cosine_min = _check_re_encoding(
        artifact_dir, guide_index, sample_record_ids, device=device, cosine_threshold=re_encode_cosine_threshold,
    )
    s3_frame = _check_self_exclusion(guide_index, sample_positions.tolist(), topk_by_row)

    s1_frame.to_csv(output_dir / "self_retrieval_results.csv", index=False, encoding="utf-8-sig")
    s1_frame.loc[~s1_frame["record_pass"]].to_csv(output_dir / "self_retrieval_failures.csv", index=False, encoding="utf-8-sig")
    s2_frame.to_csv(output_dir / "re_encode_results.csv", index=False, encoding="utf-8-sig")
    s3_frame.to_csv(output_dir / "self_exclusion_results.csv", index=False, encoding="utf-8-sig")

    record_count_matches = record_count == int(metadata.get("record_count", -1))
    record_ids_unique = len(set(guide_index.record_ids.tolist())) == record_count
    alignment_pass = bool(s1_frame["alignment_ok"].all())
    db_unchanged = sha_before == sha_after and not wal_before and not wal_after
    s2_pass = bool(s2_frame["record_pass"].all())
    s3_pass = bool(s3_frame["record_pass"].all())
    s1_pass = bool(s1_frame["record_pass"].all())
    all_pass = bool(s1_pass and alignment_pass and db_unchanged and record_count_matches and record_ids_unique and s2_pass and s3_pass)

    summary = {
        "database": str(db_path), "database_sha256_before": sha_before, "database_sha256_after": sha_after,
        "database_unchanged": db_unchanged, "record_count": record_count,
        "record_count_matches_metadata": record_count_matches, "record_ids_unique": record_ids_unique,
        "model_split_counts": {str(key): int(value) for key, value in pd.Series(guide_index.splits).value_counts().items()},
        "embedding_dim": int(metadata.get("embedding_dim", -1)),
        "s1": {
            "checked": int(len(s1_frame)), "pass_count": int(s1_frame["record_pass"].sum()),
            "pass_rate": float(s1_frame["record_pass"].mean()),
            "strict_top1_rate": float(s1_frame["strict_top1"].mean()),
            "tied_records": int((s1_frame["tied_count"] > 1).sum()),
            "self_similarity_min": float(s1_frame["self_similarity"].min()),
            "self_similarity_mean": float(s1_frame["self_similarity"].mean()),
            "self_similarity_max": float(s1_frame["self_similarity"].max()),
            "margin_min": float(s1_frame["margin"].min()), "margin_p1": float(np.percentile(s1_frame["margin"], 1)),
            "margin_median": float(s1_frame["margin"].median()),
            "alignment_failures": int((~s1_frame["alignment_ok"]).sum()),
        },
        "s2": {
            "sampled": int(len(s2_frame)), "cosine_min": float(s2_frame["cosine"].min()),
            "bit_identical_rate": float(s2_frame["bit_identical"].mean()),
            "max_abs_diff_max": float(s2_frame["max_abs_diff"].max()),
            "device": str(s2_frame["device"].iloc[0]) if len(s2_frame) else None,
            "cpu_reference_cosine_min": cpu_reference_cosine_min,
        },
        "s3": {
            "sampled": int(len(s3_frame)), "self_returned": int(s3_frame["contains_self"].sum()),
            "same_dataset_returned": int(s3_frame["contains_same_dataset"].sum()),
            "same_date_returned": int(s3_frame["contains_same_date"].sum()),
            "filter_effect_rate": float(s3_frame["unfiltered_top3_has_same_dataset"].mean()),
        },
        "thresholds": {
            "similarity": float(similarity_threshold), "tie_tolerance": float(tie_tolerance),
            "norm_tolerance": float(norm_tolerance), "re_encode_cosine": float(re_encode_cosine_threshold),
        },
        "all_pass": all_pass,
    }
    (output_dir / "self_retrieval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
