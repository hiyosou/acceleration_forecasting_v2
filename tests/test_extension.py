"""忘却因子付き線形近似による正解の延長(`datasets.extension`、2026-10-22)のテスト。"""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
import torch

from synthetic_artifacts import build_artifacts, start_date

from acceleration_forecasting_v2.common.constants import PHYSICAL_MAX, PHYSICAL_MIN
from acceleration_forecasting_v2.datasets.anchors import select_anchors_for_segment
from acceleration_forecasting_v2.datasets.extension import (
    apply_extensions_to_catalog,
    build_cycle_extensions,
    build_extended_artifacts,
    cycle_start,
    extend_cycle,
    month_index,
    strip_synthetic,
)
from acceleration_forecasting_v2.datasets.normalization import Normalization
from acceleration_forecasting_v2.datasets.prepare import SPLITS, prepare_datasets
from acceleration_forecasting_v2.datasets.verify import verify_dataset_leakage
from acceleration_forecasting_v2.evaluation.evaluate import load_target_masks
from acceleration_forecasting_v2.evaluation.forgetting_factor import (
    DEFAULT_LAMBDAS,
    forgetting_factor_line,
    predict_forgetting_factor,
)
from acceleration_forecasting_v2.training.train import validation_loss


def _cycle(months=10, start="2023-01-10", every_days=10, slope=0.002, base=1.0):
    dates = pd.date_range(pd.Timestamp(start) + pd.Timedelta(days=5),
                          pd.Timestamp(start) + pd.DateOffset(months=months), freq=f"{every_days}D")
    values = base + slope * (dates - dates[0]).days.to_numpy(float)
    return dates, values


# --- extend_cycle ----------------------------------------------------------------------

def test_lambda_one_without_fluctuation_matches_the_least_squares_line():
    dates, values = _cycle()
    values = values + np.random.default_rng(0).normal(0, 0.05, len(values))
    frame, lam = extend_cycle(dates, values, pd.Timestamp("2023-01-10"), rng=np.random.default_rng(0),
                              lam=1.0, fluctuation=False)
    x = (dates - dates.max()).days.to_numpy(float)
    slope, intercept = np.polyfit(x, values, 1)
    expected = intercept + slope * (pd.DatetimeIndex(frame["month"]) - dates.max()).days.to_numpy(float)
    assert lam == 1.0
    np.testing.assert_allclose(frame["value"], np.clip(expected, PHYSICAL_MIN, PHYSICAL_MAX), rtol=1e-6)


def test_months_run_from_the_month_after_maintenance_to_24_months_later():
    dates, values = _cycle(months=10)
    frame, _ = extend_cycle(dates, values, pd.Timestamp("2023-01-10"), rng=np.random.default_rng(0))
    months = pd.DatetimeIndex(frame["month"])
    assert months[0] == pd.Timestamp("2023-02-01")
    assert months[-1] == pd.Timestamp("2025-01-01")
    assert len(frame) == 24


def test_kinds_separate_observed_gap_and_extension_months():
    dates, values = _cycle(months=10)
    # 2023-05前後(±15日)の計測を抜いて、途中の欠測月を作る
    keep = np.abs((dates - pd.Timestamp("2023-05-01")).days) > 15
    frame, _ = extend_cycle(dates[keep], values[keep], pd.Timestamp("2023-01-10"), rng=np.random.default_rng(0))
    kind = dict(zip(pd.DatetimeIndex(frame["month"]).strftime("%Y-%m"), frame["kind"]))
    assert kind["2023-05"] == "gap"
    assert kind["2023-04"] == "observed"
    last_month = dates[keep].max().strftime("%Y-%m")
    after = [month for month in kind if month > last_month]
    assert after and all(kind[month] == "extension" for month in after)
    assert all(kind[month] != "extension" for month in kind if month <= last_month)


def test_cycles_shorter_than_six_months_after_maintenance_are_not_extended():
    dates, values = _cycle(months=4)
    frame, lam = extend_cycle(dates, values, pd.Timestamp("2023-01-10"), rng=np.random.default_rng(0))
    assert frame.empty and lam is None


def test_values_are_clipped_to_physical_bounds():
    dates, values = _cycle(months=8, slope=0.03)
    frame, _ = extend_cycle(dates, values, pd.Timestamp("2023-01-10"), rng=np.random.default_rng(0))
    assert frame["value"].max() <= PHYSICAL_MAX and frame["value"].min() >= PHYSICAL_MIN
    assert frame["value"].max() == PHYSICAL_MAX


def test_lambda_is_drawn_between_point_nine_and_one():
    dates, values = _cycle()
    lams = [extend_cycle(dates, values, pd.Timestamp("2023-01-10"), rng=np.random.default_rng(seed))[1]
            for seed in range(50)]
    assert min(lams) >= 0.90 and max(lams) <= 1.00 and np.std(lams) > 0.01


def _raw_rows(dataset_ids=("A", "B"), maintenance=("2023-01-10", "")):
    frames, rng = [], np.random.default_rng(0)
    for dataset_id, date in zip(dataset_ids, maintenance):
        dates, values = _cycle()
        values = values + rng.normal(0, 0.05, len(values))  # 揺らぎとλで値が変わるように、直線からずらす
        frames.append(pd.DataFrame({"dataset_id": dataset_id, "measurement_date": dates, "acc_z_max": values,
                                    "previous_maintenance_date": date}))
    return pd.concat(frames, ignore_index=True)


def test_same_seed_gives_the_same_series_and_cycles_get_their_own_draws():
    raw = _raw_rows()
    splits = {"A": "model_train", "B": "inference"}
    cycles1, ext1 = build_cycle_extensions(raw, splits, seed=42)
    cycles2, ext2 = build_cycle_extensions(raw, splits, seed=42)
    pd.testing.assert_frame_equal(ext1, ext2)
    assert cycles1["lambda"].nunique() == 2
    _, ext3 = build_cycle_extensions(raw, splits, seed=7)
    assert not np.allclose(ext1["value"], ext3["value"])


def test_cycle_start_is_the_maintenance_date_or_the_first_measurement():
    raw = _raw_rows()
    assert cycle_start(raw.loc[raw["dataset_id"] == "A"]) == pd.Timestamp("2023-01-10")
    assert cycle_start(raw.loc[raw["dataset_id"] == "B"]) == pd.Timestamp("2023-01-15")
    cycles, _ = build_cycle_extensions(raw, {"A": "model_train", "B": "model_train"})
    assert cycles.set_index("dataset_id")["cycle_start"].to_dict() == {"A": "2023-01-10", "B": "2023-01-15"}


def test_datasets_without_a_split_are_skipped():
    cycles, extensions = build_cycle_extensions(_raw_rows(), {"A": "model_train"})
    assert cycles["dataset_id"].tolist() == ["A"] and set(extensions["dataset_id"]) == {"A"}


# --- catalog ------------------------------------------------------------------------------

def _catalog_row(anchor, mask):
    values = [1.5 if flag else None for flag in mask]
    dates = [f"2023-{offset + 2:02d}-03" if flag else None for offset, flag in enumerate(mask)]
    return {"trend_id": "t1", "dataset_id": "A", "measurement_date": anchor, "direction": "U",
            "bin_start_m": 0.0, "bin_end_m": 100.0, "current_acc_z_max": 1.5,
            "future_values": json.dumps(values), "future_mask": json.dumps(mask), "selected_dates": json.dumps(dates)}


def test_only_masked_months_are_filled_and_marked_as_synthetic():
    mask = [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    catalog = pd.DataFrame([_catalog_row("2023-01-20", mask)])
    cycles = pd.DataFrame({"dataset_id": ["A"], "cycle_start": ["2023-01-10"]})
    months = pd.date_range("2023-02-01", "2025-01-01", freq="MS")
    extensions = pd.DataFrame({"dataset_id": "A", "month": months.strftime("%Y-%m-%d"), "value": 3.0})
    out = apply_extensions_to_catalog(catalog, cycles, extensions)
    values = json.loads(out.loc[0, "future_values"])
    synthetic = json.loads(out.loc[0, "future_synthetic_mask"])
    assert json.loads(out.loc[0, "future_mask"]) == [1] * 12
    assert synthetic == [0 if flag else 1 for flag in mask]
    assert [value for value, flag in zip(values, mask) if flag] == [1.5] * sum(mask)
    assert [value for value, flag in zip(values, mask) if not flag] == [3.0] * (12 - sum(mask))
    assert json.loads(out.loc[0, "selected_dates"])[2] == "2023-04-01"
    assert out.loc[0, "cycle_start"] == "2023-01-10"


def test_months_beyond_the_extension_stay_missing():
    catalog = pd.DataFrame([_catalog_row("2023-01-20", [1] * 4 + [0] * 8)])
    cycles = pd.DataFrame({"dataset_id": ["A"], "cycle_start": ["2023-01-10"]})
    extensions = pd.DataFrame({"dataset_id": "A", "month": ["2023-06-01", "2023-07-01"], "value": 2.0})
    out = apply_extensions_to_catalog(catalog, cycles, extensions)
    assert json.loads(out.loc[0, "future_mask"]) == [1] * 6 + [0] * 6


def test_strip_synthetic_restores_the_original_catalog_row():
    mask = [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    catalog = pd.DataFrame([_catalog_row("2023-01-20", mask)])
    cycles = pd.DataFrame({"dataset_id": ["A"], "cycle_start": ["2023-01-10"]})
    months = pd.date_range("2023-02-01", "2025-01-01", freq="MS")
    extensions = pd.DataFrame({"dataset_id": "A", "month": months.strftime("%Y-%m-%d"), "value": 3.0})
    filled = apply_extensions_to_catalog(catalog, cycles, extensions)
    restored = strip_synthetic(filled, pd.Series([False]))
    for column in ("future_values", "future_mask", "selected_dates"):
        assert json.loads(restored.loc[0, column]) == json.loads(catalog.loc[0, column])
    kept = strip_synthetic(filled, pd.Series([True]))
    assert kept.loc[0, "future_values"] == filled.loc[0, "future_values"]


# --- anchors -----------------------------------------------------------------------------

def _segment(months=14, start="2023-01-10"):
    rows = []
    for i in range(months):
        date = pd.Timestamp("2023-01-15") + pd.DateOffset(months=i)
        rows.append({"trend_id": f"t{i}", "dataset_id": "A", "measurement_date": date, "direction": "U",
                     "bin_start_m": 0.0, "bin_end_m": 100.0, "current_acc_z_max": 1.0 + 0.01 * i,
                     "future_values": json.dumps([1.0] * 12), "future_mask": json.dumps([1] * 12),
                     "future_synthetic_mask": json.dumps([0] * 11 + [1]), "cycle_start": start})
    return pd.DataFrame(rows)


def test_maintenance_rule_keeps_only_anchors_five_months_after_maintenance():
    segment = _segment()
    without = select_anchors_for_segment(segment, split="model_train", min_history_months=3, min_target_months=8)
    with_rule = select_anchors_for_segment(segment, split="model_train", min_history_months=3, min_target_months=8,
                                           min_months_after_maintenance=5)
    assert min(anchor.months_after_maintenance for anchor in without) == 2
    assert [anchor.months_after_maintenance for anchor in with_rule] == list(range(5, 14))
    assert with_rule[0].future_synthetic_mask.tolist() == [0] * 11 + [1]


def test_max_months_after_maintenance_masks_targets_past_the_cap():
    # 補修から数えて24か月を超える正解の月は、実測も合成も欠測になる
    segment = _segment(months=30)
    capped = select_anchors_for_segment(segment, split="model_train", min_history_months=3, min_target_months=1,
                                        max_months_after_maintenance=24)
    free = select_anchors_for_segment(segment, split="model_train", min_history_months=3, min_target_months=1)
    by_month = {anchor.months_after_maintenance: anchor for anchor in capped}
    # 補修後20か月の起点は目標月が21〜32か月目なので、24か月目までの4か月だけが残る
    anchor = by_month[20]
    assert anchor.future_mask.tolist() == [1] * 4 + [0] * 8
    assert anchor.valid_future_months == 4
    assert anchor.future_synthetic_mask.tolist() == [0] * 12
    assert np.isnan(anchor.future_values[4:]).all()
    # 補修後12か月以内の起点は目標月が24か月目に届かないので、上限なしと同じ
    for month in range(2, 13):
        assert by_month[month].future_mask.tolist() == next(
            a for a in free if a.months_after_maintenance == month).future_mask.tolist()


def test_max_months_after_maintenance_counts_toward_min_target_months():
    segment = _segment(months=30)
    capped = select_anchors_for_segment(segment, split="model_train", min_history_months=3, min_target_months=8,
                                        max_months_after_maintenance=24)
    assert max(anchor.months_after_maintenance for anchor in capped) == 16  # 24 - 8 = 16か月目まで


def test_max_months_after_maintenance_needs_the_cycle_start_column():
    segment = _segment().drop(columns="cycle_start")
    with pytest.raises(ValueError, match="cycle_start"):
        select_anchors_for_segment(segment, split="model_train", min_history_months=3, min_target_months=8,
                                   max_months_after_maintenance=24)


def test_eighteen_month_history_is_a_growing_window_that_ends_at_the_anchor():
    # 補修より前のデータは同じdataset_idに無いので、窓の先頭側は欠測になる(窓の長さ=補修からの経過)
    segment = _segment(months=30)
    # 計測日を10日にずらす(15日だと2月だけ月初から14日で前月と同距離になり、窓の長さで割り当てが変わる)
    segment["measurement_date"] = segment["measurement_date"] - pd.Timedelta(days=5)
    six = select_anchors_for_segment(segment, split="model_train", min_history_months=3, min_target_months=1,
                                     history_months=6)
    wide = select_anchors_for_segment(segment, split="model_train", min_history_months=3, min_target_months=1,
                                      history_months=18)
    assert len(wide[0].history_values) == 18
    for narrow, anchor in zip(six, wide):
        assert anchor.trend_id == narrow.trend_id
        np.testing.assert_array_equal(anchor.history_values[-6:], narrow.history_values)  # 末尾6か月は6か月版と同じ
        np.testing.assert_array_equal(anchor.history_mask[-6:], narrow.history_mask)
        n_valid = int(anchor.history_mask.sum())
        assert anchor.history_mask[:18 - n_valid].sum() == 0 and anchor.history_mask[18 - n_valid:].all()


def test_maintenance_rule_needs_the_cycle_start_column():
    segment = _segment().drop(columns="cycle_start")
    with pytest.raises(ValueError, match="cycle_start"):
        select_anchors_for_segment(segment, split="model_train", min_history_months=3, min_target_months=8,
                                   min_months_after_maintenance=5)


# --- validation loss ---------------------------------------------------------------------

class _MaskProcess:
    steps = 10

    def __init__(self):
        self.model = torch.nn.Identity()

    def per_record_loss(self, batch, noise=None, timesteps=None):
        mask = batch["target_mask"]
        return (batch["target"] * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def test_validation_loss_also_reports_the_loss_on_real_months_only():
    items = [
        {"target": torch.tensor([1.0, 3.0]), "target_mask": torch.tensor([1.0, 1.0]),
         "target_synthetic_mask": torch.tensor([0.0, 1.0]), "segment_weight": torch.tensor(1.0)},
        {"target": torch.tensor([5.0, 7.0]), "target_mask": torch.tensor([1.0, 1.0]),
         "target_synthetic_mask": torch.tensor([1.0, 1.0]), "segment_weight": torch.tensor(1.0)},
    ]
    loader = torch.utils.data.DataLoader(items, batch_size=2)
    result = validation_loss(_MaskProcess(), loader, torch.device("cpu"))
    assert result["loss"] == pytest.approx((2.0 + 6.0) / 2)
    assert result["real_loss"] == pytest.approx(1.0)  # 実測の月が無い2件目は除く


# --- end to end ----------------------------------------------------------------------------

def _write_raw_trend_dir(trend_dir, n_datasets, months, staggered_start=False):
    """`build_artifacts`と同じ計測を、元のtrend CSVの形で書き出す。"""
    trend_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n_datasets):
        bin_start = 2000.0 + 1000.0 * k
        dates = [start_date(k, staggered_start) + pd.DateOffset(months=i) for i in range(months)]
        pd.DataFrame([{
            "dataset_id": f"seg{k:02d}", "direction": "U", "bin_start_m": bin_start, "bin_end_m": bin_start + 100.0,
            "segment_index": 1, "measurement_date": d.strftime("%Y-%m-%d"), "acc_z_max": 1.0 + 0.01 * k + 0.02 * i,
            "measured_at": d.strftime("%Y-%m-%d"), "previous_maintenance_date": "",
            "maintenance_type": "", "maintenance_description": "",
        } for i, d in enumerate(dates)]).to_csv(trend_dir / f"seg{k:02d}.csv", index=False, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def extended(tmp_path_factory):
    root = tmp_path_factory.mktemp("extension")
    artifact_dir, trend_dir, extended_dir = root / "artifacts", root / "trends", root / "extended"
    manifest = build_artifacts(artifact_dir, n_datasets=12, months=22, seed=0)
    _write_raw_trend_dir(trend_dir, 12, 22)
    summary = build_extended_artifacts(artifact_dir, extended_dir, trend_dir=trend_dir, seed=42)
    plain = prepare_datasets(artifact_dir, root / "plain", device="cpu", inference_max_per_segment=10)
    options = dict(device="cpu", inference_max_per_segment=10, trend_catalog_path=extended_dir / "trend_catalog.csv",
                   database_path=extended_dir / "vector_database.sqlite", min_months_after_maintenance=5)
    ext = prepare_datasets(artifact_dir, root / "ext", extend_splits=("model_train", "model_validation"),
                           maintenance_rule_splits=("model_train", "model_validation"), **options)
    evalext = prepare_datasets(artifact_dir, root / "evalext", extend_splits=SPLITS, maintenance_rule_splits=SPLITS,
                               **options)
    return dict(root=root, artifact_dir=artifact_dir, extended_dir=extended_dir, manifest=manifest, summary=summary,
                plain=plain, ext=ext, evalext=evalext)


def test_extended_database_holds_synthetic_futures_only_for_development_rows(extended):
    connection = sqlite3.connect(extended["extended_dir"] / "vector_database.sqlite")
    rows = pd.read_sql_query("SELECT model_split, future_mask, future_synthetic_mask, extension_lambda FROM trends",
                             connection)
    connection.close()
    assert set(rows["model_split"]) <= {"model_train", "model_validation"}
    synthetic = rows["future_synthetic_mask"].map(lambda text: sum(json.loads(text)))
    assert synthetic.sum() > 0
    assert rows["extension_lambda"].between(0.9, 1.0).all()
    original = sqlite3.connect(extended["artifact_dir"] / "vector_database.sqlite")
    columns = [row[1] for row in original.execute("PRAGMA table_info(trends)")]
    original.close()
    assert "future_synthetic_mask" not in columns  # 元のDBは変えない


def test_inference_without_extension_matches_the_plain_build(extended):
    root = extended["root"]
    plain = pd.read_csv(root / "plain" / "inference" / "metadata.csv", encoding="utf-8-sig")
    ext = pd.read_csv(root / "ext" / "inference" / "metadata.csv", encoding="utf-8-sig")
    assert plain["trend_id"].tolist() == ext["trend_id"].tolist()
    for name in ("target_values", "target_masks", "history_values"):
        np.testing.assert_array_equal(np.load(root / "plain" / "inference" / f"{name}.npy"),
                                      np.load(root / "ext" / "inference" / f"{name}.npy"))
    assert not np.load(root / "ext" / "inference" / "target_synthetic_masks.npy").any()


def test_extension_adds_training_anchors_and_respects_the_maintenance_rule(extended):
    root = extended["root"]
    train = pd.read_csv(root / "ext" / "model_train" / "metadata.csv", encoding="utf-8-sig")
    assert (train["months_after_maintenance"] >= 5).all()
    assert train["synthetic_future_months"].sum() > 0
    real = load_target_masks(root / "ext" / "model_train", "real")
    assert (real.sum(axis=1) < 8).any()  # 合成の月を数えて初めて8か月を満たす起点がある
    evalext = np.load(root / "evalext" / "inference" / "target_synthetic_masks.npy")
    assert evalext.any()


def test_target_normalization_uses_real_months_only(extended):
    split_dir = extended["root"] / "ext" / "model_train"
    values = np.load(split_dir / "target_values.npy")
    real = load_target_masks(split_dir, "real")
    norm = Normalization.load(extended["root"] / "ext" / "target_normalization.json")
    assert norm.count == int((real > 0).sum())
    assert norm.mean == pytest.approx(float(values[real > 0].mean()), rel=1e-5)


def test_verify_accepts_both_builds_and_flags_rule_violations(extended):
    root, manifest = extended["root"], extended["manifest"]
    rules = dict(min_months_after_maintenance=5, maintenance_rule_splits=("model_train", "model_validation"))
    assert verify_dataset_leakage(root / "ext", manifest, no_synthetic_splits=("inference",), **rules) == []
    assert verify_dataset_leakage(root / "evalext", manifest, **rules) == []
    assert verify_dataset_leakage(root / "evalext", manifest, no_synthetic_splits=("inference",))
    assert verify_dataset_leakage(root / "plain", manifest, **rules)  # 補修直後の起点がある


def test_prepare_with_18_month_history_and_target_cap(extended):
    # 履歴18か月・補修から6か月以上・目標月は補修から16か月まで(合成データは22か月分しかないので16で切る)
    root = extended["root"]
    options = dict(device="cpu", inference_max_per_segment=10,
                   trend_catalog_path=extended["extended_dir"] / "trend_catalog.csv",
                   database_path=extended["extended_dir"] / "vector_database.sqlite",
                   min_months_after_maintenance=6, history_months=18, max_months_after_maintenance=16)
    summary = prepare_datasets(extended["artifact_dir"], root / "h18", extend_splits=SPLITS,
                               maintenance_rule_splits=SPLITS, **options)
    assert summary["config"]["history_months"] == 18
    assert summary["config"]["max_months_after_maintenance"] == 16
    for split in SPLITS:
        frame = pd.read_csv(root / "h18" / split / "metadata.csv", encoding="utf-8-sig")
        history_values = np.load(root / "h18" / split / "history_values.npy")
        history_masks = np.load(root / "h18" / split / "history_masks.npy")
        assert history_values.shape == (len(frame), 18) and history_masks.shape == (len(frame), 18)
        months = frame["months_after_maintenance"].to_numpy()
        assert months.min() >= 6 and months.max() <= 16 - 8 + 0  # 目標月8か月以上のうち、16か月目までに収まる起点
        masks = np.load(root / "h18" / split / "target_masks.npy") > 0
        assert not (masks & (months[:, None] + np.arange(1, 13)[None, :] > 16)).any()
        # 窓の長さ=補修からの経過: 有効な履歴は先頭(補修直後)側から欠けない(窓は起点に右詰め)
        valid = history_masks.sum(axis=1)
        assert (valid <= months + 1).all()
    assert verify_dataset_leakage(root / "h18", extended["manifest"], min_months_after_maintenance=6,
                                  maintenance_rule_splits=SPLITS, max_months_after_maintenance=16) == []
    # 上限を厳しくして検査すると、違反として検出される
    assert verify_dataset_leakage(root / "h18", extended["manifest"], max_months_after_maintenance=12)


@pytest.fixture(scope="module")
def staggered(tmp_path_factory):
    """開始月をずらした合成データ(補修からの経過月数が同じでも計測日が違う候補がある)と、18か月の窓の検索器。"""
    from acceleration_forecasting_v2.retrieval.database import connect_read_only
    from acceleration_forecasting_v2.retrieval.search import GuideIndex
    from acceleration_forecasting_v2.retrieval.waveform_history import build_waveform_history_index
    root = tmp_path_factory.mktemp("staggered")
    artifact_dir, trend_dir, extended_dir = root / "artifacts", root / "trends", root / "extended"
    manifest = build_artifacts(artifact_dir, n_datasets=16, months=22, seed=0, staggered_start=True)
    _write_raw_trend_dir(trend_dir, 16, 22, staggered_start=True)
    build_extended_artifacts(artifact_dir, extended_dir, trend_dir=trend_dir, seed=42)
    options = dict(device="cpu", trend_catalog_path=extended_dir / "trend_catalog.csv",
                   database_path=extended_dir / "vector_database.sqlite", min_months_after_maintenance=6,
                   history_months=18, max_months_after_maintenance=16)
    prepare_datasets(artifact_dir, root / "src", extend_splits=SPLITS, maintenance_rule_splits=SPLITS, **options)
    guide_index = GuideIndex(connect_read_only(extended_dir / "vector_database.sqlite"))
    index = build_waveform_history_index(guide_index, artifact_dir, device="cpu", months=18,
                                         trend_catalog_path=extended_dir / "trend_catalog.csv")
    return dict(root=root, artifact_dir=artifact_dir, extended_dir=extended_dir, manifest=manifest,
                guide_index=guide_index, index=index, options=options)


def _queries(staggered, split="model_validation"):
    frame = pd.read_csv(staggered["root"] / "src" / split / "metadata.csv", encoding="utf-8-sig")
    return list(frame.itertuples(index=False))


def _search(index, row, config, allowed=("model_train",)):
    return index.search(str(row.trend_id), query_date=str(row.anchor_date), query_dataset_id=str(row.dataset_id),
                        query_current=float(row.current_acc_z_max), query_bin_start_m=float(row.bin_start_m),
                        allowed_splits=allowed, config=config)


def test_waveform_history_window_is_18_months_and_age_is_months_since_cycle_start(staggered):
    index = staggered["index"]
    assert index.window_values.shape[1] == 18 and index.window_masks.shape[1] == 18
    row = _queries(staggered)[0]
    assert index.window_age[index.window_lookup[str(row.trend_id)]] == row.months_after_maintenance
    # 補修より前の月は欠測なので、有効な月数は経過月数+1以下
    assert index.window(str(row.trend_id))[1].sum() <= row.months_after_maintenance + 1


def test_same_cycle_age_search_returns_only_candidates_of_the_same_age(staggered):
    from acceleration_forecasting_v2.retrieval.search import SearchConfig
    index = staggered["index"]
    config = SearchConfig(same_cycle_age=True, cycle_age_fallback_months=0)
    found = 0
    for row in _queries(staggered):
        guides = _search(index, row, config)
        for guide in guides:
            assert index.window_age[index.window_lookup[str(guide["trend_id"])]] == row.months_after_maintenance
            assert guide["dataset_id"] != str(row.dataset_id)
        found += len(guides)
    assert found > 0


def test_same_cycle_age_fallback_widens_by_one_month_only_when_short(staggered):
    from acceleration_forecasting_v2.retrieval.search import SearchConfig
    index = staggered["index"]
    # 現在値の許容差を狭めて候補を減らし、同じ月数の候補が3件に満たない起点を作る
    strict = SearchConfig(same_cycle_age=True, cycle_age_fallback_months=0, max_current_difference=0.015)
    relaxed = SearchConfig(same_cycle_age=True, cycle_age_fallback_months=1, max_current_difference=0.015)
    widened_any = False
    for row in _queries(staggered):
        exact = _search(index, row, strict)
        widened = _search(index, row, relaxed)
        assert [g["trend_id"] for g in widened[:len(exact)]] == [g["trend_id"] for g in exact]  # 同じ月の候補が先
        assert len(exact) <= len(widened) <= relaxed.top_k
        for guide in widened[len(exact):]:
            gap = abs(index.window_age[index.window_lookup[str(guide["trend_id"])]] - row.months_after_maintenance)
            assert gap == 1
            widened_any = True
        if len(exact) >= relaxed.top_k:
            assert widened == exact
    assert widened_any


def test_default_search_is_unchanged_when_the_new_options_are_off(staggered):
    from acceleration_forecasting_v2.retrieval.search import SearchConfig
    index = staggered["index"]
    default, explicit_off = SearchConfig(), SearchConfig(same_cycle_age=False, max_months_after_maintenance=None)
    assert default == explicit_off
    for row in _queries(staggered):
        guides = _search(index, row, default)
        assert len(guides) <= default.top_k
        # 従来の選び方(類似度の高い順・同日重複なし)をそのまま使う: 類似度は降順
        similarities = [g["similarity"] for g in guides]
        assert similarities == sorted(similarities, reverse=True)
        assert len({g["measurement_date"] for g in guides}) == len(guides)


def test_guide_futures_are_capped_at_months_after_maintenance(staggered):
    from acceleration_forecasting_v2.retrieval.search import SearchConfig
    index = staggered["index"]
    cap = 16
    config = SearchConfig(max_months_after_maintenance=cap)
    capped_any = False
    for row in _queries(staggered):
        for guide in _search(index, row, config):
            age = int(index.window_age[index.window_lookup[str(guide["trend_id"])]])
            for offset, (value, mask) in enumerate(zip(guide["future_values"], guide["future_mask"]), start=1):
                if age + offset > cap:
                    assert mask == 0 and value is None
                    capped_any = True
            assert guide["valid_months"] >= config.min_valid_months  # 上限で切ったあとの有効月数で選ばれている
    assert capped_any


def test_rebuild_guides_with_18_months_keeps_history_and_targets(staggered):
    from acceleration_forecasting_v2.datasets.history_search_dataset import build_history_search_dataset
    from acceleration_forecasting_v2.retrieval.search import SearchConfig
    root = staggered["root"]
    config = SearchConfig(same_cycle_age=True, max_months_after_maintenance=16)
    summary = build_history_search_dataset(
        root / "src", staggered["artifact_dir"], root / "out", search_config=config,
        search_method="waveform_history_cosine", device="cpu", history_months=18,
        database_path=staggered["extended_dir"] / "vector_database.sqlite",
        trend_catalog_path=staggered["extended_dir"] / "trend_catalog.csv")
    assert summary["search_config"]["same_cycle_age"] is True
    for split in SPLITS:
        if not (root / "src" / split).is_dir():
            continue
        for name in ("history_values", "history_masks", "target_values", "target_masks"):
            np.testing.assert_array_equal(np.load(root / "src" / split / f"{name}.npy"),
                                          np.load(root / "out" / split / f"{name}.npy"))
        masks = np.load(root / "out" / split / "guide_masks.npy")
        assert masks.shape[2] == 12
        frame = pd.read_csv(root / "out" / split / "metadata.csv", encoding="utf-8-sig")
        assert (frame["guide_search_method"] == "waveform_history_cosine").all()
    assert np.load(root / "out" / "model_validation" / "guide_masks.npy").sum() > 0
    assert verify_dataset_leakage(root / "out", staggered["manifest"], min_months_after_maintenance=6,
                                  maintenance_rule_splits=SPLITS, max_months_after_maintenance=16) == []


def test_evaluate_guide_value_compares_search_settings_on_the_same_anchors(staggered, capsys):
    from acceleration_forecasting_v2.cli import main
    from acceleration_forecasting_v2.datasets.history_search_dataset import build_history_search_dataset
    from acceleration_forecasting_v2.evaluation.guide_value import evaluate_guide_value, run_records
    from acceleration_forecasting_v2.retrieval.search import SearchConfig
    root = staggered["root"]
    common = dict(search_method="waveform_history_cosine", device="cpu",
                  database_path=staggered["extended_dir"] / "vector_database.sqlite",
                  trend_catalog_path=staggered["extended_dir"] / "trend_catalog.csv")
    build_history_search_dataset(root / "src", staggered["artifact_dir"], root / "r0", history_months=6,
                                 search_config=SearchConfig(max_months_after_maintenance=16), **common)
    build_history_search_dataset(root / "src", staggered["artifact_dir"], root / "r_align", history_months=18,
                                 search_config=SearchConfig(same_cycle_age=True, max_months_after_maintenance=16),
                                 **common)
    result = evaluate_guide_value({"R0": root / "r0", "R_align": root / "r_align"}, root / "guide_value",
                                  baseline="R0", use_cycle_features=False, bootstrap=50)
    assert set(result["runs"]) == {"R0", "R_align"}
    for name, item in result["runs"].items():
        assert item["records"] > 0 and item["segments"] > 0
        assert np.isfinite(item["guide_copy_MAE"]["mean"]) and np.isfinite(item["added_value"]["mean"])
        low, high = item["added_value"]["ci"]
        assert low <= item["added_value"]["mean"] <= high or np.isclose(low, high)
        assert item["by_months_after_maintenance"]
    difference = result["runs"]["R_align"]["difference_from_baseline"]
    assert set(difference) == {"guide_copy_MAE", "ridge_guide_MAE", "added_value"}
    assert "difference_from_baseline" not in result["runs"]["R0"]
    assert (root / "guide_value" / "guide_value_summary.csv").is_file()
    # 評価は実測の月だけ: 合成の月だけの起点は対象外になる
    frame = run_records(root / "r0", use_cycle_features=False)
    assert (frame["valid_months"] >= 1).all()
    # 同じ起点の並びでなければ差は出さない
    assert main(["evaluate-guide-value", "--run", f"R0={root / 'r0'}", "--run", f"R_align={root / 'r_align'}",
                 "--baseline", "R0", "--output-dir", str(root / "guide_value_cli"), "--no-cycle-features",
                 "--bootstrap", "20"]) == 0
    capsys.readouterr()


def test_month_filter_real_drops_synthetic_months(extended):
    split_dir = extended["root"] / "evalext" / "inference"
    all_masks = load_target_masks(split_dir, "all")
    real = load_target_masks(split_dir, "real")
    synthetic = np.load(split_dir / "target_synthetic_masks.npy")
    np.testing.assert_array_equal(real, all_masks * (1 - synthetic))
    with pytest.raises(ValueError):
        load_target_masks(split_dir, "synthetic")


def test_predict_forgetting_factor_writes_one_line_per_lambda(extended):
    root = extended["root"]
    output = root / "ff"
    summary = predict_forgetting_factor(root / "ext", extended["artifact_dir"] / "trend_catalog.csv", output)
    with np.load(output / "samples.npz") as npz:
        samples, trend_ids = npz["samples"], npz["trend_ids"]
    metadata = pd.read_csv(root / "ext" / "inference" / "metadata.csv", encoding="utf-8-sig")
    assert samples.shape == (len(metadata), len(DEFAULT_LAMBDAS), 12)
    assert trend_ids.tolist() == metadata["trend_id"].astype(str).tolist()
    assert summary["record_count"] == len(metadata)
    catalog = pd.read_csv(extended["artifact_dir"] / "trend_catalog.csv", encoding="utf-8-sig")
    row = metadata.iloc[0]
    rows = catalog.loc[(catalog["dataset_id"] == row["dataset_id"])
                       & (pd.to_datetime(catalog["measurement_date"]) <= pd.Timestamp(row["anchor_date"]))]
    grid = pd.date_range(pd.Timestamp(row["anchor_date"]).replace(day=1) + pd.DateOffset(months=1), periods=12,
                         freq="MS")
    expected = forgetting_factor_line(pd.to_datetime(rows["measurement_date"]), rows["current_acc_z_max"], grid, 1.0)
    np.testing.assert_allclose(samples[0, 0], expected, rtol=1e-5)
    predictions = pd.read_csv(output / "predictions.csv", encoding="utf-8-sig")
    assert len(predictions) == len(metadata) * 12


def test_summary_counts_extended_cycles_and_synthetic_months(extended):
    summary = extended["summary"]
    assert sum(item["extended"] for item in summary["cycles"].values()) == 12
    assert summary["database"]["rows_with_synthetic"] > 0
    assert month_index("2024-01-01") - month_index("2022-01-15") == 24


def test_building_only_inference_matches_the_full_build_and_copies_normalization(extended):
    root, extended_dir = extended["root"], extended["extended_dir"]
    output = root / "evalext_only_inference"
    prepare_datasets(extended["artifact_dir"], output, device="cpu", inference_max_per_segment=10,
                     trend_catalog_path=extended_dir / "trend_catalog.csv",
                     database_path=extended_dir / "vector_database.sqlite", min_months_after_maintenance=5,
                     extend_splits=SPLITS, maintenance_rule_splits=SPLITS, splits=("inference",),
                     normalization_dir=root / "ext")
    assert not (output / "model_train").exists()
    for name in ("condition_normalization.json", "target_normalization.json"):
        assert (output / name).read_bytes() == (root / "ext" / name).read_bytes()
    for name in ("target_values", "target_masks", "target_synthetic_masks", "history_values", "guide_values"):
        np.testing.assert_array_equal(np.load(output / "inference" / f"{name}.npy"),
                                      np.load(root / "evalext" / "inference" / f"{name}.npy"))
    with pytest.raises(ValueError, match="normalization_dir"):
        prepare_datasets(extended["artifact_dir"], root / "x", device="cpu", splits=("inference",))
