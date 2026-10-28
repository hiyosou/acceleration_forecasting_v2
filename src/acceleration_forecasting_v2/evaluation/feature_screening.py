"""入力に足す特徴の候補を、安い予測器(月ごとのridge・小さなMLP)でふるいにかける(2026-10-27)。

拡散モデルは1回試すのに半日かかる。その前に「忘却因子Dの直線」「月内の全計測」を入力の特徴に足すと
履歴6か月だけの予測より良くなるかを、`baselines.py`の履歴ridgeに特徴を足して比べる。

- D系: 同じサイクルの補修(サイクル開始)から起点までの全計測で引いた版Dの直線(`forgetting_factor.py`)。
- H系: 履歴6か月それぞれの「月内の全計測」の中央値・回数・最大・最小・標準偏差と、直近30日の中央値。
  過去5か月はカレンダー月の全計測、今月は月初から起点の日まで。起点より後の計測は使わない。

学習は`model_train`、候補の選択は`model_validation`、最終の成績は`inference`で一度だけ出す。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from acceleration_forecasting_v2.common.constants import PHYSICAL_MAX, PHYSICAL_MIN
from acceleration_forecasting_v2.evaluation.baselines import (
    _anchor_months, _load_history, apply_weighted_ridge, fit_weighted_ridge, history_features,
)
from acceleration_forecasting_v2.evaluation.forgetting_factor import (
    D4_LAMBDAS, DEFAULT_LAMBDAS, d4_features, forecast_grid, forgetting_factor_coefficients, forgetting_factor_lines,
)

HISTORY_MONTHS = 6
COEFFICIENT_LAMBDAS = D4_LAMBDAS
COEFFICIENT_LAMBDAS_6 = (1.00, 0.98, 0.96, 0.94, 0.92, 0.90)
TRIM_FRACTION = 0.2  # トリム平均で、上下それぞれ落とす割合
RIDGE_LAMBDAS = (1.0, 10.0, 100.0)
MIN_TARGET_MONTHS = 8
UP_THRESHOLD, FLAT_THRESHOLD = 0.5, 0.2
SPLITS = ("model_train", "model_validation", "inference")


def _catalog_groups(trend_catalog_path):
    catalog = pd.read_csv(trend_catalog_path, encoding="utf-8-sig",
                          usecols=["dataset_id", "measurement_date", "current_acc_z_max"])
    catalog["dataset_id"] = catalog["dataset_id"].astype(str)
    catalog["measurement_date"] = pd.to_datetime(catalog["measurement_date"]).dt.normalize()
    catalog["value"] = pd.to_numeric(catalog["current_acc_z_max"], errors="coerce")
    catalog = catalog.loc[catalog["value"].notna()].sort_values(["dataset_id", "measurement_date"], kind="mergesort")
    return {dataset_id: (pd.DatetimeIndex(group["measurement_date"]), group["value"].to_numpy(float))
            for dataset_id, group in catalog.groupby("dataset_id", sort=False)}


def _month_summary(values, fallback):
    """月内の全計測の(中央値, 回数, 最大, 最小, 標準偏差, 平均, トリム平均)。計測のない月は値をfallbackで補い回数0。"""
    if len(values) == 0:
        return fallback, 0.0, fallback, fallback, 0.0, fallback, fallback
    ordered = np.sort(values)
    drop = int(TRIM_FRACTION * len(ordered))
    return (float(np.median(values)), float(len(values)), float(values.max()), float(values.min()),
            float(values.std()), float(values.mean()), float(ordered[drop:len(ordered) - drop].mean()))


def _residual_std(dates, values):
    """λ=1.00の直線(全計測)からの残差の標準偏差。計測が2点未満なら0。"""
    if len(values) < 2:
        return 0.0
    intercept, slope = forgetting_factor_coefficients(dates, values, 1.0)
    fitted = intercept + slope * (dates - dates.max()).days.to_numpy(float)
    return float(np.std(np.asarray(values, dtype=float) - fitted))


def record_features(dates, values, anchor_date, current, *, months=HISTORY_MONTHS, forecast_months=12):
    """1つの予測起点の特徴(起点の日までの計測だけを使う)。

    Args:
        dates, values: 同じdataset_id(=補修サイクル)の全計測(日付昇順)。
        current: 起点の値(`current_acc_z_max`)。計測のない月の補完に使う。
    Returns:
        dict: D_samples (11, forecast_months)、D_coef (len(COEFFICIENT_LAMBDAS), 2: 起点の値・30日あたりの傾き)、
        H (months, 7: 中央値・回数・最大・最小・標準偏差・平均・トリム平均)、D_coef6 (6本のλの水準・傾き)、resid_std (λ=1.00の直線の残差の標準偏差)、R30 (2: 直近30日の中央値・回数)。
    """
    anchor = pd.Timestamp(anchor_date).normalize()
    stop = dates.searchsorted(anchor, side="right")
    past_dates, past_values = dates[:stop], values[:stop]
    if len(past_values) == 0:
        past_dates, past_values = pd.DatetimeIndex([anchor]), np.array([current])
    anchor_month = anchor.replace(day=1)
    history = []
    for offset in range(months - 1, -1, -1):
        start = anchor_month - pd.DateOffset(months=offset)
        end = min(start + pd.DateOffset(months=1), anchor + pd.Timedelta(days=1))
        picked = past_values[(past_dates >= start) & (past_dates < end)]
        history.append(_month_summary(picked, current))
    recent = past_values[past_dates > anchor - pd.Timedelta(days=30)]
    return {
        "D_samples": forgetting_factor_lines(past_dates, past_values, forecast_grid(anchor, forecast_months),
                                             DEFAULT_LAMBDAS),
        "D_coef": d4_features(past_dates, past_values),
        "D_coef6": np.array([[intercept, slope * 30.0] for intercept, slope in
                             (forgetting_factor_coefficients(past_dates, past_values, lam)
                              for lam in COEFFICIENT_LAMBDAS_6)]),
        "resid_std": np.array([_residual_std(past_dates, past_values)]),
        "H": np.array(history),
        "R30": np.array([float(np.median(recent)) if len(recent) else current, float(len(recent))]),
    }


def compute_split_features(dataset_dir, split, trend_catalog_path, *, groups=None):
    """データセットの1split分の特徴を、metadataの行の並びで作る。"""
    metadata = pd.read_csv(Path(dataset_dir) / split / "metadata.csv", encoding="utf-8-sig")
    groups = groups if groups is not None else _catalog_groups(trend_catalog_path)
    records = [record_features(*groups[str(row.dataset_id)], row.anchor_date, float(row.current_acc_z_max))
               for row in metadata.itertuples(index=False)]
    return {key: np.stack([record[key] for record in records]) for key in records[0]}


def load_or_compute_features(dataset_dir, split, trend_catalog_path, cache_dir, *, groups=None):
    cache = Path(cache_dir) / f"features_{split}.npz"
    if cache.is_file():
        with np.load(cache) as archive:
            return {key: archive[key] for key in archive.files}
    features = compute_split_features(dataset_dir, split, trend_catalog_path, groups=groups)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **features)
    return features


def _d_median(features):
    return np.median(features["D_samples"], axis=1)


def _d2(features):
    samples, median = features["D_samples"], _d_median(features)
    return np.column_stack([median, samples[:, 0, [0, -1]], samples[:, -1, [0, -1]],
                            (median[:, -1] - median[:, 0])[:, None]])


def _h(features, columns):
    return np.column_stack([features["H"][:, :, column] for column in columns])


# 候補名 → (グループ, 特徴を作る関数)。H列の並び: 0中央値 1回数 2最大 3最小 4標準偏差
CANDIDATES = {
    "D1": ("D", _d_median),
    "D2": ("D", _d2),
    "D3": ("D", lambda features: features["D_samples"].reshape(len(features["D_samples"]), -1)),
    "D4": ("D", lambda features: features["D_coef"].reshape(len(features["D_coef"]), -1)),
    "H1": ("H", lambda features: _h(features, (0, 1))),
    "H2": ("H", lambda features: _h(features, (0, 1, 2, 3))),
    "H3": ("H", lambda features: _h(features, (0, 1, 2, 3, 4))),
    "H4": ("H", lambda features: features["R30"]),
    "E1": ("E", lambda features: features["resid_std"]),
    "E2": ("E", lambda features: (features["D_coef"][:, -1, 0] - features["D_coef"][:, 0, 0])[:, None]),
    "E3": ("E", lambda features: features["D_coef6"].reshape(len(features["D_coef6"]), -1)),
}
# 履歴6か月の月の値を、月内の全計測の集計に置き換える(列は H の並びの添字)
REPLACEMENTS = {"Rmedian": 0, "Rmean": 5, "Rtrim": 6}


def candidate_matrix(name, features):
    """名前(`B0`、`D2`、`C:D2+H3`のような組み合わせ)から追加特徴の行列を作る。B0は列なし。"""
    parts = [part for part in name.removeprefix("C:").split("+") if part in CANDIDATES]
    if not parts:
        return np.zeros((len(next(iter(features.values()))), 0))
    return np.column_stack([CANDIDATES[part][1](features) for part in parts])


def design_matrix(name, data, features):
    """ridge・MLPに渡す特徴行列。履歴の特徴(`Rmedian`などを含む名前では、履歴6か月の月の値を月内の全計測の
    集計に置き換えたもの)の右に、追加特徴(`candidate_matrix`)を並べる。"""
    replacement = [REPLACEMENTS[part] for part in name.removeprefix("C:").split("+") if part in REPLACEMENTS]
    base = data.base
    if replacement:
        counts, values = features["H"][:, :, 1], features["H"][:, :, replacement[0]]
        base = history_features(np.where(counts > 0, values, np.nan), _anchor_months(data.metadata))
    return np.column_stack([base, candidate_matrix(name, features)])


class SplitData:
    """1splitの履歴・正解・metadata(ridgeやMLPに渡す形)。"""

    def __init__(self, dataset_dir, split):
        self.dir = Path(dataset_dir) / split
        self.metadata = pd.read_csv(self.dir / "metadata.csv", encoding="utf-8-sig")
        self.history = _load_history(self.dir)
        self.target = np.load(self.dir / "target_values.npy").astype(float)
        self.mask = (np.load(self.dir / "target_masks.npy") > 0) & np.isfinite(self.target)
        self.weights = np.load(self.dir / "segment_weights.npy").astype(float)
        self.current = self.metadata["current_acc_z_max"].to_numpy(float)
        self.segments = self.metadata["dataset_id"].astype(str).to_numpy()
        self.base = history_features(self.history, _anchor_months(self.metadata))
        self.eligible = np.flatnonzero(self.mask.sum(1) >= MIN_TARGET_MONTHS)


def _auc(score, label):
    if label.sum() == 0 or (~label).sum() == 0:
        return float("nan")
    rank = pd.Series(score).rank().to_numpy()
    positives, negatives = label.sum(), (~label).sum()
    return float((rank[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _segment_mean(values, segments):
    return float(pd.Series(values).groupby(segments).mean().mean())


def _bootstrap_ci(diff, segments, iterations=2000, seed=42):
    per_segment = pd.Series(diff).groupby(segments).mean().to_numpy()
    rng = np.random.default_rng(seed)
    means = [per_segment[rng.integers(0, len(per_segment), len(per_segment))].mean() for _ in range(iterations)]
    return np.percentile(means, [2.5, 97.5])


def residual_ensemble(prediction, residual_pool, size=100):
    """予測に、月ごとの検証残差の分位点(size個)を足した確率予測。shape (N, size, months)。"""
    levels = (np.arange(size) + 0.5) / size
    offsets = np.stack([np.quantile(residual_pool[month], levels) for month in range(prediction.shape[1])], axis=1)
    return np.clip(prediction[:, None, :] + offsets[None], PHYSICAL_MIN, PHYSICAL_MAX)


def fair_crps(ensemble, target):
    """fair版CRPS(アンサンブル m 本、割る m(m−1))。ensemble (N, m, months)、target (N, months) → (N, months)。"""
    m = ensemble.shape[1]
    absolute = np.abs(ensemble - np.nan_to_num(target)[:, None, :]).mean(1)
    ordered = np.sort(ensemble, axis=1).transpose(0, 2, 1)
    pairs = 2 * (ordered * (2 * np.arange(1, m + 1) - m - 1)).sum(-1)
    return absolute - pairs / (2 * m * (m - 1))


def residual_pool(prediction, data: SplitData):
    """検証データでの月ごとの残差(正解 − 予測)。選択対象の起点だけで作る。"""
    idx = data.eligible
    return [(data.target[idx, month] - prediction[idx, month])[data.mask[idx, month]]
            for month in range(prediction.shape[1])]


def score(prediction, data: SplitData, *, reference=None, pool=None):
    """選択対象の起点(正解8か月以上)での指標。セグメント単位の2段階平均。"""
    idx, mask, target = data.eligible, data.mask, data.target
    last3 = lambda values, i: values[i][np.flatnonzero(mask[i])[-3:]].mean()  # noqa: E731
    delta = np.array([last3(target, i) - data.current[i] for i in idx])
    up, flat = delta >= UP_THRESHOLD, np.abs(delta) < FLAT_THRESHOLD
    predicted_rise = np.array([last3(prediction, i) - data.current[i] for i in idx])
    error = np.array([np.abs(prediction[i] - target[i])[mask[i]].mean() for i in idx])
    segments = data.segments[idx]
    high = np.concatenate([(prediction[i] - target[i])[mask[i] & (target[i] >= 3)] for i in idx])
    row = {
        "MAE": _segment_mean(error, segments),
        "バイアス": _segment_mean([(prediction[i] - target[i])[mask[i]].mean() for i in idx], segments),
        "上昇件数": int(up.sum()), "上昇件 MAE": float(error[up].mean()) if up.any() else float("nan"),
        "上昇件 AUC": _auc(predicted_rise, up),
        "上昇件 予測の上昇": float(predicted_rise[up].mean()) if up.any() else float("nan"),
        "横ばい件 MAE": float(error[flat].mean()) if flat.any() else float("nan"),
        "3以上の月の平均誤差": float(high.mean()) if len(high) else float("nan"),
    }
    if reference is not None:
        diff = error - reference["error"]
        low, high_ci = _bootstrap_ci(diff, segments)
        row.update({"B0との差": _segment_mean(diff, segments), "差CI低": float(low), "差CI高": float(high_ci)})
    if pool is not None:
        ensemble = residual_ensemble(prediction[idx], pool)
        crps = fair_crps(ensemble, target[idx])
        row["CRPS"] = _segment_mean([crps[k][mask[i]].mean() for k, i in enumerate(idx)], segments)
    row["error"] = error
    return row


def fit_ridge_candidate(name, train: SplitData, validation: SplitData, features):
    """候補の追加特徴を付けてridgeを学習し、検証MAEの最良のridgeλを選ぶ。

    Returns:
        dict(ridge_lambda, model, matrices: split → 特徴行列, validation_prediction)。
    """
    matrices = {split: design_matrix(name, data, features[split])
                for split, data in (("model_train", train), ("model_validation", validation))}
    best = None
    for ridge_lambda in RIDGE_LAMBDAS:
        model = fit_weighted_ridge(matrices["model_train"], train.dir, ridge_lambda=ridge_lambda)
        prediction = apply_weighted_ridge(model, matrices["model_validation"])
        idx = validation.eligible
        mae = _segment_mean([np.abs(prediction[i] - validation.target[i])[validation.mask[i]].mean() for i in idx],
                            validation.segments[idx])
        if best is None or mae < best["mae"]:
            best = {"ridge_lambda": ridge_lambda, "model": model, "validation_prediction": prediction, "mae": mae}
    return best


def fit_mlp(train_x, train_current, train: SplitData, validation_x, validation_current, validation: SplitData, *,
            seeds=(0, 1, 2), hidden=64, max_epochs=60, patience=8):
    """追加特徴を含む特徴で、起点の値に足す形の2層MLP(重み付きL1、検証損失で早期終了)を学習する。

    Returns:
        predict(特徴行列, 起点の値) → 3つのseedの平均予測 (N, months)。
    """
    import torch

    mean, std = train_x.mean(0), train_x.std(0) + 1e-9
    months = train.target.shape[1]

    def tensors(x, current, data):
        return (torch.tensor((x - mean) / std, dtype=torch.float32), torch.tensor(current, dtype=torch.float32),
                torch.tensor(np.nan_to_num(data.target), dtype=torch.float32),
                torch.tensor(data.mask, dtype=torch.float32),
                torch.tensor(data.weights / data.weights.mean(), dtype=torch.float32))

    def loss_of(network, batch):
        x, current, target, mask, weight = batch
        prediction = current[:, None] + network(x)
        return ((prediction - target).abs() * mask).sum(1).div(mask.sum(1).clamp(min=1)).mul(weight).mean()

    train_batch, validation_batch = tensors(train_x, train_current, train), tensors(validation_x, validation_current,
                                                                                    validation)
    networks = []
    for seed in seeds:
        torch.manual_seed(seed)
        network = torch.nn.Sequential(torch.nn.Linear(train_x.shape[1], hidden), torch.nn.ReLU(),
                                      torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, months))
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
        best_loss, best_state, waited = float("inf"), None, 0
        for _ in range(max_epochs):
            network.train()
            order = torch.randperm(len(train_batch[0]))
            for start in range(0, len(order), 512):
                chosen = order[start:start + 512]
                optimizer.zero_grad()
                loss_of(network, tuple(part[chosen] for part in train_batch)).backward()
                optimizer.step()
            network.eval()
            with torch.no_grad():
                validation_loss = float(loss_of(network, validation_batch))
            if validation_loss < best_loss:
                best_loss, waited = validation_loss, 0
                best_state = {key: value.clone() for key, value in network.state_dict().items()}
            else:
                waited += 1
                if waited >= patience:
                    break
        network.load_state_dict(best_state)
        network.eval()
        networks.append(network)

    def predict(x, current):
        tensor = torch.tensor((x - mean) / std, dtype=torch.float32)
        base = torch.tensor(current, dtype=torch.float32)[:, None]
        with torch.no_grad():
            return np.clip(np.mean([(base + network(tensor)).numpy() for network in networks], axis=0),
                           PHYSICAL_MIN, PHYSICAL_MAX)

    return predict


# 段階ごとの候補。stage 1: 特徴の種類(D系・H系)のふるい、stage 2: 版Dの要約(D4)への追加特徴と、月の値の置き換え。
STAGE2_D_NAMES = ("D4", "D4+E1", "D4+E2", "D4+E1+E2", "E3")
STAGE2_R_NAMES = tuple(REPLACEMENTS)


def _stage_plan(stage):
    if stage == 1:
        return {"D": [n for n, (k, _) in CANDIDATES.items() if k == "D"],
                "H": [n for n, (k, _) in CANDIDATES.items() if k == "H"]}
    return {"D": list(STAGE2_D_NAMES), "H": list(STAGE2_R_NAMES)}


def screen_features(dataset_dir, trend_catalog_path, output_dir, *, mlp=True, stage=1):
    """候補を学習・選択・評価して、表とsummary.jsonを`output_dir`に書く。

    stage 1: D1〜D4とH1〜H4を比べ、各グループの最良の組み合わせCも評価する。
    stage 2: D4への追加特徴(E1・E2)と6本のλ(E3)、月の値の置き換え(Rmedian・Rmean・Rtrim)を比べ、
        最良の置き換え+最良のDの組み合わせCも評価する。置き換えがB0より悪ければ組み合わせは作らない。
    """
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = _catalog_groups(trend_catalog_path)
    features = {split: load_or_compute_features(dataset_dir, split, trend_catalog_path, output_dir, groups=groups)
                for split in SPLITS}
    data = {split: SplitData(dataset_dir, split) for split in SPLITS}
    train, validation, inference = data["model_train"], data["model_validation"], data["inference"]
    plan = _stage_plan(stage)

    fitted, validation_rows, b0 = {}, [], None

    def validate(name, reference):
        fitted[name] = fit_ridge_candidate(name, train, validation, features)
        result = score(fitted[name]["validation_prediction"], validation, reference=reference)
        validation_rows.append({"候補": name, "ridgeλ": fitted[name]["ridge_lambda"],
                                "追加特徴数": design_matrix(name, validation, features["model_validation"]).shape[1]
                                - validation.base.shape[1],
                                **{key: value for key, value in result.items() if key != "error"}})
        return result

    b0 = validate("B0", None)
    for name in [*plan["D"], *plan["H"]]:
        validate(name, b0)
    best = {group: min(plan[group], key=lambda name: fitted[name]["mae"]) for group in plan}
    if stage == 2 and fitted[best["H"]]["mae"] >= fitted["B0"]["mae"]:
        combined = best["D"]  # 置き換えが効かなかった: 組み合わせは作らない
    else:
        combined = f"C:{best['H'] if stage == 2 else best['D']}+{best['D'] if stage == 2 else best['H']}"
    if combined not in fitted:
        validate(combined, b0)
    pd.DataFrame(validation_rows).to_csv(output_dir / "validation_table.csv", index=False, encoding="utf-8-sig")

    final_names = list(dict.fromkeys(["B0", best["D"], best["H"], combined]))
    inference_rows, reference = [], None
    for name in final_names:
        prediction = apply_weighted_ridge(fitted[name]["model"], design_matrix(name, inference, features["inference"]))
        pool = residual_pool(fitted[name]["validation_prediction"], validation)
        result = score(prediction, inference, reference=reference, pool=pool)
        reference = reference or result
        inference_rows.append({"候補": f"ridge {name}", "ridgeλ": fitted[name]["ridge_lambda"],
                               **{key: value for key, value in result.items() if key != "error"}})
    if mlp:
        for name in dict.fromkeys(["B0", best["D"], combined]):
            matrices = [design_matrix(name, data[split], features[split]) for split in SPLITS]
            predict = fit_mlp(matrices[0], train.current, train, matrices[1], validation.current, validation)
            pool = residual_pool(predict(matrices[1], validation.current), validation)
            result = score(predict(matrices[2], inference.current), inference, reference=reference, pool=pool)
            inference_rows.append({"候補": f"MLP {name}", "ridgeλ": float("nan"),
                                   **{key: value for key, value in result.items() if key != "error"}})
    table = pd.DataFrame(inference_rows)
    table.to_csv(output_dir / "inference_table.csv", index=False, encoding="utf-8-sig")
    summary = {"stage": stage, "best_D": best["D"], "best_H": best["H"], "combined": combined,
               "ridge_lambdas": {name: fitted[name]["ridge_lambda"] for name in fitted},
               "eligible_inference": int(len(inference.eligible)), "eligible_validation": int(len(validation.eligible)),
               "mlp": bool(mlp)}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"validation": pd.DataFrame(validation_rows), "inference": table, "summary": summary}
