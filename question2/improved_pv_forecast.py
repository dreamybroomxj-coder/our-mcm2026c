"""问题二光伏预测改进实验，并与原四模型进行统一滚动回测比较。"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from forecast_models import PV_SHEET, date_indices, read_power_sheet


SEED = 2026
OLD_MODELS = ("Baseline", "XGBoost", "LightGBM", "SARIMA")
NEW_MODELS = ("日总量+归一化曲线", "晴空包络短期外推", "自适应加权Baseline")


def daily_total_features(totals: np.ndarray, date: datetime, day_index: int) -> np.ndarray:
    """当天 0:00 已知的日总量滞后与日历特征。"""
    history_7 = totals[day_index - 7 : day_index]
    history_14 = totals[day_index - 14 : day_index]
    doy = date.timetuple().tm_yday
    dow = date.weekday()
    return np.asarray(
        [
            totals[day_index - 1],
            totals[day_index - 2],
            totals[day_index - 3],
            totals[day_index - 7],
            totals[day_index - 14],
            history_7.mean(),
            history_7.std(),
            history_14.mean(),
            history_14.std(),
            (totals[day_index - 1] - totals[day_index - 3]) / 2,
            math.sin(2 * math.pi * doy / 365),
            math.cos(2 * math.pi * doy / 365),
            math.sin(2 * math.pi * dow / 7),
            math.cos(2 * math.pi * dow / 7),
        ],
        dtype=float,
    )


def normalized_mean_profile(values: np.ndarray, day_index: int, days: int = 7) -> np.ndarray:
    history = values[day_index - days : day_index]
    totals = history.sum(axis=1)
    valid = totals > 1e-8
    profiles = history[valid] / totals[valid, None]
    # 越近的日期权重越高，同时保留 7 天平滑能力。
    weights = np.geomspace(0.65, 1.0, len(profiles))
    profile = np.average(profiles, axis=0, weights=weights)
    profile = np.clip(profile, 0.0, None)
    return profile / profile.sum()


def energy_profile_predict(
    values: np.ndarray, dates: list[datetime], indices: np.ndarray, n_estimators: int
) -> np.ndarray:
    """XGBoost 预测日总量，近期归一化曲线预测日内形状。"""
    totals = values.sum(axis=1)
    predictions = []
    for day_index in indices:
        train_days = np.arange(14, day_index)
        x_train = np.vstack([daily_total_features(totals, dates[j], j) for j in train_days])
        y_train = totals[train_days]
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=5.0,
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(x_train, y_train)
        predicted_total = max(
            0.0,
            float(model.predict(daily_total_features(totals, dates[day_index], day_index)[None, :])[0]),
        )
        predictions.append(predicted_total * normalized_mean_profile(values, day_index))
    return np.vstack(predictions)


def smooth_curve(curve: np.ndarray) -> np.ndarray:
    kernel = np.asarray([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    return np.convolve(curve, kernel, mode="same")


def extrapolated_clear_sky_envelope(values: np.ndarray, day_index: int) -> np.ndarray:
    """由两个历史窗口的高分位包络估计季节趋势，并短期外推至目标日。"""
    history_start = max(0, day_index - 42)
    history = values[history_start:day_index]
    split = max(7, len(history) // 2)
    old = history[:-split]
    recent = history[-split:]
    if len(old) < 5:
        old = history[: max(5, len(history) // 2)]
    old_envelope = np.quantile(old, 0.90, axis=0)
    recent_envelope = np.quantile(recent, 0.90, axis=0)

    center_gap = max(1.0, (len(old) + len(recent)) / 2)
    horizon_from_recent_center = (len(recent) + 1) / 2
    slope = (recent_envelope - old_envelope) / center_gap
    # 限制短期趋势，防止云量异常导致高分位包络爆炸。
    slope_limit = np.maximum(2.0, 0.02 * np.maximum(recent_envelope, 1.0))
    slope = np.clip(slope, -slope_limit, slope_limit)
    envelope = recent_envelope + slope * horizon_from_recent_center
    envelope = smooth_curve(np.clip(envelope, 0.0, None))

    historical_cap = np.max(history, axis=0)
    envelope = np.minimum(envelope, np.maximum(1.15 * historical_cap, recent_envelope))
    # 历史持续无发电的夜间强制为零。
    envelope[historical_cap < 1.0] = 0.0
    return np.clip(envelope, 0.0, None)


def clear_sky_predict(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    envelopes: dict[int, np.ndarray] = {}

    def envelope(day_index: int) -> np.ndarray:
        if day_index not in envelopes:
            envelopes[day_index] = extrapolated_clear_sky_envelope(values, day_index)
        return envelopes[day_index]

    predictions = []
    for day_index in indices:
        target_envelope = envelope(day_index)
        ratios = []
        for historical_day in range(day_index - 7, day_index):
            historical_envelope = envelope(historical_day)
            ratio = np.divide(
                values[historical_day],
                historical_envelope,
                out=np.zeros(144, dtype=float),
                where=historical_envelope > 1.0,
            )
            ratios.append(np.clip(ratio, 0.0, 1.25))
        # 最近日权重更大；晴空指数承载短期天气持续性。
        weights = np.geomspace(0.55, 1.0, 7)
        predicted_ratio = np.average(np.vstack(ratios), axis=0, weights=weights)
        prediction = target_envelope * predicted_ratio
        predictions.append(np.minimum(np.clip(prediction, 0.0, None), 1.10 * target_envelope))
    return np.vstack(predictions)


def baseline_candidates(values: np.ndarray, day_index: int) -> np.ndarray:
    return np.column_stack(
        [
            values[day_index - 1],
            values[day_index - 2],
            values[day_index - 7],
            values[day_index - 7 : day_index].mean(axis=0),
        ]
    )


def adaptive_baseline_predict(
    values: np.ndarray, indices: np.ndarray, calibration_days: int = 28
) -> tuple[np.ndarray, np.ndarray]:
    """滚动 NNLS 学习四个历史基准的非负、和为 1 的自适应权重。"""
    predictions = []
    all_weights = []
    for day_index in indices:
        first_train_day = max(7, day_index - calibration_days)
        train_days = range(first_train_day, day_index)
        x_train = np.vstack([baseline_candidates(values, j) for j in train_days])
        y_train = np.concatenate([values[j] for j in train_days])
        daylight = np.logical_or(y_train > 1.0, x_train.max(axis=1) > 1.0)
        weights, _ = nnls(x_train[daylight], y_train[daylight])
        if weights.sum() <= 1e-12:
            weights = np.full(4, 0.25)
        else:
            weights /= weights.sum()
        prediction = baseline_candidates(values, day_index) @ weights
        predictions.append(np.clip(prediction, 0.0, None))
        all_weights.append(weights)
    return np.vstack(predictions), np.vstack(all_weights)


def extended_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    # 所有模型共用真实功率阈值，避免模型相关的“日间”样本集合造成不公平比较。
    daylight = actual > 1.0
    # 每个样本代表 10 分钟，功率和乘 1/6 h 后得到日电量。
    actual_energy = actual.sum(axis=1) / 6.0
    predicted_energy = predicted.sum(axis=1) / 6.0
    actual_peak = actual.max(axis=1)
    predicted_peak = predicted.max(axis=1)
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "WAPE_percent": float(np.abs(error).sum() / actual.sum() * 100),
        "Bias": float(error.mean()),
        "Daylight_MAE": float(np.abs(error[daylight]).mean()),
        "Daylight_WAPE_percent": float(np.abs(error[daylight]).sum() / actual[daylight].sum() * 100),
        "R2": float(r2_score(actual.ravel(), predicted.ravel())),
        "Daily_energy_MAPE_percent": float(
            np.mean(np.abs(predicted_energy - actual_energy) / actual_energy) * 100
        ),
        "Daily_energy_MAE": float(np.abs(predicted_energy - actual_energy).mean()),
        "Daily_peak_MAE": float(np.abs(predicted_peak - actual_peak).mean()),
        "Overforecast_rate_percent": float((predicted[daylight] > actual[daylight]).mean() * 100),
    }


def comparison_tables(predictions, actual, dates, indices):
    overall_rows = []
    monthly_rows = []
    for model_name, predicted in predictions.items():
        overall_rows.append({"模型": model_name, **extended_metrics(actual, predicted)})
        for month in sorted({dates[i].month for i in indices}):
            mask = np.asarray([dates[i].month == month for i in indices])
            monthly_rows.append({"模型": model_name, "月份": month, **extended_metrics(actual[mask], predicted[mask])})
    overall = pd.DataFrame(overall_rows).sort_values("WAPE_percent").reset_index(drop=True)
    monthly = pd.DataFrame(monthly_rows)
    return overall, monthly


def save_excel(path: Path, predictions, actual, dates, times, indices, weights) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        long_parts = []
        for model_name, predicted in predictions.items():
            frame = pd.DataFrame(predicted, columns=[str(item) for item in times])
            frame.insert(0, "日期\\时间", [dates[i] for i in indices])
            frame.to_excel(writer, sheet_name=model_name[:31], index=False)
            long_parts.append(
                pd.DataFrame(
                    {
                        "模型": model_name,
                        "日期": np.repeat([dates[i] for i in indices], 144),
                        "时段序号": np.tile(np.arange(1, 145), len(indices)),
                        "时刻": np.tile(times, len(indices)),
                        "预测功率": predicted.ravel(),
                        "实际功率": np.tile(np.nan, predicted.size),
                    }
                )
            )
            long_parts[-1]["实际功率"] = actual.ravel()
        pd.concat(long_parts, ignore_index=True).to_excel(writer, sheet_name="全部模型长表", index=False)
        pd.DataFrame(
            weights,
            columns=["前1天", "前2天", "前7天", "过去7天均值"],
            index=[dates[i] for i in indices],
        ).rename_axis("日期").to_excel(writer, sheet_name="自适应权重")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=script_dir.parent / "附件2.xlsx")
    parser.add_argument("--old-results", type=Path, default=script_dir / "results")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "improved_pv_results")
    parser.add_argument("--start-date", default="2025-02-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    dates, times, values = read_power_sheet(args.input, PV_SHEET)
    indices = date_indices(dates, datetime.fromisoformat(args.start_date), datetime.fromisoformat(args.end_date))
    actual = values[indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}
    for model_name in OLD_MODELS:
        path = args.old_results / f"{model_name}_光伏_predictions.npy"
        full_prediction = np.load(path)
        positions = indices - 31  # 旧结果固定从 2025-02-01 开始。
        if positions.min() < 0 or positions.max() >= len(full_prediction):
            raise ValueError(f"旧模型 {model_name} 不覆盖本次日期范围")
        predicted = full_prediction[positions]
        if predicted.shape != actual.shape:
            raise ValueError(f"旧模型 {model_name} 截取后维度异常：{predicted.shape}")
        predictions[model_name] = predicted

    started = time.perf_counter()
    predictions[NEW_MODELS[0]] = energy_profile_predict(values, dates, indices, args.n_estimators)
    print(f"{NEW_MODELS[0]} 完成，耗时 {time.perf_counter() - started:.1f} 秒", flush=True)

    started = time.perf_counter()
    predictions[NEW_MODELS[1]] = clear_sky_predict(values, indices)
    print(f"{NEW_MODELS[1]} 完成，耗时 {time.perf_counter() - started:.1f} 秒", flush=True)

    started = time.perf_counter()
    predictions[NEW_MODELS[2]], adaptive_weights = adaptive_baseline_predict(values, indices)
    print(f"{NEW_MODELS[2]} 完成，耗时 {time.perf_counter() - started:.1f} 秒", flush=True)
    if np.any(adaptive_weights < -1e-12) or not np.allclose(adaptive_weights.sum(axis=1), 1.0):
        raise AssertionError("自适应组合权重必须非负且逐日和为 1")

    for model_name in NEW_MODELS:
        predicted = predictions[model_name]
        if predicted.shape != actual.shape or not np.isfinite(predicted).all() or np.any(predicted < 0):
            raise AssertionError(f"{model_name} 输出存在维度、有限性或非负性错误")
        np.save(args.output_dir / f"{model_name}_predictions.npy", predicted)

    overall, monthly = comparison_tables(predictions, actual, dates, indices)
    overall.to_csv(args.output_dir / "光伏模型综合指标.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(args.output_dir / "光伏模型逐月指标.csv", index=False, encoding="utf-8-sig")
    save_excel(
        args.output_dir / "光伏改进模型预测与对比.xlsx",
        predictions,
        actual,
        dates,
        times,
        indices,
        adaptive_weights,
    )
    config = {
        "prediction_period": [args.start_date, args.end_date],
        "information_boundary": "每个预测日仅使用该日0:00以前的历史数据",
        "new_models": list(NEW_MODELS),
        "energy_profile_n_estimators": args.n_estimators,
        "adaptive_calibration_days": 28,
        "clear_sky_history_days": 42,
        "clear_sky_quantile": 0.90,
        "seed": SEED,
    }
    (args.output_dir / "改进实验配置.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
