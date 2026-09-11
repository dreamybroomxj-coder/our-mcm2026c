"""评价次日 00:10 负载的两种补点方法及最后一个区间平均负载。"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from forecast_models import LOAD_SHEET, read_power_sheet


BASE_MODELS = ("Baseline", "XGBoost", "LightGBM", "SARIMA")
BOUNDARY_METHODS = ("XGBoost跨日边界", "历史跨日比例", "目标时刻上周同期")
XGB_FEATURE_NAMES = (
    "前1日00:10", "前2日00:10", "前7日00:10", "前14日00:10",
    "近7日00:10均值", "近7日00:10标准差", "近14日00:10均值", "近14日00:10标准差",
    "前日23:20", "前日23:30", "前日23:40", "前日23:50", "前日24:00",
    "前日全天均值", "前日全天标准差", "近7日全天均值", "近7日全天标准差",
    "目标日星期正弦", "目标日星期余弦", "目标日年周期正弦", "目标日年周期余弦",
)


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "样本数": int(len(actual)),
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(math.sqrt(np.square(error).mean())),
        "WAPE_percent": float(np.abs(error).sum() / np.abs(actual).sum() * 100),
        "Bias": float(error.mean()),
        "MAPE_percent": float(np.mean(np.abs(error) / np.maximum(np.abs(actual), 1e-8)) * 100),
        "R2": float(1 - np.square(error).sum() / np.square(actual - actual.mean()).sum()),
    }


def rolling_ratio(values: np.ndarray, planning_day: int, window: int) -> tuple[float, int]:
    """只用规划日前已完整观测的跨日比例。"""
    # r_j = L_{j+1,00:10} / L_{j,24:00}; 在 planning_day 的 00:00，最大可用 j 为 day-2。
    stop = planning_day - 1
    start = max(0, stop - window)
    denominators = values[start:stop, -1]
    numerators = values[start + 1 : stop + 1, 0]
    valid = denominators > 1e-8
    ratios = numerators[valid] / denominators[valid]
    if len(ratios) == 0:
        raise ValueError(f"规划日索引 {planning_day} 没有可用跨日比例")
    return float(np.median(ratios)), int(len(ratios))


def xgb_boundary_features(values: np.ndarray, dates: list[datetime], planning_day: int) -> np.ndarray:
    """构造规划日 00:00 已知的特征，目标为次日 00:10。"""
    first_slot_7 = values[planning_day - 7 : planning_day, 0]
    first_slot_14 = values[planning_day - 14 : planning_day, 0]
    prior_days_7 = values[planning_day - 7 : planning_day]
    target_date = dates[planning_day + 1]
    dow = target_date.weekday()
    doy = target_date.timetuple().tm_yday
    return np.asarray(
        [
            values[planning_day - 1, 0], values[planning_day - 2, 0],
            values[planning_day - 7, 0], values[planning_day - 14, 0],
            first_slot_7.mean(), first_slot_7.std(), first_slot_14.mean(), first_slot_14.std(),
            *values[planning_day - 1, -5:],
            values[planning_day - 1].mean(), values[planning_day - 1].std(),
            prior_days_7.mean(), prior_days_7.std(),
            math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7),
            math.sin(2 * math.pi * doy / 365), math.cos(2 * math.pi * doy / 365),
        ],
        dtype=float,
    )


def train_xgb_boundary(
    values: np.ndarray,
    dates: list[datetime],
    planning_indices: np.ndarray,
    n_estimators: int,
) -> tuple[dict[int, float], pd.DataFrame]:
    """逐日扩展窗口训练；最后一个训练标签必须在规划日前已经观测。"""
    predictions: dict[int, float] = {}
    importance_rows = []
    for planning_day in planning_indices:
        # 样本 j 的标签是 j+1 日 00:10。规划日 i 的最后可用样本为 j=i-2。
        train_days = np.arange(14, planning_day - 1)
        x_train = np.vstack([xgb_boundary_features(values, dates, j) for j in train_days])
        y_train = values[train_days + 1, 0]
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            max_depth=2,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=8.0,
            random_state=2026,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(x_train, y_train)
        prediction = float(model.predict(xgb_boundary_features(values, dates, planning_day)[None, :])[0])
        predictions[planning_day] = max(0.0, prediction)
        importance_rows.append(model.feature_importances_)
    importance = pd.DataFrame(
        {
            "特征": XGB_FEATURE_NAMES,
            "全年日模型平均重要性": np.vstack(importance_rows).mean(axis=0),
        }
    ).sort_values("全年日模型平均重要性", ascending=False)
    return predictions, importance.reset_index(drop=True)


def build_predictions(
    values: np.ndarray,
    dates: list[datetime],
    old_results: Path,
    ratio_window: int,
    n_estimators: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 2025-12-31 的次日 00:10 不在附件中，故最后可评价的规划日为 12 月 30 日。
    planning_indices = np.arange(31, len(dates) - 1)
    ratio_rows = []
    ratio_by_day = {}
    for day_index in planning_indices:
        ratio, sample_count = rolling_ratio(values, day_index, ratio_window)
        ratio_by_day[day_index] = ratio
        ratio_rows.append(
            {
                "规划日期": dates[day_index],
                "比例窗口": ratio_window,
                "有效历史比例数": sample_count,
                "跨日比例中位数": ratio,
            }
        )

    xgb_by_day, xgb_importance = train_xgb_boundary(values, dates, planning_indices, n_estimators)

    rows = []
    for base_model in BASE_MODELS:
        full_prediction = np.load(old_results / f"{base_model}_负载_predictions.npy")
        if full_prediction.shape != (334, 144):
            raise ValueError(f"{base_model} 负载预测矩阵维度异常：{full_prediction.shape}")
        for day_index in planning_indices:
            prediction_row = day_index - 31
            predicted_2400 = float(full_prediction[prediction_row, -1])
            actual_2400 = float(values[day_index, -1])
            actual_next_0010 = float(values[day_index + 1, 0])
            actual_interval = (actual_2400 + actual_next_0010) / 2

            boundary_predictions = {
                "XGBoost跨日边界": xgb_by_day[day_index],
                "历史跨日比例": predicted_2400 * ratio_by_day[day_index],
                # 目标为 d+1 日 00:10，其 7 天前为 d-6 日 00:10。
                "目标时刻上周同期": float(values[day_index - 6, 0]),
            }
            for boundary_method, predicted_next_0010 in boundary_predictions.items():
                rows.append(
                    {
                        "规划日期": dates[day_index],
                        "目标日期": dates[day_index + 1],
                        "24:00基础模型": base_model,
                        "边界预测方法": boundary_method,
                        "预测_当日24:00": predicted_2400,
                        "实际_当日24:00": actual_2400,
                        "预测_次日00:10": predicted_next_0010,
                        "实际_次日00:10": actual_next_0010,
                        "预测_最后区间平均负载": (predicted_2400 + predicted_next_0010) / 2,
                        "实际_最后区间平均负载": actual_interval,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(ratio_rows), xgb_importance


def evaluate(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    monthly_rows = []
    for (base_model, method), group in predictions.groupby(["24:00基础模型", "边界预测方法"], sort=False):
        endpoint_metrics = error_metrics(
            group["实际_次日00:10"].to_numpy(), group["预测_次日00:10"].to_numpy()
        )
        interval_metrics = error_metrics(
            group["实际_最后区间平均负载"].to_numpy(),
            group["预测_最后区间平均负载"].to_numpy(),
        )
        overall_rows.extend(
            [
                {"24:00基础模型": base_model, "边界预测方法": method, "评价对象": "次日00:10端点", **endpoint_metrics},
                {"24:00基础模型": base_model, "边界预测方法": method, "评价对象": "最后10分钟区间平均", **interval_metrics},
            ]
        )
        for month, monthly_group in group.groupby(group["规划日期"].dt.month):
            for label, actual_column, prediction_column in (
                ("次日00:10端点", "实际_次日00:10", "预测_次日00:10"),
                ("最后10分钟区间平均", "实际_最后区间平均负载", "预测_最后区间平均负载"),
            ):
                monthly_rows.append(
                    {
                        "24:00基础模型": base_model,
                        "边界预测方法": method,
                        "评价对象": label,
                        "月份": int(month),
                        **error_metrics(
                            monthly_group[actual_column].to_numpy(),
                            monthly_group[prediction_column].to_numpy(),
                        ),
                    }
                )
    overall = pd.DataFrame(overall_rows).sort_values(
        ["评价对象", "WAPE_percent", "24:00基础模型", "边界预测方法"]
    )
    return overall.reset_index(drop=True), pd.DataFrame(monthly_rows)


def save_excel(path: Path, predictions, ratios, importance, overall, monthly) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        predictions.to_excel(writer, sheet_name="逐日预测", index=False)
        overall.to_excel(writer, sheet_name="综合指标", index=False)
        monthly.to_excel(writer, sheet_name="逐月指标", index=False)
        ratios.to_excel(writer, sheet_name="滚动跨日比例", index=False)
        importance.to_excel(writer, sheet_name="XGBoost特征重要性", index=False)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=script_dir.parent / "附件2.xlsx")
    parser.add_argument("--old-results", type=Path, default=script_dir / "results")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "cross_day_load_results")
    parser.add_argument("--ratio-window", type=int, default=28)
    parser.add_argument("--xgb-n-estimators", type=int, default=100)
    args = parser.parse_args()
    if args.ratio_window <= 0 or args.xgb_n_estimators <= 0:
        raise ValueError("比例窗口和 XGBoost 树数量必须为正整数")

    dates, _, values = read_power_sheet(args.input, LOAD_SHEET)
    predictions, ratios, importance = build_predictions(
        values, dates, args.old_results, args.ratio_window, args.xgb_n_estimators
    )
    overall, monthly = evaluate(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "跨日负载逐日预测.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(args.output_dir / "跨日负载综合指标.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(args.output_dir / "跨日负载逐月指标.csv", index=False, encoding="utf-8-sig")
    ratios.to_csv(args.output_dir / "滚动跨日比例.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(args.output_dir / "XGBoost特征重要性.csv", index=False, encoding="utf-8-sig")
    save_excel(args.output_dir / "跨日负载预测与评价.xlsx", predictions, ratios, importance, overall, monthly)
    (args.output_dir / "实验配置.json").write_text(
        json.dumps(
            {
                "planning_period": ["2025-02-01", "2025-12-30"],
                "sample_days": 333,
                "ratio_window": args.ratio_window,
                "ratio_statistic": "median",
                "xgb_n_estimators": args.xgb_n_estimators,
                "information_boundary": "规划日00:00时仅使用前一日及更早的完整观测",
                "base_models": list(BASE_MODELS),
                "boundary_methods": list(BOUNDARY_METHODS),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
