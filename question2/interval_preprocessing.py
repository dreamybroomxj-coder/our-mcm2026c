"""将附件中的瞬时端点数据转换为 10 分钟区间平均数据并保存。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill


PERIODS = 144
STEP_MINUTES = 10


def minute_label(minutes: int, next_day: bool = False) -> str:
    if minutes == 1440 or next_day:
        return "次日00:00"
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def period_mapping() -> pd.DataFrame:
    rows = []
    for k in range(PERIODS):
        start = k * STEP_MINUTES
        end = (k + 1) * STEP_MINUTES
        midpoint = start + STEP_MINUTES // 2
        rows.append(
            {
                "时段序号": k + 1,
                "开始时刻": minute_label(start),
                "结束时刻": minute_label(end),
                "中点时刻": minute_label(midpoint),
                "时间区间": f"{minute_label(start)}—{minute_label(end)}",
                "时长/h": 1 / 6,
            }
        )
    return pd.DataFrame(rows)


def read_attachment1(path: Path) -> tuple[list[str], np.ndarray]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook[workbook.sheetnames[0]]
    rows = list(worksheet.iter_rows(values_only=True))
    workbook.close()
    if len(rows) != 145 or len(rows[0]) != 4:
        raise ValueError(f"附件1应为145行×4列，实际为{len(rows)}行×{len(rows[0])}列")
    headers = [str(item) for item in rows[0]]
    values = np.asarray([[row[j] for j in range(1, 4)] for row in rows[1:]], dtype=float)
    if values.shape != (144, 3) or not np.isfinite(values).all():
        raise ValueError("附件1数值区域异常")
    return headers, values


def attachment1_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    headers, endpoints_10_to_24 = read_attachment1(path)
    # 按题设修订约定：单日附件缺少 00:00，使用该表 24:00 数值补齐。
    endpoints = np.vstack([endpoints_10_to_24[-1], endpoints_10_to_24])
    endpoint_labels = ["00:00"] + [minute_label((k + 1) * 10) for k in range(144)]
    endpoint_table = pd.DataFrame(endpoints, columns=headers[1:])
    endpoint_table.insert(0, headers[0], endpoint_labels)

    averages = (endpoints[:-1] + endpoints[1:]) / 2
    mapping = period_mapping()
    interval_table = mapping.copy()
    for column_index, column_name in enumerate(headers[1:]):
        interval_table[f"区间平均{column_name}"] = averages[:, column_index]
    return endpoint_table, interval_table


def read_attachment2_sheet(path: Path, sheet_name: str):
    workbook = load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook[sheet_name]
    rows = list(worksheet.iter_rows(values_only=True))
    workbook.close()
    if len(rows) != 366 or len(rows[0]) != 145:
        raise ValueError(f"{sheet_name}应为366行×145列")
    dates = [pd.Timestamp(row[0]).to_pydatetime() for row in rows[1:]]
    values = np.asarray([row[1:] for row in rows[1:]], dtype=float)
    if values.shape != (365, 144) or not np.isfinite(values).all():
        raise ValueError(f"{sheet_name}数值区域异常")
    return dates, values


def interval_average_actual(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回每天真实 00:00 边界和 144 个区间平均值。"""
    starts = np.empty(values.shape[0], dtype=float)
    starts[0] = values[0, -1]  # 仅首日缺少前一日边界，按约定近似。
    starts[1:] = values[:-1, -1]
    endpoints = np.column_stack([starts, values])
    averages = (endpoints[:, :-1] + endpoints[:, 1:]) / 2
    return starts, averages


def wide_table(dates, averages: np.ndarray, interval_labels: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(averages, columns=interval_labels)
    frame.insert(0, "日期\\区间", dates)
    return frame


def build_output(attachment1: Path, attachment2: Path, output: Path) -> None:
    endpoint_table, attachment1_intervals = attachment1_tables(attachment1)
    mapping = period_mapping()
    interval_labels = mapping["时间区间"].tolist()

    load_dates, load_values = read_attachment2_sheet(attachment2, "小区负载")
    pv_dates, pv_values = read_attachment2_sheet(attachment2, "光伏发电实际功率")
    if load_dates != pv_dates:
        raise ValueError("附件2两个工作表的日期不一致")
    load_00, load_intervals = interval_average_actual(load_values)
    pv_00, pv_intervals = interval_average_actual(pv_values)

    long_table = pd.DataFrame(
        {
            "日期": np.repeat(load_dates, PERIODS),
            "时段序号": np.tile(np.arange(1, PERIODS + 1), len(load_dates)),
            "时间区间": np.tile(interval_labels, len(load_dates)),
            "区间平均负载功率": load_intervals.ravel(),
            "区间平均光伏功率": pv_intervals.ravel(),
        }
    )
    boundaries = pd.DataFrame(
        {
            "日期": load_dates,
            "00:00真实负载功率": load_00,
            "00:00真实光伏功率": pv_00,
            "边界来源": ["首日以本日24:00近似"] + ["前一日24:00真实值"] * (len(load_dates) - 1),
        }
    )

    # 论文展示表对应附件1的144个规划区间，列名简化以便直接复制到论文。
    paper_table = attachment1_intervals.rename(
        columns={
            attachment1_intervals.columns[6]: "平均电价",
            attachment1_intervals.columns[7]: "平均负载功率",
            attachment1_intervals.columns[8]: "平均光伏功率",
        }
    )[["时段序号", "时间区间", "平均电价", "平均负载功率", "平均光伏功率"]]

    output.parent.mkdir(parents=True, exist_ok=True)
    notes = pd.DataFrame(
        [
            ["数据含义", "附件原值视为每10分钟时刻的瞬时端点值"],
            ["规划时刻", "每天00:00制定计划，并获得该时刻真实电价、负载和光伏功率"],
            ["预测范围", "模型预测00:10至次日00:00共144个未来端点"],
            ["区间转换", "第k个区间平均值=(区间左端点+区间右端点)/2"],
            ["附件1的00:00", "按约定使用附件1中24:00的数值补齐"],
            ["附件2的00:00", "1月2日起使用前一日24:00真实值；1月1日以本日24:00近似"],
            ["区间数量", "145个端点形成144个10分钟区间"],
            ["电量换算", "区间平均功率乘1/6小时得到该区间电量"],
        ],
        columns=["项目", "处理规则"],
    )
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="处理说明", index=False)
        mapping.to_excel(writer, sheet_name="时段映射", index=False)
        endpoint_table.to_excel(writer, sheet_name="附件1_145个端点", index=False)
        attachment1_intervals.to_excel(writer, sheet_name="附件1_区间平均", index=False)
        paper_table.to_excel(writer, sheet_name="论文展示表", index=False)
        boundaries.to_excel(writer, sheet_name="每日00点真实边界", index=False)
        wide_table(load_dates, load_intervals, interval_labels).to_excel(
            writer, sheet_name="负载区间平均_宽表", index=False
        )
        wide_table(pv_dates, pv_intervals, interval_labels).to_excel(
            writer, sheet_name="光伏区间平均_宽表", index=False
        )
        long_table.to_excel(writer, sheet_name="全年区间平均_长表", index=False)

    format_workbook(output)
    validate_output(output)


def format_workbook(path: Path) -> None:
    workbook = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
        for column_cells in worksheet.columns:
            sample = list(column_cells)[:200]
            width = min(24, max(10, max(len(str(cell.value or "")) for cell in sample) + 2))
            worksheet.column_dimensions[column_cells[0].column_letter].width = width
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.0000"
    workbook.save(path)


def validate_output(path: Path) -> None:
    workbook = load_workbook(path, read_only=True, data_only=True)
    expected = {
        "处理说明": (9, 2),
        "时段映射": (145, 6),
        "附件1_145个端点": (146, 4),
        "附件1_区间平均": (145, 9),
        "论文展示表": (145, 5),
        "每日00点真实边界": (366, 4),
        "负载区间平均_宽表": (366, 145),
        "光伏区间平均_宽表": (366, 145),
        "全年区间平均_长表": (365 * 144 + 1, 5),
    }
    for sheet_name, shape in expected.items():
        worksheet = workbook[sheet_name]
        actual = (worksheet.max_row, worksheet.max_column)
        if actual != shape:
            raise AssertionError(f"{sheet_name}维度异常：期望{shape}，实际{actual}")
    workbook.close()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment1", type=Path, default=script_dir.parent / "附件1.xlsx")
    parser.add_argument("--attachment2", type=Path, default=script_dir.parent / "附件2.xlsx")
    parser.add_argument("--output", type=Path, default=script_dir / "区间平均预处理数据.xlsx")
    args = parser.parse_args()
    build_output(args.attachment1, args.attachment2, args.output)
    print(f"已生成：{args.output}")
    print("附件1：145个瞬时端点 -> 144个区间平均值")
    print("附件2：365天 × 144个区间；2月以后00:00边界均来自前一日24:00真实值")


if __name__ == "__main__":
    main()
