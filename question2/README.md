# 问题二：负载与光伏日 ahead 预测

## 已实现方法

1. **Baseline**：负载取前 7 天同一时段；光伏取此前 7 天同一时段平均。
2. **XGBoost**：使用 1、2、7、14 天滞后、同一时段滚动统计量、时段及星期周期特征；每天重新训练。
3. **LightGBM**：与 XGBoost 使用相同信息集和滚动口径；每天重新训练，便于公平比较。
4. **SARIMA**：对每日总功率和建立周周期 SARIMA，再利用预测日前历史归一化日曲线分配到 144 个 10 分钟时段。

所有方法预测某日时都只使用该日 0:00 以前的数据。预测范围为 2025-02-01 至 2025-12-31。

## Pipeline 用 baseline 文件

`baseline_predictions.xlsx` 包含：

- `小区负载预测`：与附件 2 相同结构的 334×144 宽表；
- `光伏发电预测`：与附件 2 相同结构的 334×144 宽表；
- `预测长表`：日期、时段序号、时刻、负载预测功率、光伏预测功率，共 48,096 行。

生成命令：

```powershell
& 'F:\miniconda3\envs\fond\python.exe' '.\baseline_forecast.py'
```

## 四模型完整回测

```powershell
& 'F:\miniconda3\envs\fond\python.exe' '.\forecast_models.py'
```

主要输出位于 `results`：

- `all_model_predictions.xlsx`：四模型预测值和实际值；
- `overall_metrics.csv`：全年综合指标；
- `monthly_metrics.csv`：逐月指标；
- `*_predictions.npy`：便于 Python pipeline 快速载入的预测矩阵；
- `run_config.json`：本次运行参数。

## 当前正式回测结果

| 模型 | 对象 | MAE | RMSE | WAPE |
|---|---|---:|---:|---:|
| Baseline | 负载 | 176.796 | 244.287 | 3.829% |
| XGBoost | 负载 | **127.968** | **172.502** | **2.772%** |
| LightGBM | 负载 | 128.209 | 172.817 | 2.777% |
| SARIMA | 负载 | 420.060 | 564.178 | 9.099% |
| Baseline | 光伏 | **153.400** | **303.858** | **6.448%** |
| XGBoost | 光伏 | 156.226 | 313.012 | 6.567% |
| LightGBM | 光伏 | 155.438 | 310.584 | 6.534% |
| SARIMA | 光伏 | 157.216 | 311.374 | 6.609% |

负载方面 XGBoost 与 LightGBM 几乎持平，XGBoost 略优；光伏方面在没有天气预报变量时，7 天同期均值 baseline 的全年结果最好。

## 光伏改进实验

新增的日总量分解、晴空包络短期外推和自适应加权 Baseline 位于：

- `improved_pv_forecast.py`；
- `improved_pv_results/光伏改进模型预测与对比.xlsx`；
- `improved_pv_results/光伏模型综合指标.csv`；
- `improved_pv_results/光伏模型逐月指标.csv`；
- `光伏改进实验报告.md`。

运行命令：

```powershell
& 'F:\miniconda3\envs\fond\python.exe' '.\improved_pv_forecast.py'
```

三种改进中，自适应加权 Baseline 最优，全年 WAPE 为 6.398%，略优于原 7 天均值的 6.448%。

## 瞬时端点到区间平均值

按照每天 00:00 可获得真实边界值的假设：模型仍预测 00:10 至次日 00:00 共 144 个端点，再由相邻端点平均得到 144 个 10 分钟区间值。

- `interval_preprocessing.py`：生成附件 1、附件 2 的统一区间预处理表；
- `区间平均预处理数据.xlsx`：含处理说明、论文展示表、全年宽表和长表；
- `interval_forecast_postprocess.py`：转换所有既有预测并重新评价；
- `interval_results/baseline_interval_predictions.xlsx`：后续优化 pipeline 使用的区间 baseline；
- `interval_results/全部模型区间预测与指标.xlsx`：全部区间预测和指标。
