# 问题三补充预报价值分析

在 C 题根目录使用 `fond` 环境运行：

```powershell
F:\miniconda3\envs\fond\python.exe 附件\question3\judge\analyze_forecast_value.py
```

程序读取第三问八种主方案的 `daily.csv`，完成测试期与全年 Shapley 分解、条件边际收益、二时刻组合效应，以及7日/14日配对移动区块 Bootstrap。输出位于本目录的 `results`、`figures`，解释文档为 `问题三_补充预报价值分析.md`。

快速检查：

```powershell
F:\miniconda3\envs\fond\python.exe 附件\question3\judge\analyze_forecast_value.py --smoke
```
