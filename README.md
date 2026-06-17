# 基于Spark框架的电影评分数据多维分析与推荐系统

## 项目简介
使用 Apache Spark 对 MovieLens 电影评分数据集进行多维分析，并基于 ALS 协同过滤算法构建电影推荐系统。

## 技术栈
- **Spark SQL** — 多维数据分析（6个维度）
- **Spark MLlib** — ALS 协同过滤推荐（RMSE=0.873）
- **Hive** — 数据仓库查询
- **Matplotlib** — 可视化图表

## 数据集
MovieLens ml-latest-small：100,836 条评分，9,742 部电影，610 个用户

## 项目结构
- `scripts/` — 源代码（预处理/分析/推荐/Hive/可视化）
- `charts/` — 分析图表
- `screenshots/` — 运行截图
- `output/` — 清洗后数据
- 讲解视频在压缩包里，需要密码
