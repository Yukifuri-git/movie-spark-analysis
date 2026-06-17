#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据可视化脚本 — 从分析结果生成五张论文级图表
----------------------------------------------------------------------
对应论文：第8章（数据可视化）

在系统中的角色：
  Spark/Hive分析完成 → 结果数值硬编码或从HDFS读取 → Matplotlib绑图 → PNG输出

五张图：
  图1 评分分布柱状图  — 展示评分行为的右偏特征（4.0分最多）
  图2 热门电影TOP10    — 横向柱状图，同时呈现"热度"和"均分"
  图3 类型分布饼图    — Comedy+Drama合占约50%
  图4 年份趋势双轴图   — 折线(均分)+柱状(评分量)，双Y轴设计
  图5 ALS模型评估      — RMSE+训练/测试集量

绑图设计考量：
  ① 选横向柱状图而非纵向 —— "Forrest Gump"这种电影名太长，纵向标签必重叠
  ② 图4用双Y轴 —— 均分(折线, 3.3-3.6)和数量(柱状, 91-2498)量级差两个数量级，
     单Y轴的话低量级指标会压成一条贴着底边的直线
  ③ 图3饼图标注百分比 —— 一眼看出Comedy(28.0%)和Drama(22.4%)的占比差距
  ④ 无GUI后端 Agg —— 虚拟机上Matplotlib默认TkAgg在SSH连接下会报错
     "no display name and no $DISPLAY environment variable"
"""
import os
import matplotlib
matplotlib.use('Agg')            # 无GUI后端，SSH环境必需
import matplotlib.pyplot as plt
import numpy as np

# 中文字体：Ubuntu下用DejaVu Sans（无中文字体时英文标注即可，论文里用英文图表）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "/home/zonghuiyang/movie_recommendation/output"


# ======================================================================
#  图1：评分分布柱状图 — 横轴10个评分档位，纵轴数量
# ======================================================================
def plot_rating_distribution():
    """
    数据来源：02_spark_analysis.scala分析一 → HDFS output/rating_distribution
    手工复制到这里的硬编码数值（为了方便绑定图，避免每次从HDFS解析CSV）
    """
    print("生成评分分布图...")

    ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    counts = [1370, 2811, 1791, 7551, 5550, 20047, 13136, 26818, 8551, 13211]

    fig, ax = plt.subplots(figsize=(10, 6))
    # 红绿色谱渐变——低分偏红，高分偏绿，视觉上直观体现分布
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 10))
    bars = ax.bar(ratings, counts, color=colors, width=0.3, edgecolor='black')

    # 每根柱体上方标注数值
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                str(count), ha='center', fontsize=9)

    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Fig.1 Rating Distribution in MovieLens Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(ratings)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_rating_distribution.png", dpi=150)
    plt.close()
    print("  完成 → fig1_rating_distribution.png")


# ======================================================================
#  图2：热门电影TOP10 — 横向柱状图（barh），适合长文本标签的场景
# ======================================================================
def plot_popular_movies():
    """
    为什么选barh：Forrest Gump / Shawshank Redemption 这些片名有15-20个字符，
    纵向柱状图的X轴标签要么旋转45度（阅读困难），要么被截断。横向图完美兼容长标签。
    每个柱体旁标注 "评分次数 (均分)"，同时呈现"热度"和"质量"两个维度。
    """
    print("生成热门电影TOP10图...")

    movies = [
        "Forrest Gump", "Shawshank Redemption", "Pulp Fiction",
        "Silence of the Lambs", "Matrix", "Star Wars: Episode IV",
        "Jurassic Park", "Braveheart", "Terminator 2", "Schindler's List"
    ]
    counts = [329, 317, 307, 279, 278, 251, 238, 237, 224, 220]
    avg_ratings = [4.16, 4.43, 4.20, 4.16, 4.19, 4.23, 3.75, 4.03, 3.97, 4.23]

    fig, ax = plt.subplots(figsize=(10, 7))
    # 从底向上画，y_pos = range(len-1, -1, -1)
    y_pos = range(len(movies)-1, -1, -1)
    bars = ax.barh(y_pos, counts, color=plt.cm.Blues(np.linspace(0.4, 0.9, 10)), edgecolor='black')

    for i, (bar, count, rating) in enumerate(zip(bars, counts, avg_ratings)):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                f'{count} (avg:{rating})', va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(movies)
    ax.set_xlabel('Number of Ratings', fontsize=12)
    ax.set_title('Fig.2 Top 10 Most Rated Movies', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 400)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_popular_movies.png", dpi=150)
    plt.close()
    print("  完成 → fig2_popular_movies.png")


# ======================================================================
#  图3：电影类型分布饼图 — 10种主类型占比
# ======================================================================
def plot_genre_distribution():
    """
    数据来源：02_spark_analysis.scala分析四 → HDFS output/genre_distribution
    Comedy(28.0%)和Drama(22.4%)合计占了TOP10的一半——MovieLens的片库偏向叙事类电影
    """
    print("生成类型分布饼图...")

    genres = ["Comedy", "Drama", "Action", "Adventure", "Crime",
              "Horror", "Documentary", "Animation", "Children", "Thriller"]
    counts = [2779, 2226, 1828, 653, 537, 468, 386, 298, 197, 84]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    wedges, texts, autotexts = ax.pie(counts, labels=genres, autopct='%1.1f%%',
                                       colors=colors, startangle=90, pctdistance=0.85)

    for autotext in autotexts:
        autotext.set_fontsize(8)

    ax.set_title('Fig.3 Movie Genre Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_genre_pie.png", dpi=150)
    plt.close()
    print("  完成 → fig3_genre_pie.png")


# ======================================================================
#  图4：年份-评分趋势双轴图 — 折线(左Y轴: 均分) + 柱状(右Y轴: 评分量)
# ======================================================================
def plot_year_trend():
    """
    双Y轴设计的必要性：
      年均评分在3.4-3.6之间波动（变化幅度仅0.2），而评分量从2498掉到91（变化25倍+）
      如果共用Y轴，评分量的柱状图会正常显示但均分折线几乎是一条直线——完全看不出3.36到3.58的波动
      双轴能让两个量纲差两个数量级的指标在同一张图里和谐共存
    """
    print("生成年份-评分趋势图...")

    years = ['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']
    avg_ratings = [3.36, 3.50, 3.52, 3.53, 3.57, 3.57, 3.46, 3.51, 3.46, 3.51, 3.41, 3.39, 3.58, 3.48]
    counts = [2498, 2584, 2316, 2149, 1856, 1715, 1438, 1385, 1201, 1318, 1088, 785, 461, 91]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左Y轴：折线 — 均分
    color1 = 'tab:blue'
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Average Rating', color=color1, fontsize=12)
    line1 = ax1.plot(years, avg_ratings, 'o-', color=color1, linewidth=2, markersize=8, label='Avg Rating')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(3.0, 4.0)

    # 右Y轴：柱状 — 评分量
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Rating Count', color=color2, fontsize=12)
    bars = ax2.bar(years, counts, alpha=0.3, color=color2, label='Count')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines = line1 + [bars]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title('Fig.4 Rating Trends by Movie Release Year', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_year_trend.png", dpi=150)
    plt.close()
    print("  完成 → fig4_year_trend.png")


# ======================================================================
#  图5：ALS模型评估指标 — RMSE+训练/测试集量
# ======================================================================
def plot_als_evaluation():
    """
    三根柱子展示三个核心指标：RMSE(0.873)、训练集(80,578)、测试集(20,258)
    RMSE=0.873意味着预测评分与真实评分平均偏差约0.87分——在4.5分跨度且稀疏度1.70%的
    背景下是合理的。如果把regParam改成1.0强制正则化，RMSE反而会飙到0.95+，
    说明当前参数达到了偏差-方差的合理平衡。
    """
    print("生成ALS模型评估图...")

    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = ['RMSE', 'Training Set', 'Test Set']
    values = [0.873, 80578, 20258]
    colors_bar = ['#e74c3c', '#3498db', '#2ecc71']

    bars = ax.bar(metrics, values, color=colors_bar, edgecolor='black')
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Fig.5 ALS Model Performance Metrics', fontsize=14, fontweight='bold')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                str(val), ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_als_evaluation.png", dpi=150)
    plt.close()
    print("  完成 → fig5_als_evaluation.png")


# ==== 主入口 ====
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 50)
    print("  数据可视化图表生成")
    print("=" * 50)

    plot_rating_distribution()
    plot_popular_movies()
    plot_genre_distribution()
    plot_year_trend()
    plot_als_evaluation()

    print("五张图表全部生成完毕")
    print(f"  输出目录: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"    - {f}")
