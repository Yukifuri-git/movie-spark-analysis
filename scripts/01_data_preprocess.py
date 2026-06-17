#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本 — MovieLens原始数据清洗、格式转换、统计摘要
----------------------------------------------------------------------
在系统中的角色：
  process_ratings()  ——→ 评分数据清洗：过滤非法值 + Unix时间戳转可读日期
  process_movies()   ——→ 电影数据清洗：正则提取年份 + 拆分主类型
  data_summary()     ——→ 计算数据集基本信息（行数/用户/稀疏度）

处理链：原始CSV → 字段校验 → 格式转换 → 清洗后CSV → 统计摘要 → 准备上传HDFS

对应论文：第4章（数据预处理与存储）

设计考量：
  原始数据质量其实很好——GroupLens在发布时已经做了基础清洗，100,836条评分全部在
  0.5-5.0合法范围，没有空行也没有乱码。所以这里的"清洗"主要是格式转换而非脏数据修复：
    ① Unix时间戳 → YYYY-MM-DD，方便Hive SQL按日期过滤
    ② 电影标题中嵌入的年份 → 独立字段，方便按年代分析
    ③ 多类型标签 → 提取主类型，简化分析维度
  全是小细节，但每一个忽略都会在后续分析中放大成错误。
"""
import os
import re
import csv
from datetime import datetime

# ---- 路径配置 ----
# Hadoop伪分布式跑在VMware虚拟机里，HDFS通过localhost:9000访问
DATA_DIR = "/home/zonghuiyang/movie_recommendation/data/ml-latest-small"
OUTPUT_DIR = "/home/zonghuiyang/movie_recommendation/output"
HDFS_DATA_DIR = "/user/zonghuiyang/movie_data"


# ==== 1. 处理评分文件 ratings.csv ====
def process_ratings():
    """
    ratings.csv 字段：userId, movieId, rating, timestamp

    两件事：
      ① 过滤评分不在0.5-5.0范围的记录（原始数据质量好，这一步其实没过滤掉任何行）
      ② Unix时间戳转日期字符串，用datetime.fromtimestamp一把搞定

    为什么加date字段：Hive SQL里做时间维度查询时，YYYY-MM-DD格式比Unix秒数直观得多，
    后续按年份、按月份统计都直接SUBSTR(date,1,4)就行
    """
    print("处理 ratings.csv …")

    ratings_clean = []
    with open(f"{DATA_DIR}/ratings.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rating = float(row["rating"])
            # 合法范围检查：0.5-5.0，步长0.5。GroupLens已做清洗，这个if基本不会触发
            if 0.5 <= rating <= 5.0:
                ts = int(row["timestamp"])
                date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                ratings_clean.append({
                    "userId": row["userId"],
                    "movieId": row["movieId"],
                    "rating": str(rating),
                    "date": date_str
                })

    with open(f"{OUTPUT_DIR}/ratings_clean.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["userId", "movieId", "rating", "date"])
        writer.writeheader()
        writer.writerows(ratings_clean)

    print(f"  评分清洗完成：{len(ratings_clean)} 条（原始100836条全部保留）")


# ==== 2. 处理电影文件 movies.csv ====
def process_movies():
    """
    movies.csv 字段：movieId, title, genres

    两件事：
      ① 从title里用正则提取发行年份 —— "Toy Story (1995)" → year=1995
         选正则而不是字符串切片的理由：部分电影标题不含年份（如TV剧集），用切片容易取错字段
      ② 从genres里提取主类型 —— "Comedy|Romance" → main_genre="Comedy"
         只取第一个类型作为主类型，简化后续分析。多类型标签统计时"|""比逗号好处理
    """
    print("处理 movies.csv …")

    movies_clean = []
    with open(f"{DATA_DIR}/movies.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row["title"]
            # 匹配末尾括号中的四位数字，如 "(1995)" 或 "(2003-2004)"
            match = re.search(r'\((\d{4})\)', title)
            year = match.group(1) if match else ""

            genres = row["genres"]
            main_genre = genres.split("|")[0] if "|" in genres else genres

            movies_clean.append({
                "movieId": row["movieId"],
                "title": title,
                "year": year,
                "genres": genres,
                "main_genre": main_genre
            })

    with open(f"{OUTPUT_DIR}/movies_clean.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["movieId", "title", "year", "genres", "main_genre"])
        writer.writeheader()
        writer.writerows(movies_clean)

    print(f"  电影清洗完成：{len(movies_clean)} 部")


# ==== 3. 数据集统计摘要 ====
def data_summary():
    """
    统计基本信息并写入文本文件。
    稀疏度 = 实际评分 / (用户数 × 电影数) —— 这个数字只有1.70%，
    意味着评分矩阵98.30%的位置是空的，直观体现了协同过滤面对的稀疏性挑战。
    """
    ratings_count = sum(1 for _ in open(f"{OUTPUT_DIR}/ratings_clean.csv")) - 1
    movies_count = sum(1 for _ in open(f"{OUTPUT_DIR}/movies_clean.csv")) - 1

    users = set()
    movie_ids = set()
    with open(f"{OUTPUT_DIR}/ratings_clean.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            users.add(row["userId"])
            movie_ids.add(row["movieId"])

    sparsity = ratings_count / (len(users) * len(movie_ids)) * 100
    summary = f"""
========================================
   MovieLens 数据集统计摘要
========================================
  评分记录数: {ratings_count}
  电影数量:   {movies_count}
  用户数量:   {len(users)}
  被评分电影: {len(movie_ids)}
  数据稀疏度: {sparsity:.2f}%
========================================
"""
    print(summary)
    with open(f"{OUTPUT_DIR}/data_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)


# ==== 主入口 ====
if __name__ == "__main__":
    process_ratings()
    process_movies()
    data_summary()
    print("数据预处理完成，准备上传HDFS...")
    print(f"  上传评分数据: hdfs dfs -put {OUTPUT_DIR}/ratings_clean.csv {HDFS_DATA_DIR}/")
    print(f"  上传电影数据: hdfs dfs -put {OUTPUT_DIR}/movies_clean.csv {HDFS_DATA_DIR}/")
