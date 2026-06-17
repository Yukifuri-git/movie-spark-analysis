/*
 * Spark SQL 多维分析脚本 — 从HDFS读数据，六个维度统计分析
 * ============================================================================
 * 对应论文：第5章（Spark多维数据分析）
 *
 * 在系统中的角色：
 *   spark-shell --master yarn --deploy-mode client 提交运行
 *   从HDFS读取预处理后的评分和电影数据 → 六个维度逐一分析 → 结果写回HDFS
 *
 * 六个分析维度：
 *   ① 评分分布     → 每个分值的数量，了解整体评分行为
 *   ② 热门电影TOP10 → 按评分次数排名，前三都是1994年电影
 *   ③ 高分电影TOP10 → 按均分排名但限制最少50评，避免小样本偏差
 *   ④ 类型分布TOP10 → 按主类型计数，Comedy和Drama合占约50%
 *   ⑤ 用户活跃度     → 人均/最多/最少评分数，评分行为呈长尾分布
 *   ⑥ 年份趋势     → 按发行年统计，评分量逐年下降但均分稳定
 *
 * 为什么选YARN模式而非local：分析类任务SQL操作多，单次扫描数据量大但内存消耗可控，
 * YARN模式下Spark可以充分利用分布式计算优势。这和ALS推荐用local[4]是不同判断，
 * 见03_als_recommend.scala的说明。
 * ============================================================================
 */

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

// ==== 1. 创建SparkSession ====
// YARN模式提交——分析阶段数据量大但计算模式简单（groupBy+agg），适合分布式
val spark = SparkSession.builder()
  .appName("MovieLens-MultiDimension-Analysis")
  .master("yarn")
  .getOrCreate()

import spark.implicits._

println("=" * 60)
println("  基于Spark的电影评分多维分析")
println("=" * 60)

// ==== 2. 从HDFS读取数据 ====
// 数据预处理在01_data_preprocess.py中完成，结果上传到HDFS的movie_data目录
val ratingsDF = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("hdfs://localhost:9000/user/zonghuiyang/movie_data/ratings_clean.csv")

val moviesDF = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("hdfs://localhost:9000/user/zonghuiyang/movie_data/movies_clean.csv")

println(s"评分数据行数: ${ratingsDF.count()}")
println(s"电影数据行数: ${moviesDF.count()}")

// 缓存：两个DataFrame后续六个分析都要用，cache避免每次从HDFS重新扫描
ratingsDF.cache()
moviesDF.cache()

// ======================================================================
//  分析一：评分分布统计
// ======================================================================
// 每个分值(0.5~5.0)出现了多少次——这是分析的起点，先看整体评分行为模式
// 结果：4.0分档位最高(26,818条，占26.6%)，3.0分(20,047条)次之，整体呈右偏分布
// 这符合一般评分习惯——用户倾向于打整数分（3.0、4.0最多），半分的频次明显少于整分

println("\n" + "=" * 40)
println("  分析一：评分分布统计")
println("=" * 40)

val ratingDistribution = ratingsDF.groupBy("rating")
  .count()
  .orderBy(desc("rating"))

println("评分值 | 数量")
ratingDistribution.collect().foreach { row =>
  println(f"  ${row.getDouble(0)}%.1f  | ${row.getLong(1)}")
}

ratingDistribution.coalesce(1)
  .write.mode("overwrite")
  .option("header", "true")
  .csv("hdfs://localhost:9000/user/zonghuiyang/movie_data/output/rating_distribution")


// ======================================================================
//  分析二：热门电影TOP10（按评分次数排名）
// ======================================================================
// 用评分次数衡量"热度"而非"质量"——看被人讨论最多的电影，不一定是评分最高的
// 结果前三：Forrest Gump(329评)、Shawshank Redemption(317评)、Pulp Fiction(307评)
// 巧的是这三部都是1994年发行——1994年确实是电影史上很特殊的一年

println("\n" + "=" * 40)
println("  分析二：热门电影 TOP10")
println("=" * 40)

val popularMovies = ratingsDF.groupBy("movieId")
  .agg(
    count("rating").as("rating_count"),
    round(avg("rating"), 2).as("avg_rating")
  )
  .join(moviesDF, "movieId")
  .select("title", "rating_count", "avg_rating")
  .orderBy(desc("rating_count"))
  .limit(10)

popularMovies.show(false)

popularMovies.coalesce(1)
  .write.mode("overwrite")
  .option("header", "true")
  .csv("hdfs://localhost:9000/user/zonghuiyang/movie_data/output/popular_movies")


// ======================================================================
//  分析三：高分电影TOP10（均分最高，最少50次评分）
// ======================================================================
// 关键设计：加 filter(count >= 50)。不加的话3人评5.0的冷门电影排第一——
// 样本量不足的"高分"在统计上没有意义。50不是黄金数字，但对9000+电影的数据集
// 过滤掉评分数<50的电影后仍有足够的候选，这个阈值是合理的

println("\n" + "=" * 40)
println("  分析三：高分电影 TOP10（≥50次评分）")
println("=" * 40)

val topRatedMovies = ratingsDF.groupBy("movieId")
  .agg(
    count("rating").as("rating_count"),
    round(avg("rating"), 2).as("avg_rating")
  )
  .filter(col("rating_count") >= 50)       // 过滤低样本，保证统计可信度
  .join(moviesDF, "movieId")
  .select("title", "rating_count", "avg_rating")
  .orderBy(desc("avg_rating"))
  .limit(10)

topRatedMovies.show(false)

topRatedMovies.coalesce(1)
  .write.mode("overwrite")
  .option("header", "true")
  .csv("hdfs://localhost:9000/user/zonghuiyang/movie_data/output/top_rated_movies")


// ======================================================================
//  分析四：电影类型分布TOP10
// ======================================================================
// 按主类型（genres字段的第一个标签）统计数量，过滤掉"(no genres listed)"
// 结果：Comedy(2779部)和Drama(2226部)是数量最多的两大类型，合计约占TOP10的45%+
// 使用main_genre字段而非genres字段——01预处理时已经拆分好了主类型，直接用

println("\n" + "=" * 40)
println("  分析四：电影类型分布 TOP10")
println("=" * 40)

val genreDistribution = moviesDF
  .filter(col("main_genre") =!= "(no genres listed)")
  .groupBy("main_genre")
  .count()
  .orderBy(desc("count"))
  .limit(10)

genreDistribution.show(false)

genreDistribution.coalesce(1)
  .write.mode("overwrite")
  .option("header", "true")
  .csv("hdfs://localhost:9000/user/zonghuiyang/movie_data/output/genre_distribution")


// ======================================================================
//  分析五：用户活跃度分析
// ======================================================================
// 人均165.3条评分，但中位数远低于均值——典型的长尾分布
// 最活跃用户评了2398条，而大部分用户只评几十条，这个差异在推荐系统里叫"用户偏差"
// ALS建模时不做特殊处理（冷启动drop掉即可），但知道数据有这个特征有助于理解结果

println("\n" + "=" * 40)
println("  分析五：用户活跃度统计")
println("=" * 40)

val userActivity = ratingsDF.groupBy("userId")
  .count()
  .withColumnRenamed("count", "rating_count")

val avgRatingsPerUser = userActivity.agg(round(avg("rating_count"), 1)).first().getDouble(0)
val maxRatingsPerUser = userActivity.agg(max("rating_count")).first().getLong(0)
val minRatingsPerUser = userActivity.agg(min("rating_count")).first().getLong(0)

println(f"  平均每位用户评分数: $avgRatingsPerUser")
println(f"  最多评分: $maxRatingsPerUser 条")
println(f"  最少评分: $minRatingsPerUser 条")

userActivity.coalesce(1)
  .write.mode("overwrite")
  .option("header", "true")
  .csv("hdfs://localhost:9000/user/zonghuiyang/movie_data/output/user_activity")


// ======================================================================
//  分析六：年份-评分趋势（取近15年）
// ======================================================================
// 将ratings和movies做JOIN，按发行年(year)分组
// 评分量随年份递减是合理的——越新的电影累计评分自然越少，不是"人气下降"
// 但年均分稳定在3.4-3.6之间说明用户的评价标准比较一致，不因年代偏好而明显波动

println("\n" + "=" * 40)
println("  分析六：电影年份评分趋势（部分年份）")
println("=" * 40)

val yearRatings = ratingsDF
  .join(moviesDF, "movieId")
  .filter(col("year").isNotNull && col("year") =!= "")
  .groupBy("year")
  .agg(
    count("rating").as("count"),
    round(avg("rating"), 2).as("avg_rating")
  )
  .orderBy(desc("year"))
  .limit(15)

yearRatings.show(false)

yearRatings.coalesce(1)
  .write.mode("overwrite")
  .option("header", "true")
  .csv("hdfs://localhost:9000/user/zonghuiyang/movie_data/output/year_ratings")


// ======================================================================
//  9. 汇总统计 —— 六个维度的核心数字汇在一起，方便在控制台和Hive端核对
// ======================================================================

println("\n" + "=" * 40)
println("  分析汇总")
println("=" * 40)

val totalRatings = ratingsDF.count()
val totalMovies = moviesDF.count()
val totalUsers = ratingsDF.select("userId").distinct().count()
val avgRating = ratingsDF.agg(round(avg("rating"), 2)).first().getDouble(0)

println(f"  总评分数:    $totalRatings")
println(f"  总电影数:    $totalMovies")
println(f"  总用户数:    $totalUsers")
println(f"  整体平均分:  $avgRating")

// 释放缓存 → 停止Spark
ratingsDF.unpersist()
moviesDF.unpersist()
spark.stop()
println("分析结束 — 六个维度的结果已写回HDFS output目录")
