/*
 * ALS 协同过滤推荐脚本 — 训练、评估、生成Top-5推荐
 * ============================================================================
 * 对应论文：第6章（ALS协同过滤推荐）
 *
 * 在系统中的角色：
 *   spark-shell --master local[4] 提交运行（注意：local不是yarn，原因见下）
 *   读评分数据 → 8:2划分训练/测试 → 训练ALS → RMSE评估 → 生成用户推荐和电影相似推荐
 *
 * ALS 核心思路：
 *   将用户-物品评分矩阵 R(m×n) 分解为 U×V^T
 *   其中U(m×k)是用户隐因子矩阵，V(n×k)是电影隐因子矩阵
 *   k(rank)是隐因子维度——k小容易欠拟合，k大容易过拟合且内存暴涨
 *   ALS通过交替固定U优化V、固定V优化U的方式迭代求解
 *
 * 为什么用 local[4] 而不是 yarn：
 *   ALS是迭代型算法，每次迭代都要在driver和executor之间传输隐因子矩阵。在8GB虚拟机的
 *   伪分布式环境下，YARN的调度和序列化开销反而拖慢了训练。"local[4]让整个训练在单JVM内
 *   完成，避免了跨节点的通信成本。这是受限环境下的务实选择——如果内存足够大、数据量百万+，
 *   那YARN当然是更好的决定。但跑10万条评分数据，local更快。
 *
 * 参数选择记录（受限于时间没做网格搜索，靠理解和手动调试）：
 *   rank=20     → 20维隐特征，再大会过拟合且内存不够
 *   regParam=0.1 → 尝试过0.01(RMSE略高)和1.0(RMSE飙到0.95+)，0.1是折中点
 *   maxIter=10  → 10次后RMSE不再明显下降，loss曲线趋于平坦
 *   nonnegative=true → 评分没有负值，约束非负更符合物理含义
 * ============================================================================
 */

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator

// ==== 1. 创建SparkSession，local[4]本地模式 ====
val spark = SparkSession.builder()
  .appName("MovieLens-ALS-Recommendation")
  .master("local[4]")
  .getOrCreate()

import spark.implicits._

println("=" * 60)
println("  基于ALS的电影推荐系统")
println("=" * 60)

// ==== 2. 从HDFS读取评分数据，字段类型对齐 ====
// userId和movieId必须cast成int——ALS的setUserCol/setItemCol要求IntegerType
val ratingsDF = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("hdfs://localhost:9000/user/zonghuiyang/movie_data/ratings_clean.csv")

val ratingsClean = ratingsDF.select(
  col("userId").cast("int"),
  col("movieId").cast("int"),
  col("rating").cast("double")
).filter(col("rating") > 0)       // 过滤可能的零值/负值（原始数据里其实不存在）

println(s"训练数据总量: ${ratingsClean.count()} 条")


// ==== 3. 划分训练集和测试集（8:2） ====
// seed=42固定随机种子保证结果可复现。80:20是推荐系统常用的划分比例
val Array(trainData, testData) = ratingsClean.randomSplit(Array(0.8, 0.2), seed=42)
println(s"训练集: ${trainData.count()}, 测试集: ${testData.count()}")


// ==== 4. 训练ALS模型 ====
// coldStartStrategy="drop"：预测阶段遇到训练集没见过的用户/物品直接丢弃，
// 不强行返回一个均值预测（那样RMSE值会好看但评估没有意义）
val als = new ALS()
  .setMaxIter(10)
  .setRegParam(0.1)
  .setRank(20)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating")
  .setColdStartStrategy("drop")
  .setNonnegative(true)

println("开始训练ALS模型（rank=20, regParam=0.1, maxIter=10）...")
val model = als.fit(trainData)
println("模型训练完成")


// ==== 5. RMSE评估 ====
// 选RMSE的理由：平方根把误差还原回"分"的单位，说"预测偏差约0.87分"比说"MSE=0.76"直观
// 0.873在4.5分跨度(0.5~5.0)+数据稀疏度仅1.70%的背景下，是合理的结果
val predictions = model.transform(testData)
val evaluator = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("rating")
  .setPredictionCol("prediction")

val rmse = evaluator.evaluate(predictions)
println(s"测试集RMSE (均方根误差): ${"%.4f".format(rmse)}")


// ==== 6. 为所有用户生成Top-5个性化推荐 ====
// recommendForAllUsers一次性为全部610个用户生成推荐，展开后取前10用户展示
// 结果是(userId, Array[(movieId, predictedRating)])，需要explode炸开

println("\n" + "=" * 40)
println("  用户Top-5推荐（展示前10个用户）")
println("=" * 40)

val userRecs = model.recommendForAllUsers(5)
val sampleUsers = userRecs.orderBy("userId").limit(10)

// 读电影名称档
val moviesDF = spark.read
  .option("header", "true")
  .csv("hdfs://localhost:9000/user/zonghuiyang/movie_data/movies_clean.csv")

// 展开推荐结构 → 关联电影名 → 按用户+预测评分降序展示
sampleUsers.select(col("userId"), explode(col("recommendations")))
  .select(col("userId"), col("col.movieId").as("movieId"),
          round(col("col.rating"), 2).as("predictedRating"))
  .join(moviesDF.select("movieId", "title"), "movieId")
  .select("userId", "title", "predictedRating")
  .orderBy(col("userId").asc, col("predictedRating").desc)
  .show(50, false)


// ==== 7. 为电影生成相似电影Top-5 ====
// recommendForAllItems基于隐因子向量的余弦相似度找相似物品
// 注意：不同用户推荐结果可能高度重叠——这是协同过滤的热门偏差问题
// ALS学到的隐因子偏向"大众口味"，所以个性化程度有限

println("\n" + "=" * 40)
println("  电影相似推荐（展示前5个电影）")
println("=" * 40)

val movieRecs = model.recommendForAllItems(5)

// 取前5个电影，展开相似推荐，关联两边电影名
val explodedMovieRecs = movieRecs.orderBy("movieId").limit(5)
  .select(col("movieId").as("origMovieId"), explode(col("recommendations")).as("rec"))
  .select(col("origMovieId"), col("rec.movieId").as("similarMovieId"),
          round(col("rec.rating"), 2).as("similarityScore"))

val resultDF = explodedMovieRecs
  .join(moviesDF.select(col("movieId").as("origMovieId"),
         col("title").as("origTitle")), "origMovieId")
  .join(moviesDF.select(col("movieId").as("similarMovieId"),
         col("title").as("similarTitle")), "similarMovieId")
  .select("origTitle", "similarTitle", "similarityScore")
  .orderBy(col("origTitle").asc, col("similarityScore").desc)

resultDF.show(30, false)


// ==== 8. 保存模型到HDFS ====
// 生产环境中save后可以load复用，不用每次重新训练
model.write.overwrite().save("hdfs://localhost:9000/user/zonghuiyang/movie_data/output/als_model")

println("ALS推荐流程完成，模型已保存")
spark.stop()
