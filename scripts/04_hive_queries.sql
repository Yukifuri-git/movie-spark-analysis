-- ============================================================================
-- Hive 数据仓库查询脚本 — 外部表 + 六条多维度分析SQL
-- ============================================================================
-- 对应论文：第7章（Hive数据仓库查询）
--
-- 在系统中的角色：
--   通过 hive -f 04_hive_queries.sql 以批处理模式提交执行
--   建表 → 多条SELECT分析 → CTAS物化表，覆盖聚合+多表关联+窗口函数+物化四种场景
--
-- 为什么用EXTERNAL TABLE：
--   外部表删除时只删元数据不删HDFS文件。我反复调试建表语句，如果用了内部表，
--   DROP TABLE的瞬间数据就没了。在Hive的Derby Metastore单用户模式下，
--   这个设计决策避免了"改一行代码就要重新上传数据"的尴尬。
--
-- 为什么有 skip.header.line.count=1：
--   CSV文件第一行是字段名（userId,movieId,rating,date）。如果不跳过表头，
--   "userId"这个字符串会被当作一条评分记录参与聚合，Hive会把它转成NULL或0，
--   导致结果多一行诡异的数据。这个属性在文档角落里，很容易漏掉。
-- ============================================================================

-- ==== 1. 创建评分外部表 ====
-- LOCATION指向HDFS目录而非单个文件——Spark分析时coalesce(1)写入的csv是
-- 单文件，但Hive建表时指目录更规范，后续扩展也方便
CREATE EXTERNAL TABLE IF NOT EXISTS ratings_ext (
    userId INT,
    movieId INT,
    rating DOUBLE,
    `date` STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/zonghuiyang/movie_data/ratings'
TBLPROPERTIES ("skip.header.line.count"="1");

-- ==== 2. 创建电影外部表 ====
CREATE EXTERNAL TABLE IF NOT EXISTS movies_ext (
    movieId INT,
    title STRING,
    `year` STRING,
    genres STRING,
    main_genre STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/zonghuiyang/movie_data/movies'
TBLPROPERTIES ("skip.header.line.count"="1");


-- ======================================================================
--  查询1：评分分布统计
-- ======================================================================
-- SUM(COUNT(*)) OVER() 窗口函数一次扫描同时得到绝对计数和百分比
-- 如果写成两条独立的SELECT（先count再算比例），要多扫一次全表
SELECT
    rating,
    COUNT(*) as cnt,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM ratings_ext
GROUP BY rating
ORDER BY rating DESC;


-- ======================================================================
--  查询2：各类型电影平均评分
-- ======================================================================
-- 只统计评分量>=100的类型——HAVING过滤小样本，和Spark分析三(min 50评)思路一致
-- 阈值提高到100是因为按类型聚合后单个类型的样本量比单部电影大得多
SELECT
    m.main_genre,
    COUNT(DISTINCT r.movieId) as movie_count,
    COUNT(r.rating) as rating_count,
    ROUND(AVG(r.rating), 2) as avg_rating
FROM ratings_ext r
JOIN movies_ext m ON r.movieId = m.movieId
WHERE m.main_genre != '(no genres listed)'
GROUP BY m.main_genre
HAVING COUNT(r.rating) >= 100
ORDER BY avg_rating DESC
LIMIT 10;


-- ======================================================================
--  查询3：年度电影产量与平均评分趋势
-- ======================================================================
-- 按电影发行年份(year字段，来自01预处理的正则提取)分组
-- 取最近20年——再往前的年份评分量太少，统计意义不大
SELECT
    m.`year`,
    COUNT(DISTINCT r.movieId) as movie_count,
    COUNT(r.rating) as rating_count,
    ROUND(AVG(r.rating), 2) as avg_rating
FROM ratings_ext r
JOIN movies_ext m ON r.movieId = m.movieId
WHERE m.`year` IS NOT NULL AND m.`year` != ''
GROUP BY m.`year`
ORDER BY m.`year` DESC
LIMIT 20;


-- ======================================================================
--  查询4：高分电影筛选（≥100评，均分≥4.0）
-- ======================================================================
-- 两个硬约束：评分量>=100保证统计可靠性，均分>=4.0保证"实至名归"
-- GROUP BY里包含了movieId——标题可能重复但ID唯一，这是更严谨的写法
SELECT
    m.title,
    m.main_genre,
    COUNT(r.rating) as rating_count,
    ROUND(AVG(r.rating), 2) as avg_rating
FROM ratings_ext r
JOIN movies_ext m ON r.movieId = m.movieId
GROUP BY m.movieId, m.title, m.main_genre
HAVING COUNT(r.rating) >= 100 AND AVG(r.rating) >= 4.0
ORDER BY avg_rating DESC, rating_count DESC;


-- ======================================================================
--  查询5：最活跃用户TOP10
-- ======================================================================
-- 除了评分数量，还统计了AVG(用户均分)和COUNT(DISTINCT movieId)(观影多样性)
-- 活跃不代表打分松或口味单一——从实际结果看这类用户均分在3.0-3.5之间很正常
SELECT
    r.userId,
    COUNT(r.rating) as rating_count,
    ROUND(AVG(r.rating), 2) as avg_user_rating,
    COUNT(DISTINCT r.movieId) as movie_variety
FROM ratings_ext r
GROUP BY r.userId
ORDER BY rating_count DESC
LIMIT 10;


-- ======================================================================
--  查询6：CTAS创建分析物化表
-- ======================================================================
-- CTAS (CREATE TABLE AS SELECT)：将频繁查询的聚合结果物化，后续直接查这张表
-- 适合：movie_stats这种会被多个下游查询用到的中间结果
-- 不适合：一次性查询——建表本身有写HDFS开销
CREATE TABLE IF NOT EXISTS movie_stats AS
SELECT
    m.movieId,
    m.title,
    m.main_genre,
    COUNT(r.rating) as rating_cnt,
    ROUND(AVG(r.rating), 2) as avg_rating,
    MIN(r.rating) as min_rating,
    MAX(r.rating) as max_rating
FROM ratings_ext r
JOIN movies_ext m ON r.movieId = m.movieId
GROUP BY m.movieId, m.title, m.main_genre;

-- 从物化表直接查——比每次JOIN+GROUP BY快，代价是占用额外磁盘空间
SELECT * FROM movie_stats WHERE rating_cnt >= 100 ORDER BY avg_rating DESC LIMIT 10;

