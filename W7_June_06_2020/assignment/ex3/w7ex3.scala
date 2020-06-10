// Databricks notebook source
// MAGIC %fs ls dbfs:/FileStore/tables/

// COMMAND ----------

// MAGIC %scala
// MAGIC import org.apache.spark.sql.SparkSession
// MAGIC import org.apache.spark.sql.functions._
// MAGIC import org.apache.spark.util.SizeEstimator

// COMMAND ----------

// MAGIC %scala
// MAGIC def main(args: Array[String]): Unit = 
// MAGIC   {
// MAGIC     val spark = SparkSession
// MAGIC       .builder
// MAGIC       .master("local[2]")
// MAGIC       .appName("Insurance")
// MAGIC       .config("spark.some.config.option", "some-value")
// MAGIC       .getOrCreate()
// MAGIC   }

// COMMAND ----------

// MAGIC %md #### Read insurance.csv file

// COMMAND ----------

// MAGIC %scala
// MAGIC val df = spark.read.options(Map("inferSchema" -> "true", "delimiter" -> ",", "header" -> "true")).csv("dbfs:/FileStore/tables/insurance.csv")
// MAGIC val cols = df.columns
// MAGIC for (col <- cols) {println(col)}

// COMMAND ----------

// MAGIC %md #### Print the size in memory and size of records

// COMMAND ----------

// MAGIC %md #### Length of records

// COMMAND ----------

// MAGIC %scala
// MAGIC val rec_count = df.count()
// MAGIC println(s"Length of records: $rec_count rows")

// COMMAND ----------

// MAGIC %md #### Size in memory

// COMMAND ----------

// MAGIC  %scala
// MAGIC  val mem_size = SizeEstimator.estimate(df)
// MAGIC  println(s"Memory size: $mem_size bytes")

// COMMAND ----------

// MAGIC %md #### Print sex and count of sex (use group by in sql)

// COMMAND ----------

// MAGIC %scala 
// MAGIC df.groupBy("sex").count().show()

// COMMAND ----------

// MAGIC %md #### Filter smoker=yes and print again the sex,count of sex

// COMMAND ----------

// MAGIC %scala 
// MAGIC df.filter("smoker == 'yes'").groupBy("sex").count().show()

// COMMAND ----------

// MAGIC %md #### Group by region and sum the charges (in each region), then print rows by descending order (with respect to sum)

// COMMAND ----------

// MAGIC %scala
// MAGIC df.groupBy("region").sum("charges").orderBy(desc("sum(charges)")).show()
