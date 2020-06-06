import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.util.SizeEstimator

case class Record(key: Int, value: String)

object Insurance {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[2]")
      .appName("Insurance")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    //1. Read insurance.csv file
    val df = spark.read.options(Map("inferSchema" -> "true", "delimiter" -> ",", "header" -> "true")).csv("data/insurance.csv")
    val cols = df.columns
    for (col <- cols) {println(col)}

    //2. Print the size
    // record length
    val rec_count = df.count()
    println(s"Length of records: $rec_count rows")
    // size in mem
    val mem_size = SizeEstimator.estimate(df)
    println(s"Memory size: $mem_size bytes")

    //3. Print sex and count of sex (use group by in sql)
    df.groupBy("sex").count().show()

    //4. Filter smoker=yes and print again the sex,count of sex
        df.filter("smoker == 'yes'").groupBy("sex").count().show()

    //5. Group by region and sum the charges (in each region), then print rows by descending order (with respect to sum)
    df.groupBy("region").sum("charges").orderBy(desc("sum(charges)")).show()


    spark.stop()
  }

}
