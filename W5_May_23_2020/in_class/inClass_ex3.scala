import org.apache.log4j._
import org.apache.spark.sql.SparkSession
object wordCount {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val session = SparkSession
      .builder
      .master("local[2]")
      .appName("wordCount")
      .getOrCreate()
    val input = session.read.textFile("TheHungerGames.txt")
    import session.implicits._
    val words = input.flatMap(x => x.split(" "))
  //println(words.getClass)
/*	
    val wordCounts = words.rdd.countByValue()
    wordCounts.foreach(println)
  }
  def clean(s: String): String =
  {
    s.replaceAll("[^\\w\\s]", "")
  }
*/
}
