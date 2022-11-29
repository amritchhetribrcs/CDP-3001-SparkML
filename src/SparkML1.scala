import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.ml.regression.LinearRegression

object SparkML1 {

  def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession.builder()
      .master("local[3]")
      .appName("Spark ML Example-1")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print(spark)
    //Loading Data sets
    val training = spark.read.format("libsvm")
      .load("C:\\InputFile.txt")
    val lRegression = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fitting Model
    val linearRegressionModel = lRegression.fit(training)
    val trainingSummary = linearRegressionModel.summary
    println(trainingSummary.totalIterations)

  }
}
