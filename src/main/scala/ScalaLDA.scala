import java.io.{PrintWriter, BufferedWriter, FileWriter}

import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Matrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

object ScalaLDA {
  def main(args: Array[String]) {
    val n_topics = 3

    // Look at configuration to determine parallelism and run node
    val conf = new SparkConf().setAppName("LDA test")
    val sc = new SparkContext(conf)

    // Load and parse the data
    val data = sc.textFile("data/corpus.joris")

    // Cache for faster processing
    val cachedCorpus = parseCustomCsv(data).cache()
    // Cluster the documents into three topics using LDA
    val ldaModel = new LDA().setK(n_topics).run(cachedCorpus)

    // Save as CSV for other groups
    saveAsCsv("data/lda_matrix_" + n_topics + ".csv", ldaModel.topicsMatrix)

    // Save as parquet for further processing
    //    ldaModel.save(sc, "myLDAModel")
  }

  def saveAsCsv(filename: String, matrix: Matrix) = {
    val wr = new BufferedWriter(new FileWriter(filename))
    for (i : Int <- 0 until matrix.numCols) {
      if (i > 0) wr.write("\n")
      for (j : Int <- 0 until matrix.numRows) {
        if (j > 0) wr.write(",")
        wr.write(matrix(j, i).toString)
      }
    }
    wr.close()
  }

  def parseCustomCsv(data: RDD[String]) : RDD[(Long, Vector)] = {
    // Parse the header and disregard the first line (comment)
    val header = data.take(2)(1).split(' ')
    // The second line contains the number of values
    val n_words = header(0).toInt
    val n_docs = header(1).toLong

    // Parse each row except the header separately
    return data
      .mapPartitionsWithIndex((idx, part) => if (idx == 0) part.drop(2) else part)
      .filter(!_.isEmpty())
      .map(s => {
        val values = s.trim.split(';')
        val doc_id = values(0).toLong
        val words = values(1).split(',').map(_.toInt)
        val counts = values(2).split(',').map(_.toDouble)
        (doc_id, new SparseVector(n_words, words, counts))
      })
  }

  def parseMatrixMarket(data: RDD[String]) : RDD[(Long, Vector)] = {
    // Parse the header and disregard the first line (%%MatrixMarket etc)
    val header = data.take(2)(1).split(' ')
    // The second line contains the number of values
    val n_docs = header(0).toLong
    val n_words = header(1).toInt
    val n_counts = header(2).toLong

    // Parse each row except the header separately
    val sparseRows : RDD[(Long, (Int, Double))] = data
      .mapPartitionsWithIndex((idx, part) => if (idx == 0) part.drop(2) else part)
      .map(s => {
        val values = s.trim.split(' ')
        (values(0).toLong - 1, (values(1).toInt - 1, values(2).toDouble))
      })

    // Combine the rows by document ID, and store the words and counts in Arrays
    return sparseRows
      .groupByKey()
      .map(x => {
        val doc_id = x._1
        val iter = x._2
        val cts = Array[Double]().padTo(iter.size, 0.0)
        val wds = Array[Int]().padTo(iter.size, 0)
        var i = 0
        var ct : (Int, Double) = null
        for (ct <- iter) {
          wds(i) = ct._1
          cts(i) = ct._2
          i += 1
        }
        // TODO: Sparse vector assumes that wds is sorted, we don't check or enforce that yet
        (doc_id, new SparseVector(n_words, wds, cts))
      })
  }
}