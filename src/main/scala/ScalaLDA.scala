import java.io.{PrintWriter, BufferedWriter, FileWriter}

import org.apache.spark.mllib.clustering.{EMLDAOptimizer, OnlineLDAOptimizer, LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Matrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

import scopt.OptionParser

object ScalaLDA {
  private case class Params(
     input: String = null,
     output: String = null,
     k: Int = 20,
     maxIterations: Int = 10,
     docConcentration: Double = -1,
     topicConcentration: Double = -1,
     vocabSize: Int = 10000,
     algorithm: String = "em")

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("ScalaLDA") {
      head("ScalaLDA: an LDA app for plain text data.")
      opt[Int]("k")
        .text(s"number of topics. default: ${defaultParams.k}")
        .action((x, c) => c.copy(k = x))
      opt[Int]("maxIterations")
        .text(s"number of iterations of learning. default: ${defaultParams.maxIterations}")
        .action((x, c) => c.copy(maxIterations = x))
      opt[Double]("docConcentration")
        .text(s"amount of topic smoothing to use (> 1.0) (-1=auto)." +
          s"  default: ${defaultParams.docConcentration}")
        .action((x, c) => c.copy(docConcentration = x))
      opt[Double]("topicConcentration")
        .text(s"amount of term (word) smoothing to use (> 1.0) (-1=auto)." +
          s"  default: ${defaultParams.topicConcentration}")
        .action((x, c) => c.copy(topicConcentration = x))
      opt[String]("algorithm")
        .text(s"inference algorithm to use. em and online are supported." +
          s" default: ${defaultParams.algorithm}")
        .action((x, c) => c.copy(algorithm = x))
      arg[String]("<input>")
        .text("input path to plain text (indexed) corpus." +
          "  Each text file line should hold 1 document. The first line is a comment, the second line is " +
          "[number of documents]<space>[number of words] and the other lines are ordered as " +
          "doc_id;word_id_1,...,word_id_n;word_count_1,...,word_count_n")
        .required()
        .action((x, c) => c.copy(input = x))
      arg[String]("<output>")
        .text("output csv file with a matrix with word frequencies, with rows as topics and columns as word id")
        .required()
        .action((x, c) => c.copy(output = x))
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      parser.showUsageAsError
      sys.exit(1)
    }
  }
  def run(params: Params) {
    val n_topics = 100

    // Look at configuration to determine parallelism and run node
    val conf = new SparkConf()
      .setAppName("ScalaLDA")
      .set("spark.default.parallelism", "8")
    val sc = new SparkContext(conf)

    // Load and parse the data
    val data = sc.textFile(params.input)

    // Cache for faster processing
    val cachedCorpus = parseCustomCsv(data).cache()

    val optimizer = params.algorithm.toLowerCase match {
      case "em" => new EMLDAOptimizer
      // add (1.0 / actualCorpusSize) to MiniBatchFraction be more robust on tiny datasets.
      case "online" => new OnlineLDAOptimizer().setMiniBatchFraction(0.05 + 1.0 / cachedCorpus.count())
      case _ => throw new IllegalArgumentException(
        s"Only em, online are supported but got ${params.algorithm}.")
    }

    // Cluster the documents into three topics using LDA
    val ldaModel = new LDA()
      .setOptimizer(optimizer)
      .setK(params.k)
      .setMaxIterations(params.maxIterations)
      .setDocConcentration(params.docConcentration)
      .setTopicConcentration(params.topicConcentration)
      .run(cachedCorpus)

    // Save as CSV for other groups
    saveAsCsv(sc, params.output, ldaModel.topicsMatrix)

    // Save as parquet for further processing
    //    ldaModel.save(sc, "myLDAModel")
  }

  def saveAsCsv(sc: SparkContext, filename: String, matrix: Matrix) = {
    sc.makeRDD(Range(0, matrix.numCols), 1)
      .map(i => {
        var str = ""
        for (j : Int <- Range(0, matrix.numRows)) {
          if (j > 0) str += ","
          str += matrix(j, i)
        }
        str
      })
      .saveAsTextFile(filename)
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
