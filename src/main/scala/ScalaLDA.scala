import FileUtil._
import org.apache.spark.mllib.clustering.{EMLDAOptimizer, LDA, OnlineLDAOptimizer}
import org.apache.spark.{SparkConf, SparkContext}
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
      sys.exit(1)
    }
  }
  def run(params: Params) {
    // Look at configuration to determine parallelism and run node
    val conf = new SparkConf()
      .setAppName("ScalaLDA")
      .set("spark.default.parallelism", "8")
    val sc = new SparkContext(conf)

    // Load and parse the data
    val data = sc.textFile(params.input)

    // Cache for faster processing
    val corpus = parseCustomCsv(data)

    val optimizer = params.algorithm.toLowerCase match {
      case "em" => new EMLDAOptimizer
      // add (1.0 / actualCorpusSize) to MiniBatchFraction be more robust on tiny datasets.
      case "online" => new OnlineLDAOptimizer().setMiniBatchFraction(0.05 + 1.0 / corpus.count())
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
      .run(corpus)

    // Save as CSV for other groups
    saveAsCsv(sc, params.output, ldaModel.topicsMatrix)

    // Save as parquet for further processing
    ldaModel.save(sc, params.output + ".model")
  }
}
