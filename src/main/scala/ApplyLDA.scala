import org.apache.spark.{SparkContext, SparkConf}

import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.mllib.linalg.{SparseVector, Vector, Matrix}
import org.apache.spark.rdd.RDD

import scopt.OptionParser

object ApplyLDA {
  private case class Params(
     ldaFile: String = null,
     bowFile: String = null,
     outFile: String = null
     )

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("ApplyLDA") {
      head("ApplyLDA: ...")
      arg[String]("<ldaFile>")
        .text("...")
        .required()
        .action((x, c) => c.copy(ldaFile = x))
      arg[String]("<bowFile>")
        .text("...")
        .required()
        .action((x, c) => c.copy(bowFile = x))
      arg[String]("<outFile>")
        .text("...")
        .required()
        .action((x, c) => c.copy(outFile = x))
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf()
      .setAppName("ApplyLDA")
      .set("spark.default.parallelism", "8")
    val sc = new SparkContext(conf)

    val modelName = params.ldaFile  //"data/enron_small_lda.csv.model"
    val dataFile = params.bowFile   //"data/enron_small_bow.csv/part-00000"
    val outFile = params.outFile    //"neoResponse.txt"
    val myModel = DistributedLDAModel.load(sc, modelName)
    val localModel = myModel.toLocal

    val data = sc.textFile(dataFile)
    val corpus = parseCustomCsv(data)

    val res = localModel.topicDistributions(corpus)
    res.saveAsTextFile("neoResponse.txt")
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
}
